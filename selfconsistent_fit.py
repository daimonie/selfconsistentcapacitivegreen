import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter
from igf import *
import scipy.interpolate as si
from scipy.optimize import minimize
from scipy.constants import physical_constants as pc
### parameters
tolerance = 1e-3

#bias_res = 50
#biaswindow = np.linspace( -0.5, 0.5, bias_res )

biaswindow = np.array([-0.50, -0.40, -0.30, -0.20, -0.10, 0.00, 0.10, 0.20, 0.30, 0.40, 0.50])
bias_res = len(biaswindow)


old_chances = np.zeros( (4, bias_res ))
new_chances = np.zeros( (4, bias_res ))
biasnum = 0

#parameters from fit_0 
tau = 0.010
gamma = 0.010
alpha = 0.40
capacitive = 0.300
levels = -capacitive -1e-4
beta = 250.0

for bias in biaswindow:
    realscale   = pc["elementary charge"][0] / pc["Planck constant"][0] * pc["electron volt"][0]
    epsilon_res = 100

    epsilon_window = np.linspace(-1.0, 1.0, epsilon_res)



    hamiltonian = np.zeros((2,2))
    hamiltonian[0][0] = levels + 0.5 * alpha * bias
    hamiltonian[1][1] = levels - 0.5 * alpha * bias

    tunnel = np.zeros((2,2))
    tunnel[0][1] = -tau
    tunnel[1][0] = -tau

    interaction = np.zeros((2,2))
    interaction[0][1] = capacitive
    interaction[1][0] = capacitive


    gamma_left = np.zeros((2,2))
    gamma_left[0][0] = gamma

    gamma_right = np.zeros((2,2))
    gamma_right[1][1] = gamma

    calculation = igfwl(
        hamiltonian, 
        tunnel,
        interaction, 
        gamma_left,
        gamma_right, 
        beta
    )

    superset = calculation.generate_superset(0)
    ### Construct the K matrix
    print "Many body states"
    for i in superset:
        print i, calculation.ket(i)
    K = np.zeros((2, 4))

    for i in [0, 1]:
        for j in superset:
            if (i+1) & j == (i+1):
                K[i, j] = 1.0
    print "K matrix: \n", K

    ### Calculate all (lambda) Green's functions

    #must be like this because they are lambda functions
    zero_retarded,  zero_advanced = calculation.singleparticlebackground(0)
    one_retarded,   one_advanced = calculation.singleparticlebackground(1)
    two_retarded,   two_advanced = calculation.singleparticlebackground(2)
    three_retarded, three_advanced = calculation.singleparticlebackground(3) 

    fd = lambda epsilon: 1.0 / ( 1 + np.exp( epsilon * beta ))

    zero_lesser_left = lambda epsilon, i: fd(epsilon + 0.5 * bias) * gamma * zero_retarded(epsilon).item((i,0)) * zero_advanced(epsilon).item((0, i))
    zero_lesser_right = lambda epsilon, i: fd(epsilon - 0.5 * bias) * gamma * zero_retarded(epsilon).item((i,1)) * zero_advanced(epsilon).item((1, i))
    zero_lesser = lambda epsilon, i: np.real(zero_lesser_left( epsilon, i) + zero_lesser_right( epsilon, i))

    one_lesser_left = lambda epsilon, i: fd(epsilon + 0.5 * bias) * gamma * one_retarded(epsilon).item((i,0)) * one_advanced(epsilon).item((0, i))
    one_lesser_right = lambda epsilon, i: fd(epsilon - 0.5 * bias) * gamma * one_retarded(epsilon).item((i,1)) * one_advanced(epsilon).item((1, i))
    one_lesser = lambda epsilon, i: np.real(one_lesser_left( epsilon, i) + one_lesser_right( epsilon, i))

    two_lesser_left = lambda epsilon, i: fd(epsilon + 0.5 * bias) * gamma * two_retarded(epsilon).item((i,0)) * two_advanced(epsilon).item((0, i))
    two_lesser_right = lambda epsilon, i: fd(epsilon - 0.5 * bias) * gamma * two_retarded(epsilon).item((i,1)) * two_advanced(epsilon).item((1, i))
    two_lesser = lambda epsilon, i: np.real(two_lesser_left( epsilon, i) + two_lesser_right( epsilon, i))

    three_lesser_left = lambda epsilon, i: fd(epsilon + 0.5 * bias) * gamma * three_retarded(epsilon).item((i,0)) * three_advanced(epsilon).item((0, i))
    three_lesser_right = lambda epsilon, i: fd(epsilon - 0.5 * bias) * gamma * three_retarded(epsilon).item((i,1)) * three_advanced(epsilon).item((1, i))
    three_lesser = lambda epsilon, i: np.real(three_lesser_left( epsilon, i) + three_lesser_right( epsilon, i))

    ### Calculate initial guess

    P0 = calculation.distribution()
    P = P0 
    ### define function to calculate the vector of single-particle number operator
    ###     expectation values from the Green's functions and a probability vector
    def lesser_number_vector ( P ): 
        number_vector = [0.0, 0.0]
        for k in superset:
            for l in calculation.generate_superset(k):
                ket_l =  calculation.ket(l)
                #if l == 0:
                    #print "Zero contains nothing"
                if l == 1:
                    if ket_l[0] == 1.0:
                        number_vector[0] += P[l] * np.trapz([one_lesser(e, 0) for e in epsilon_window], epsilon_window)
                    if ket_l[1] == 1.0:
                        number_vector[1] += P[l] * np.trapz([one_lesser(e, 1) for e in epsilon_window], epsilon_window)
                elif l == 2:
                    if ket_l[0] == 1.0:
                        number_vector[0] += P[l] * np.trapz([two_lesser(e, 0) for e in epsilon_window], epsilon_window)
                    if ket_l[1] == 1.0:
                        number_vector[1] += P[l] * np.trapz([two_lesser(e, 1) for e in epsilon_window], epsilon_window)
                elif l == 3:
                    if ket_l[0] == 1.0:
                        number_vector[0] += P[l] * np.trapz([three_lesser(e, 0) for e in epsilon_window], epsilon_window)
                    if ket_l[1] == 1.0:
                        number_vector[1] += P[l] * np.trapz([three_lesser(e, 1) for e in epsilon_window], epsilon_window)
                    
        number_vector = np.array(number_vector)
        number_vector /= np.sum(number_vector)
        return np.abs(number_vector)
        
    ### self-consistency loop

    difference = 1.00
    print "Start self-consistency loop"
    generation = 0
    while difference > tolerance: 
        new_number_vector = lesser_number_vector(P) 
        
        print "number\t%d\t%.4f\t%.4f" % (generation, new_number_vector[0], new_number_vector[1])
        
        number_length = lambda P: np.sum( np.square( np.dot(K, np.abs(P)) - new_number_vector))
        
        res = minimize( number_length, P, method='nelder-mead' )
        
        P_previous = P
        P = res.x / np.sum(res.x)
        
        difference = np.sum( np.square( (P - P_previous))) / np.sum( np.square( P_previous))   

        #difference = 0
        generation += 1
    print "At bias %.3f, beta %.3f:" % (bias, beta)
    print "Initial probability vector:"
    print "%.4f\t%.4f\t%.4f\t%.4f" % (P0[0], P0[1], P0[2], P0[3])
    print "Final probability vector:"
    print "%.4f\t%.4f\t%.4f\t%.4f" % (P[0], P[1], P[2], P[3])
    
    for i in superset:
        new_chances[i][biasnum] = P[i]
        old_chances[i][biasnum] = P0[i]
    biasnum += 1
    

print "Plotting results" 
##plotting
fig = plt.figure(figsize=(20, 20), dpi=1080)
ax = fig.add_subplot(111)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
 
xlabel = "Bias voltage $V$ [eV]"
ylabel = "Occupation chance $P$"
plt.rc('font', family='serif')

plt.plot(biaswindow, new_chances[0][:], 'rd', label='self-consistent $P_{00}$', markersize=10) 
plt.plot(biaswindow, new_chances[1][:], 'gd', label='self-consistent $P_{10}$', markersize=10) 
plt.plot(biaswindow, new_chances[2][:], 'bd', label='self-consistent $P_{01}$', markersize=10) 
plt.plot(biaswindow, new_chances[3][:], 'md', label='self-consistent $P_{11}$', markersize=10) 


plt.plot(biaswindow, old_chances[0][:], 'r--', label='Boltzmann $P_{00}$', linewidth=4) 
plt.plot(biaswindow, old_chances[1][:], 'g--', label='Boltzmann $P_{10}$', linewidth=4) 
plt.plot(biaswindow, old_chances[2][:], 'b--', label='Boltzmann $P_{01}$', linewidth=4) 
plt.plot(biaswindow, old_chances[3][:], 'm--', label='Boltzmann $P_{11}$', linewidth=4) 

minorLocator1 = AutoMinorLocator(5)
minorLocator2 = AutoMinorLocator(5)
ax.xaxis.set_minor_locator(minorLocator1) 
ax.yaxis.set_minor_locator(minorLocator2) 

ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.tick_params(which='both', width=2)
plt.tick_params(which='major', length=20)
plt.tick_params(which='minor', length=10)

if len(biaswindow) < 10:
    plt.xticks([-.5, -.25, 0.00, .25, .50])
else:
    plt.xticks(biaswindow)
plt.yticks([-0.10, 0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10])

plt.xlabel(xlabel, fontsize=30)
plt.ylabel(ylabel, fontsize=30)

#plt.title( "%s" % (title), fontsize=15)       
plt.legend(loc='upper right') 

plt.savefig('selfconsistent.pdf') 

