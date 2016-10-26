import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter
from igf import *
from experiment import *
from paralleltask import *
import scipy.interpolate as si
from scipy.optimize import minimize
from scipy.constants import physical_constants as pc
import argparse as argparse  
### parameters
import time
global_time_start = time.time()
tolerance = 1e-3

bias_res = 40

biaswindow = []
for bias in np.linspace( -0.25, -0.07, 5 ):
    biaswindow.append(bias)
for bias in np.linspace( -0.07, 0.07, bias_res ):
    biaswindow.append(bias)
for bias in np.linspace( -0.07, 0.25, 5 ):
    biaswindow.append(bias)
bias_res = len(biaswindow)
biaswindow = np.array(biaswindow)



old_chances = np.zeros( (4, bias_res ))
new_chances = np.zeros( (4, bias_res ))
biasnum = 0

realscale   = pc["elementary charge"][0] / pc["Planck constant"][0] * pc["electron volt"][0]
epsilon_res = 100

epsilon_window = np.linspace(-1.0, 1.0, epsilon_res)

#parameters from fit_0 are default
default_tau = 0.010
default_gamma = 0.010
default_alpha = 0.40
default_capacitive = 0.300
default_levels = -default_capacitive -1e-4 
default_beta = 250
#parameter parser
parser  = argparse.ArgumentParser(prog="current map parallel",
  description = "Self consistent current calculation, also plots experimental current.")  
 
parser.add_argument(
    '-t',
    '--tau',
    help='Tunnelling strength',
    action='store',
    type = float,
    default = default_tau
)   
parser.add_argument(
    '-gt',
    '--gamma',
    help='Molecule-lead coupling strength',
    action='store',
    type = float,
    default = default_gamma
)   
parser.add_argument(
    '-at',
    '--alpha',
    help='Bias-level coupling',
    action='store',
    type = float,
    default = default_alpha
)   
parser.add_argument(
    '-u',
    '--capacitive',
    help='Capacitive interaction strength',
    action='store',
    type = float,
    default = default_capacitive
)   
parser.add_argument(
    '-e',
    '--epsilon',
    help='Zero-bias level',
    action='store',
    type = float,
    default = default_levels
)   
parser.add_argument(
    '-b',
    '--beta',
    help='inverse temperature',
    action='store',
    type = float,
    default = default_beta
)   
args    = parser.parse_args() 
tau = args.tau
gamma = args.gamma
alpha = args.alpha
capacitive = args.capacitive
levels = args.epsilon
beta = args.beta 

for bias in biaswindow:
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
        
        print "number\t%d\t%.4e\t%.4e" % (generation, new_number_vector[0], new_number_vector[1])
        
        number_length = lambda P: np.sum( np.square( np.dot(K, np.abs(P)) - new_number_vector))
        
        res = minimize( number_length, P, method='nelder-mead' )
        
        P_previous = P
        P = res.x / np.sum(res.x)
        
        difference = np.sum( np.square( (P - P_previous))) / np.sum( np.square( P_previous))   

        #difference = 0
        generation += 1
    print "At bias %.3f, beta %.3f:" % (bias, beta)
    
    P[P < 0.001] = 0.0
    P /= np.sum(P)
    
    print "Initial probability vector:"
    print "%.4f\t%.4f\t%.4f\t%.4f" % (P0[0], P0[1], P0[2], P0[3])
    print "Final probability vector:"
    print "%.4f\t%.4f\t%.4f\t%.4f" % (P[0], P[1], P[2], P[3])
     
    for i in superset:
        new_chances[i][biasnum] = P[i]
        old_chances[i][biasnum] = P0[i]
    biasnum += 1
    

print "Plotting current" 
##plotting
mode = 1
cores = 4

#separation_array = range(638, 670)        
#data_bias = np.zeros(( len(separation_array)-1, 404))
#data_current = np.zeros(( len(separation_array)-1, 404))
separation_array = range(650, 670)        
data_bias = np.zeros(( len(separation_array), 404))
data_current = np.zeros(( len(separation_array), 404))

i = 0          
for sep in separation_array:
    if sep != 645:
        file = "exp_data/IV130328_7_%d.dat" % sep
        #print "Reading [%s]" % file
        bias, current = read_experiment(file)
     
        filter = np.ones(15)/15.0
        current = np.convolve(current, filter, mode='same')
         
        bias_array = bias 
                
        data_bias[i]    = bias
        data_current[i] = current
        
        i += 1
###
bias         = data_bias.mean(axis=0)
experimental = data_current.mean(axis=0) / 1e-9
###  
def calculate_spinless(arguments):  
    epsilon_res = 250 
    bias        = arguments[0]
    alpha       = arguments[1]
    tau         = arguments[2]
    gamma       = arguments[3]
    capacitive  = arguments[4]
    beta        = arguments[5]
    levels      = arguments[6]
    biasnum     = arguments[7] 
    
    realscale   = pc["elementary charge"][0] / pc["Planck constant"][0] * pc["electron volt"][0]

    spinless_hamiltonian = np.zeros((2,2))
    spinless_hamiltonian[0][0] = levels + 0.5 * alpha * bias
    spinless_hamiltonian[1][1] = levels - 0.5 * alpha * bias

    spinless_tunnel = np.zeros((2,2))
    spinless_tunnel[0][1] = -tau
    spinless_tunnel[1][0] = -tau

    spinless_interaction = np.zeros((2,2))
    spinless_interaction[0][1] = capacitive
    spinless_interaction[1][0] = capacitive


    spinless_gamma_left = np.zeros((2,2))
    spinless_gamma_left[0][0] = gamma

    spinless_gamma_right = np.zeros((2,2))
    spinless_gamma_right[1][1] = gamma

    spinless_calculation = igfwl(
        spinless_hamiltonian, 
        spinless_tunnel,
        spinless_interaction, 
        spinless_gamma_left,
        spinless_gamma_right, 
        beta
    )
    spinless_calculation.set_distribution( [new_chances[0][biasnum], new_chances[1][biasnum], new_chances[2][biasnum], new_chances[3][biasnum]] )
    
    epsilon = np.linspace(-bias/2.0, bias/2.0, epsilon_res);
  
    spinless_transmission = spinless_calculation.full_transmission(epsilon)
    spinless_current = realscale*np.trapz(spinless_transmission, epsilon) 
     
    
    return [bias, spinless_current]


#### calculate current

manager = taskManager( cores, calculate_spinless )  
biasnum = 0
for this_bias in biaswindow:       
    manager.add_params([this_bias, alpha, tau, gamma, capacitive, beta, levels, biasnum])  
    biasnum += 1
    
manager.execute()
results = manager.final()
results = np.array(results)

calculated_bias = results[:,0]
calculated_current = results[:,1]/1e-9

#######################################
fig = plt.figure(figsize=(20, 20), dpi=1080)
ax = fig.add_subplot(111)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
fig.subplots_adjust(left=0.30)

title = "Dummy title"
xlabel = ""
ylabel = ""
plt.rc('font', family='serif')


print "Max current %2.3e" % np.max(np.abs(calculated_current))

make_same_scale = np.max(experimental) / np.max(calculated_current)
calculated_current *= make_same_scale

plt.plot(bias, experimental, 'm-', label='Experimental Average')  
plt.plot(calculated_bias, calculated_current, 'rd', label='SCI spinless two-site model', markersize=12)  
 
ax.set_title("Scaled $I(V)$ by $%.2f$, $\\tau=%.3f, \\gamma=%.3f, \\alpha=%.3f, \\epsilon_0=%.3f, U=%.3f$" % (make_same_scale, tau, gamma, alpha, levels, capacitive), fontsize=20)

plt.legend(bbox_to_anchor=(0., 1.04, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., fontsize=25)

 
xlabel = "Bias $V_b$ [eV]"
ylabel = "Current $I(V_b)$  [nA] "
 
plt.xlim([-0.25, 0.25])
#plt.xlim([-1., 1.])
plt.xlabel(xlabel, fontsize=30)
plt.ylabel(ylabel, fontsize=30)
 
plt.xticks(np.array(range(11))*0.05-0.25) 

minorLocator1 = AutoMinorLocator(5)
minorLocator2 = AutoMinorLocator(5)
ax.xaxis.set_minor_locator(minorLocator1) 
ax.yaxis.set_minor_locator(minorLocator2) 

ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.tick_params(which='both', width=2)
plt.tick_params(which='major', length=20)
plt.tick_params(which='minor', length=10)

print "Plotting"
plt.savefig('selfconsistent_fit_current.pdf')
global_time_end = time.time ()
print "\n Time spent %.6f seconds. \n " % (global_time_end - global_time_start)