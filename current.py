import numpy as np
import sys 
from igf import *
from experiment import *
from paralleltask import *
import scipy.interpolate as si
from scipy.optimize import minimize
from scipy.constants import physical_constants as pc
import argparse as argparse  
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter
### parameters
import time
global_time_start = time.time()

tolerance = 1e-5 
bias_res = 250

biaswindow = np.linspace( -.175, .175, bias_res) 

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
  
##plotting
mode = 1
cores = 4  
###  
for this_bias in biaswindow:    
    epsilon_res = 250  
    
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
    spinless_calculation.label = "self consistent bias %.3f" % bias
    spinless_calculation.calculate_number_matrix_k()
    spinless_calculation.calculate_number_matrix_w( -bias, np.linspace( -1.0, 1.0, 1000))
    ### self-consistency loop
    P = spinless_calculation.selfconsistent_distribution(tolerance)
    P /= np.sqrt( np.sum(np.square(P)))

    spinless_calculation.set_distribution(P)
    epsilon = np.linspace(-bias/2.0, bias/2.0, epsilon_res);
    spinless_transmission = spinless_calculation.full_transmission(epsilon)
    spinless_current = -realscale*np.trapz(spinless_transmission, epsilon) 
     
    print >> sys.stderr,  P 
    results.append( [bias, spinless_current, P[0], P[1], P[2], P[3]])
    biasnum += 1

results = np.array(results);



#### moving on
calculated_bias = results[:,0]
calculated_current = results[:,1]/1e-9

for i in range( calculated_bias.shape[0] ):
    print "%.3f\t%.3f" % ( calculated_bias[i], calculated_current[i] )

#######################################
print >> sys.stderr,   "Plotting new calculation versus experiment..."

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
fig = plt.figure(figsize=(25, 15), dpi=1080) 
ax = fig.add_subplot(211)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
fig.subplots_adjust(left=0.10, hspace=0.50) 
plt.rc('font', family='serif')


print >> sys.stderr, "Max experimental current %2.3e" % np.max(np.abs(experimental))
print >> sys.stderr, "Max calculated  current %2.3e" % np.max(np.abs(calculated_current))

make_same_scale = np.max(experimental) / np.max(calculated_current)
calculated_current *= make_same_scale

plt.plot(bias, experimental, 'm-', label='Experimental result', linewidth=3)  
plt.plot(calculated_bias, calculated_current, 'rd', label='SCI spinless two-site model', markersize=12)  
 
ax.set_title("Scaled $I(V)$ by $%.2f$, $\\tau=%.3f, \\gamma=%.3f, \\alpha=%.3f, \\epsilon_0=%.3f, U=%.3f, \\beta=%.1f$" % (make_same_scale, tau, gamma, alpha, levels, capacitive, beta), fontsize=20)

plt.legend(bbox_to_anchor=(0., 1.08, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., fontsize=30)
 
xlabel = "Bias $V_b$ [eV]"
ylabel = "Current $I(V_b)$  [nA] "
 
plt.xlim([-.2, .2]) 
plt.xlabel(xlabel, fontsize=30)
plt.ylabel(ylabel, fontsize=30)
 
plt.xticks(np.linspace( -.2, .2, 9)) 

minorLocator1 = AutoMinorLocator(5)
minorLocator2 = AutoMinorLocator(5)
ax.xaxis.set_minor_locator(minorLocator1) 
ax.yaxis.set_minor_locator(minorLocator2) 

ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.tick_params(which='both', width=2)
plt.tick_params(which='major', length=20)
plt.tick_params(which='minor', length=10)

#prevents lagg on desktop
time.sleep(1)

########## subplot 2
ax = fig.add_subplot(212) 


plt.xticks(fontsize=25)
plt.yticks(fontsize=25) 
plt.rc('font', family='serif')  

print >> sys.stderr, results[:,2]
print >> sys.stderr,  results[:,3]
print >> sys.stderr,  results[:,4]
print >> sys.stderr,  results[:,5]
ax.plot(calculated_bias, results[:, 2], 'r-', label='$P_{00}$', linewidth=3)  
ax.plot(calculated_bias, results[:, 3], 'g-', label='$P_{01}$', linewidth=3)  
ax.plot(calculated_bias, results[:, 4], 'b-', label='$P_{10}$', linewidth=3)  
ax.plot(calculated_bias, results[:, 5], 'm-', label='$P_{11}$', linewidth=3)  
 
plt.legend(bbox_to_anchor=(0., 1.04, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0., fontsize=25)
 
xlabel = "Bias $V_b$ [eV]"
ylabel = "Occupation Probability"
 
plt.xlim([-.2, .2]) 
plt.ylim([-.1, 1.1])
plt.xlabel(xlabel, fontsize=30)
plt.ylabel(ylabel, fontsize=30)
 
plt.xticks(np.linspace( -.2, .2, 9)) 

minorLocator1 = AutoMinorLocator(5)
minorLocator2 = AutoMinorLocator(5)
ax.xaxis.set_minor_locator(minorLocator1) 
ax.yaxis.set_minor_locator(minorLocator2) 

ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.tick_params(which='both', width=2)
plt.tick_params(which='major', length=20)
plt.tick_params(which='minor', length=10)
#prevents lagg on desktop
time.sleep(1)
################################

plt.savefig('current.svg')
global_time_end = time.time ()
print >> sys.stderr,   "\n Time spent %.6f seconds. \n " % (global_time_end - global_time_start)
