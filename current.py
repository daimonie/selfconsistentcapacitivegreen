import numpy as np
import sys 
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
tolerance = 1e-9

bias_res = 15

biaswindow = []

lowresborder = 0.10

for bias in np.linspace( -0.25, -lowresborder , 5 ):
    biaswindow.append(bias)
for bias in np.linspace( -lowresborder , lowresborder , bias_res ):
    biaswindow.append(bias)
for bias in np.linspace( -lowresborder , 0.25, 5 ):
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
    calculation.calculate_number_matrix_k()
    calculation.calculate_number_matrix_w( 0.35, np.linspace( -1.0, 1.0, 1000))
    ### self-consistency loop
    P = calculation.selfconsistent_distribution(tolerance)
    for i in superset:
        new_chances[i][biasnum] = P[i]
    biasnum += 1
       
##plotting
mode = 1
cores = 4  
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
    spinless_calculation.set_distribution(new_chances[:, biasnum])
    epsilon = np.linspace(-bias/2.0, bias/2.0, epsilon_res);
  
    spinless_transmission = spinless_calculation.full_transmission(epsilon)
    spinless_current = realscale*np.trapz(spinless_transmission, epsilon) 
     
    
    return [bias, spinless_current]

print "Calculating current..."
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

for i in range( calculated_bias.shape[0]):
    print "%.3f\t%.3f" % ( calculated_bias[i], calculated_current[i] )


