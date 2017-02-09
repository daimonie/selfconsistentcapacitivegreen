# Numerical Python
import numpy as np
# Scientific Python
import scipy as sp
# SP.Interpolate 
import scipy.interpolate as si
# SP.Optimize
from scipy.optimize import minimize
# SP.Constants contains physical constants.
from scipy.constants import physical_constants as pc
# Mostly to have access to the STDError stream, allowing output past a pipe
import sys  
# Argument parsing
import argparse as argparse    
# Allows tic/toc on the file execution.
import time
global_time_start = time.time();
###
physicalCurrentUnit   = pc["elementary charge"][0] / pc["Planck constant"][0] * pc["electron volt"][0];

# Tunnel 
tau = 0.010;
# Lead-Molecule coupling (symmetric)
gamma = 0.010;
# Stark effect strength
alpha = 0.00;
# Interaction strength
capacitive = 0.300;
# Zero-bias level. Slightly below zero to improve convergence.
levels = -1e-8 ;
# Integration interval for the self-consistent calculation.
intervalW = np.linspace( -1.0, 1.0, 1000);
# Applied bias over the junction.
bias = 0.0;

spinlessHamiltonian = np.zeros((2,2));
spinlessHamiltonian[0][0] = levels + 0.5 * alpha * bias;
spinlessHamiltonian[1][1] = levels - 0.5 * alpha * bias;

spinlessTunnellingMatrix = np.zeros((2,2));
spinlessTunnellingMatrix[0][1] = -tau;
spinlessTunnellingMatrix[1][0] = -tau;

spinlessInteraction = np.zeros((2,2));
spinlessInteraction[0][1] = capacitive;
spinlessInteraction[1][0] = capacitive;


spinlessGammaLeft = np.zeros((2,2));
spinlessGammaLeft[0][0] = gamma;

spinlessGammaRIght = np.zeros((2,2));
spinlessGammaRIght[1][1] = gamma;

