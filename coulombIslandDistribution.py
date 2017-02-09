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
# Units of the current.
physicalCurrentUnit   = pc["elementary charge"][0] / pc["Planck constant"][0] * pc["electron volt"][0];
# Conversion for integral over W in the self-consistent equation
factorW = 1./(2.*np.pi) +0j;
# Lead-Molecule coupling (symmetric)
gamma = 0.010;
# Stark effect strength
alpha = 0.00;
# Interaction strength
capacitive = 0.300;
# Zero-bias level. Slightly below zero to improve convergence.
levels = -1e-8 ;
# Integration interval for the self-consistent calculation.
intervalW = np.linspace( -10.0, 10.0, 1e4);
# Applied bias over the junction.
bias = 0.0;

hamiltonian = np.zeros((2,2));
hamiltonian[0][0] = levels;
hamiltonian[1][1] = levels;
 

interactionKet01 = np.zeros((2,2));
interactionKet01[1][1] = capacitive; 

interactionKet10 = np.zeros((2,2));
interactionKet10[0][0] = capacitive; 


gammaLeft = np.zeros((2,2));
gammaLeft[0][0] = gamma;

gammaRight = np.zeros((2,2));
gammaRight[1][1] = gamma;

selfEnergy = 0.5j*(gammaLeft + gammaRight);
# single-particle Green's Functions. The numbers denote the state, ket{n_1 n_2}.

singleParticleGreensFunctionKet00 = lambda epsilon: np.linalg.inv( np.eye( 2 ) * epsilon - hamiltonian - selfEnergy);
singleParticleGreensFunctionKet01 = lambda epsilon: np.linalg.inv( np.linalg.inv(singleParticleGreensFunctionKet00(epsilon)) - interactionKet01);
singleParticleGreensFunctionKet10 = lambda epsilon: np.linalg.inv( np.linalg.inv(singleParticleGreensFunctionKet00(epsilon)) - interactionKet10);
singleParticleGreensFunctionKet11 = lambda epsilon: np.linalg.inv( np.linalg.inv(singleParticleGreensFunctionKet00(epsilon)) - interactionKet10 - interactionKet01);

# Cached Integrals. Adding 0j makes sure that the datatype is complex. Otherwise, it casts to real later.
singleParticleLesserKet00 = np.zeros((2,2)) +0j;
singleParticleLesserKet01 = np.zeros((2,2)) +0j;
singleParticleLesserKet10 = np.zeros((2,2)) +0j;
singleParticleLesserKet11 = np.zeros((2,2)) +0j;

# Actual integration
for i in range(2):
	for j in range(2): # this is where it would cast to real otherwise.
		singleParticleLesserKet00[i][j] = factorW * np.trapz( [singleParticleGreensFunctionKet00(epsilon).item(i, j) for epsilon in intervalW], intervalW)
		singleParticleLesserKet01[i][j] = factorW * np.trapz( [singleParticleGreensFunctionKet01(epsilon).item(i, j) for epsilon in intervalW], intervalW)
		singleParticleLesserKet10[i][j] = factorW * np.trapz( [singleParticleGreensFunctionKet10(epsilon).item(i, j) for epsilon in intervalW], intervalW)
		singleParticleLesserKet11[i][j] = factorW * np.trapz( [singleParticleGreensFunctionKet11(epsilon).item(i, j) for epsilon in intervalW], intervalW)

#K matrix
kappaMatrix = np.zeros((2,4)) +0j;
kappaMatrix[0][1] = 1;
kappaMatrix[0][3] = 1;
kappaMatrix[1][2] = 1;
kappaMatrix[1][3] = 1;

#W matrix

#Self-consistency equation
