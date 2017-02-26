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
# Easy access to nullspace calculation
from sympy import Matrix
from sympy import matrix2numpy
# Mostly to have access to the STDError stream, allowing output past a pipe
import sys  
#some small functions I use
sys.path.append('/home/daimonie/ssd/git/PythonUtils')
from utils import *
# Argument parsing
import argparse as argparse    
#supress warnings
import warnings
# Allows tic/toc on the file execution.
import time
global_time_start = time.time();

#Feedback
print >> sys.stderr, "Distribution for Coulomb Island.\nSetting parameters.\n";  
# Units of the current.
physicalCurrentUnit   = pc["elementary charge"][0] / pc["Planck constant"][0] * pc["electron volt"][0];
# Lead-Molecule coupling (symmetric)
gamma = 0.010;
# Stark effect strength
alpha = 0.00;
# Interaction strength
capacitive = 0.300;
# Zero-bias level. Slightly below zero to improve convergence.
levels = -1e-9;
# Integration interval for the self-consistent calculation.
intervalW = np.linspace( -10.0, 10.0, 1e4);
# bias
bias = 0.0
# Temperature (units U); number, min, max
betaNumber = 25;
betaMin = 1e-2 ;
betaMax = 2.0;
betaArray = np.linspace(betaMin, betaMax, betaNumber); 

betaArray = np.array( [1e-3/capacitive] );

doInv = 1; 
###    
print >> sys.stderr, "Setting system matrices.\n";
#system matrices
hamiltonian = np.zeros((2,2));
hamiltonian[0][0] = levels;
hamiltonian[1][1] = levels;

interactionKet01 = np.zeros((2,2));
interactionKet01[1][1] = capacitive; 

interactionKet10 = np.zeros((2,2));
interactionKet10[0][0] = capacitive; 

interactionKet11 = interactionKet01 + interactionKet10;

gammaLeft = np.zeros((2,2));
gammaLeft[0][0] = gamma;

gammaRight = np.zeros((2,2));
gammaRight[1][1] = gamma;


print >> sys.stderr, "Setting Green's function lambda's.\n";
selfEnergy = 0.5j*(gammaLeft + gammaRight);
# single-particle Green's Functions. The numbers denote the state, ket{n_1 n_2}.
# G^{lambda+}
singleParticleGreensFunctionKet00 = lambda epsilon: np.linalg.inv( np.eye( 2 ) * epsilon - hamiltonian + selfEnergy);
singleParticleGreensFunctionKet01 = lambda epsilon: np.linalg.inv( np.linalg.inv(singleParticleGreensFunctionKet00(epsilon)) - interactionKet01);
singleParticleGreensFunctionKet10 = lambda epsilon: np.linalg.inv( np.linalg.inv(singleParticleGreensFunctionKet00(epsilon)) - interactionKet10);
singleParticleGreensFunctionKet11 = lambda epsilon: np.linalg.inv( np.linalg.inv(singleParticleGreensFunctionKet00(epsilon)) - interactionKet11);


# inverse temperature
betaIteration = 0;
for betaFraction in betaArray:
	beta = betaFraction*capacitive;
	if doInv:
		beta = (beta)**(-1.);
	print >> sys.stderr, "Calculation for beta=%.3e (%.3e U). Progress: %d/%d ." % (beta, beta/capacitive, betaIteration, betaNumber)
	betaIteration += 1
	# Fermi-Dirac distribution

	boltzmann = lambda epsilon: np.exp(-beta*epsilon);
	fd = lambda epsilon: 1.0 / (1.0+np.exp(-beta*epsilon)); 



	# Cached Integrals. Adding 0j makes sure that the datatype is complex. Otherwise, it casts to real later.
	integralLesserKet00 = np.zeros((2,2)) +0j;
	integralLesserKet01 = np.zeros((2,2)) +0j;
	integralLesserKet10 = np.zeros((2,2)) +0j;
	integralLesserKet11 = np.zeros((2,2)) +0j;

	print >> sys.stderr, "Calculating lesser integrals.\n";
	# Conversion for integral over W in the self-consistent equation
	# NB: sum Gamma_alpha = gamma I
	factorW = 1./(2.*np.pi)*gamma +0j ; 
	# Actual integration
	for i in range(2):

		with warnings.catch_warnings():
			warnings.simplefilter("ignore");

			occupancy = lambda epsilon: 0.5 * (fd(epsilon - bias/2.)+ fd(epsilon + bias/2.));

			integralLesserKet00[i][i] += factorW * np.trapz( [occupancy(epsilon)*np.abs(singleParticleGreensFunctionKet00(epsilon).item(0, i))**2 for epsilon in intervalW], intervalW)
			integralLesserKet01[i][i] += factorW * np.trapz( [occupancy(epsilon)*np.abs(singleParticleGreensFunctionKet01(epsilon).item(0, i))**2 for epsilon in intervalW], intervalW)
			integralLesserKet10[i][i] += factorW * np.trapz( [occupancy(epsilon)*np.abs(singleParticleGreensFunctionKet10(epsilon).item(0, i))**2 for epsilon in intervalW], intervalW)
			integralLesserKet11[i][i] += factorW * np.trapz( [occupancy(epsilon)*np.abs(singleParticleGreensFunctionKet11(epsilon).item(0, i))**2 for epsilon in intervalW], intervalW)

			integralLesserKet00[i][i] += factorW * np.trapz( [occupancy(epsilon)*np.abs(singleParticleGreensFunctionKet00(epsilon).item(i, 1))**2 for epsilon in intervalW], intervalW)
			integralLesserKet01[i][i] += factorW * np.trapz( [occupancy(epsilon)*np.abs(singleParticleGreensFunctionKet01(epsilon).item(i, 1))**2 for epsilon in intervalW], intervalW)
			integralLesserKet10[i][i] += factorW * np.trapz( [occupancy(epsilon)*np.abs(singleParticleGreensFunctionKet10(epsilon).item(i, 1))**2 for epsilon in intervalW], intervalW)
			integralLesserKet11[i][i] += factorW * np.trapz( [occupancy(epsilon)*np.abs(singleParticleGreensFunctionKet11(epsilon).item(i, 1))**2 for epsilon in intervalW], intervalW)
			#
		#
	#
	integralLesserKet00 = np.real(integralLesserKet00);
	integralLesserKet01 = np.real(integralLesserKet01);
	integralLesserKet10 = np.real(integralLesserKet10);
	integralLesserKet11 = np.real(integralLesserKet11); 
	#K matrix
	kappaMatrix = np.zeros((2,4)) +0j;
	kappaMatrix[0][1] = 1;
	kappaMatrix[0][3] = 1;
	kappaMatrix[1][2] = 1;
	kappaMatrix[1][3] = 1;

	print >> sys.stderr, "Calculating omega matrix.\n"
	#W matrix
	omegaMatrix = np.zeros((2,4)) +0j;

	# w for n_1 and kappa=00
	omegaMatrix[0][0] = integralLesserKet01[0][0] + integralLesserKet11[0][0];
	omegaMatrix[1][0] = integralLesserKet10[1][1] + integralLesserKet11[1][1];

	# w for n_1 and kappa=01
	omegaMatrix[0][1] = integralLesserKet01[0][0] + integralLesserKet11[0][0];
	omegaMatrix[1][1] = integralLesserKet11[1][1];

	# w for n_1 and kappa=10
	omegaMatrix[0][2] = integralLesserKet11[0][0];
	omegaMatrix[1][2] = integralLesserKet10[1][1] + integralLesserKet11[1][1];

	# w for n_1 and kappa=11
	omegaMatrix[0][3] = integralLesserKet11[0][0];
	omegaMatrix[1][3] = integralLesserKet11[1][1];

	# Solve kappaMatrix P = omegaMatrix P
	initialGuess = np.zeros((4));
	initialGuess[0] = boltzmann(0); #00 
	initialGuess[1] = boltzmann(levels); #01 
	initialGuess[2] = boltzmann(levels); #10 
	initialGuess[3] = boltzmann(levels*2 + capacitive); #11 
	selfConsistentProbabilityVector = initialGuess

	initialGuess = initialGuess / sum(initialGuess);
	#self-consistent equation f(x) = x
	# error is E(x) = | f(x) - x |^2
	numberError = lambda p: np.sum(np.abs( np.dot( omegaMatrix, p ) - np.dot( kappaMatrix, p ))**2)

	useMinimisation = 0;

	print >> sys.stderr, "kappaMatrix:";
	printMatrix(kappaMatrix);

	print >> sys.stderr, "omegaMatrix:"; 
	printMatrix(omegaMatrix);

	densityTransform = Matrix(-(omegaMatrix - kappaMatrix));
	nullSpace = densityTransform.nullspace();
	nullSpaceList = [];

	i = 0;
	for nullVector in nullSpace:
		nullSpaceList.append(matrix2numpy(nullVector)[:, 0]);
 
		nullSpaceList[i] /= np.sum( nullSpaceList[i]); 
		print np.dot( kappaMatrix, nullSpaceList[-1]);

		i += 1;

	nullSpaceList = np.array(nullSpaceList);

	print >> sys.stderr, "nullSpace:";
	printMatrix( nullSpaceList);

	raise Exception("Abort now.");