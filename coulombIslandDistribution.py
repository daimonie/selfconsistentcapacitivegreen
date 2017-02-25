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

#Feedback
print >> sys.stderr, "Distribution for Quantum Dot.\nSetting parameters.\n";  
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
bias = 0.05
# Temperature (units U); number, min, max
betaNumber = 200;
betaMin = -4;
betaMax = 2;
###
betaFractionArray = np.zeros((betaNumber));
i = 0;
for power in np.linspace(betaMin, betaMax, betaNumber):
	betaFractionArray[i] = 10**(power);
	i += 1;
betaFractionArray = betaFractionArray[::-1];


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
for betaFraction in betaFractionArray:
	beta = capacitive*(1e-9 + betaFraction*capacitive)**(-1); # Haug & Jauho neatly show a table that uses temperatures proportional to U
	print >> sys.stderr, "Calculation for beta=%.3e (%.3e U). Progress: %d/%d ." % (beta, betaFraction, betaIteration, betaNumber)
	betaIteration += 1
	# Fermi-Dirac distribution

	boltzmann = lambda epsilon: np.exp(-beta*epsilon);
	fd = lambda epsilon: 1 / (1+np.exp(-beta*epsilon)); 



	# Cached Integrals. Adding 0j makes sure that the datatype is complex. Otherwise, it casts to real later.
	integralLesserKet00 = np.zeros((2,2)) +0j;
	integralLesserKet01 = np.zeros((2,2)) +0j;
	integralLesserKet10 = np.zeros((2,2)) +0j;
	integralLesserKet11 = np.zeros((2,2)) +0j;

	print >> sys.stderr, "Calculating lesser integrals.\n";
	# Conversion for integral over W in the self-consistent equation
	# NB: sum Gamma_alpha = gamma I
	factorW = 1./(2.*np.pi)*gamma +0j ;
	occupancy = lambda epsilon: fd(epsilon-bias/2) + fd(epsilon+bias/2);
	# Actual integration
	for i in range(2):
		#NB: We only need diagonal elements
		# this is where it would cast to real otherwise.
		for j in range(2):
			integralLesserKet00[i][i] += factorW * np.trapz( [occupancy(epsilon)*np.abs(singleParticleGreensFunctionKet00(epsilon).item(i, j))**2 for epsilon in intervalW], intervalW)
			integralLesserKet01[i][i] += factorW * np.trapz( [occupancy(epsilon)*np.abs(singleParticleGreensFunctionKet01(epsilon).item(i, j))**2 for epsilon in intervalW], intervalW)
			integralLesserKet10[i][i] += factorW * np.trapz( [occupancy(epsilon)*np.abs(singleParticleGreensFunctionKet10(epsilon).item(i, j))**2 for epsilon in intervalW], intervalW)
			integralLesserKet11[i][i] += factorW * np.trapz( [occupancy(epsilon)*np.abs(singleParticleGreensFunctionKet11(epsilon).item(i, j))**2 for epsilon in intervalW], intervalW)
		#
	integralLesserKet00 = np.real(integralLesserKet00);
	integralLesserKet01 = np.real(integralLesserKet01);
	integralLesserKet10 = np.real(integralLesserKet10);
	integralLesserKet11 = np.real(integralLesserKet11);

	def printMatrix(M):
		for i in range(2):
			print >> sys.stderr, "\t%.5e\t%.5e" % (M[i][0],M[i][1])
		print >> sys.stderr, "\n";

	printMatrix( integralLesserKet00 );
	printMatrix( integralLesserKet01 );
	printMatrix( integralLesserKet10 );
	printMatrix( integralLesserKet11 );

	#K matrix
	kappaMatrix = np.zeros((2,4)) +0j;
	kappaMatrix[0][1] = 1;
	kappaMatrix[0][3] = 1;
	kappaMatrix[1][2] = 1;
	kappaMatrix[1][3] = 1;

	print >> sys.stderr, "Calculating omega matrix.\n"
	#W matrix
	omegaMatrix = np.zeros((2,4))

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

	initialGuess = initialGuess / sum(initialGuess);

	print >> sys.stderr, "Initial guess: %.5f %.5f %.5f %.5f" % (initialGuess[0],initialGuess[1],initialGuess[2],initialGuess[3]);

	#self-consistent equation f(x) = x
	# error is E(x) = | f(x) - x |^2
	numberError = lambda p: np.sum(np.abs( np.dot( omegaMatrix, p ) - np.dot( kappaMatrix, p ))**2)

	print >> sys.stderr, "Initial Error: %.3f" % numberError(initialGuess);
 
	numericalTolerance = 1e-3;
	numericalMethod = 'SLSQP'; #Sequential Least Squares Programming
	numericalConstraints = [];
	numericalConstraints.append({
	'type': 'eq', # fun needs to equal zero; This normalises the minimised vector
	'fun': lambda p: np.sum(p)-1 
	}); 
	#Vector needs to be positive
	numericalConstraints.append({
		'type': 'ineq', # fun needs to be non negative
		'fun': lambda p: p 
	}); 

	result = minimize( numberError, initialGuess, method=numericalMethod, constraints=numericalConstraints, tol=numericalTolerance);
	selfConsistentProbabilityVector = np.array(result.x);

	separationLength = 0;
	for i in range(4):
		separationLength += (initialGuess[i] - selfConsistentProbabilityVector[i])**2;

	print >> sys.stderr, "Final Error: %.3f" % numberError(selfConsistentProbabilityVector);
	print >> sys.stderr, "Self-consistent result: %.5f %.5f %.5f %.5f" % (selfConsistentProbabilityVector[0],selfConsistentProbabilityVector[1],selfConsistentProbabilityVector[2],selfConsistentProbabilityVector[3]);
	print >> sys.stderr, "Separation length: %.5f\n" % separationLength

	print "%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t" % (betaFraction,beta,selfConsistentProbabilityVector[0],selfConsistentProbabilityVector[1],selfConsistentProbabilityVector[2],selfConsistentProbabilityVector[3], separationLength); 
