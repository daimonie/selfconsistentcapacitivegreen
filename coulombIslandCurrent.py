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
#some small functions I use, that are in a different repository
from sys import platform
if platform == "linux2":
	sys.path.append('/home/daimonie/ssd/git/PythonUtils') 
else:
	sys.path.append('K:\windows\git\PythonUtils')

import numexpr as ne
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
# Zero-bias level.  
levels = .1*capacitive;
# Integration interval for the self-consistent calculation.
intervalW = np.linspace( -10.0, 10.0, 1e4); 
# Temperature (units U); number, min, max
betaFraction = 0.1;

#Needed for Riemannian sums
differentialEpsilon = 0.01;

# Bias array
biasMinimum = 0;
biasMaximum = 1;
biasNumber = 1000;

biasArray = np.linspace(biasMinimum, biasMaximum, biasNumber);

doInv = 1; 
### loop over each bias voltage
biasIteration = 1;
for bias in biasArray:
	print >> sys.stderr, "Setting system matrices for bias=%.3f, progress %d/%d.\n" % (bias, biasIteration, biasNumber);
	biasIteration += 1;
	#system matrices
	hamiltonian = np.zeros((2,2));
	hamiltonian[0][0] = 0;
	hamiltonian[1][1] = levels;

	#actual interaction term
	andersonInteraction = np.zeros((2,2));
	andersonInteraction[0][1] = capacitive;
	andersonInteraction[1][0] = capacitive;

	#interaction self energy
	interactionKet01 = np.zeros((2,2));
	interactionKet01[1][1] = andersonInteraction[0][1]; 

	interactionKet10 = np.zeros((2,2));
	interactionKet10[0][0] = andersonInteraction[1][0]; 

	interactionKet11 = interactionKet01 + interactionKet10;

	gammaLeft = np.zeros((2,2));
	gammaLeft[0][0] = gamma;
	gammaLeft[1][1] = gamma;

	gammaRight = np.zeros((2,2));
	gammaRight[0][0] = gamma;
	gammaRight[1][1] = gamma; 

	selfEnergy = 0.5j*(gammaLeft + gammaRight);
	# single-particle Green's Functions. The numbers denote the state, ket{n_1 n_2}.
	# G^{lambda+}
	singleParticleGreensFunctionKet00 = lambda epsilon: np.linalg.inv( np.eye( 2 ) * epsilon - hamiltonian + selfEnergy);
	singleParticleGreensFunctionKet01 = lambda epsilon: np.linalg.inv( np.linalg.inv(singleParticleGreensFunctionKet00(epsilon)) - interactionKet01);
	singleParticleGreensFunctionKet10 = lambda epsilon: np.linalg.inv( np.linalg.inv(singleParticleGreensFunctionKet00(epsilon)) - interactionKet10);
	singleParticleGreensFunctionKet11 = lambda epsilon: np.linalg.inv( np.linalg.inv(singleParticleGreensFunctionKet00(epsilon)) - interactionKet11);


	# inverse temperature 
	beta = betaFraction*capacitive;
	 
	if doInv:
		beta = (beta)**(-1.); 
	# Fermi-Dirac distribution

	boltzmann = lambda epsilon: np.exp(-beta*epsilon);
	fd = lambda epsilon: 1.0 / (1.0+np.exp(-beta*epsilon)); 



	# Cached Integrals. Adding 0j makes sure that the datatype is complex. Otherwise, it casts to real later.
	integralLesserKet00 = np.zeros((2,2)) +0j;
	integralLesserKet01 = np.zeros((2,2)) +0j;
	integralLesserKet10 = np.zeros((2,2)) +0j;
	integralLesserKet11 = np.zeros((2,2)) +0j;

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

	densityTransform = Matrix(omegaMatrix - kappaMatrix);
	nullSpace = densityTransform.nullspace();
	nullSpaceList = [];

	i = 0;
	#Remember, the vectors P are chances; the normalisation we seek is sum P = 1
	for nullVector in nullSpace:
		nullSpaceList.append(matrix2numpy(nullVector)[:, 0]);

		nullSpaceList[i] /= np.sum( nullSpaceList[i]); 

		i += 1;

	nullSpaceList = np.array(nullSpaceList);
	nullSpaceShape = nullSpaceList.shape; 

	if nullSpaceShape[0] == 0:
		raise Exception("Abort: The single-particle occupation expectations do not converge.");

	n = np.dot( kappaMatrix, nullSpaceList[0]);

	print >> sys.stderr, "Found n_0=%.9f, n_1=%.9f" % (n[0], n[1]);

	for i in range(2):
		if n[i] >= 0 and n[i] <= 1:
			continue;
		else:
			raise Exception("Occupation number n[%d]=%.3f is unphysical." % (i, n[i]));


	m0 = n[0];
	m1 = n[1];
	mbGreensFunction = lambda epsilon: (1-m0)*(1-m1) * singleParticleGreensFunctionKet00(epsilon) + m0 * (1-m1) *singleParticleGreensFunctionKet01(epsilon) + m1 * (1-m0) *singleParticleGreensFunctionKet10(epsilon) + m0 * m1 *singleParticleGreensFunctionKet11(epsilon);

	leftMatrix = lambda epsilon: np.dot(gammaLeft, mbGreensFunction(epsilon));
	rightMatrix = lambda epsilon: np.dot(gammaRight, mbGreensFunction(epsilon).conj());

	T = lambda epsilon: np.dot( leftMatrix(epsilon), rightMatrix(epsilon));

	epsilonArray = np.arange(-bias/2, bias, differentialEpsilon);

	transport = np.array([np.real(np.trace(T(eps))) for eps in epsilonArray]);
 
	current = np.trapz(transport, epsilonArray);
	current = -np.real(complex(current)); #current is -2e/hbar Int(T(e))
	
	print '%.9e\t%.9e\t%.9e' % (bias, current, betaFraction);

	#print >> sys.stderr, '%.3e \t\t %.3e' % (bias, current);

	#print '%.9e\t%.9e\t%.9e' % (betaFraction[i], beta[i], np.sum(transport)/(8*capacitive));
 	#print "%.9e\t%.9e\t%.9e\t%.9e\t%.9e\t" % (betaFraction, beta, bias, n[0], n[1]);

#toc
global_time_end = time.time ()
print >> sys.stderr, "\n Time spent %.6f seconds. \n " % (global_time_end - global_time_start)