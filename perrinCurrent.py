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

#plots are helpful for debug
import matplotlib.pyplot as plt
#next
import numexpr as ne
from utils import *
# Argument parsing
import argparse as argparse    
#supress warnings
import warnings
# Allows tic/toc on the file execution.
import time
global_time_start = time.time();

parser  = argparse.ArgumentParser(prog="Coulomb Island Current",
  description = "Temperature required.")
parser.add_argument(
    '-b',
    '--beta',
    help='T in units [U]',
    action='store',
    type = float,
    default = 1e-3
)    
args    = parser.parse_args() ;
# Temperature (units U); number, min, max 
betaInverse = args.beta;
#Feedback
print >> sys.stderr, "Distribution for Perrin Molecule.\nSetting parameters, beta = %.3e,\n" % betaInverse;  
# Units of the current.
physicalCurrentUnit   = 2.0 * pc["elementary charge"][0]**2.0 / pc["Planck constant"][0];
# Lead-Molecule coupling (symmetric)
debugFlag = 0;

gamma = 0.005;
# Stark effect strength
alpha = 0.55;
# Interlevel tunnelling
tau = 0.024;
# Interaction strength
capacitive = 0.35;
# Zero-bias level.  
levels = -capacitive;
# Integration interval for the self-consistent calculation.
intervalW = np.linspace( -10.0, 10.0, 1e4); 

# Bias array
biasMinimum = -.250;
biasMaximum = .250;
biasNumber = 200;

biasArray = np.linspace(biasMinimum, biasMaximum, biasNumber);

print >> sys.stderr, "\tlevels = %.3e, tau = %.3e, \n" % (levels, tau);
print >> sys.stderr, "\tgamma = %.3e, capacitive = %.3e, \n" % (gamma, capacitive);
print >> sys.stderr, "Recall unit of current is %.3e.\n" % physicalCurrentUnit;
doInv = 1; 
### loop over each bias voltage
biasIteration = 1;
for bias in biasArray:
	print >> sys.stderr, "Calculating current for bias=%.3f, progress %d/%d.\n" % (bias, biasIteration, biasNumber);
	biasIteration += 1;
	#system matrices
	hamiltonian = np.zeros((2,2));
	hamiltonian[0][0] = levels + 0.5 * alpha * bias;
	hamiltonian[1][1] = levels - 0.5 * alpha * bias;
	hamiltonian[0][1] = -tau;
	hamiltonian[1][0] = -tau;

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

	gammaRight = np.zeros((2,2)); 
	gammaRight[1][1] = gamma; 

	selfEnergy = 0.5j*(gammaLeft + gammaRight);
	# single-particle Green's Functions. The numbers denote the state, ket{n_1 n_2}.
	# G^{lambda+}
	singleParticleGreensFunctionKet00 = lambda epsilon: np.linalg.inv( np.eye( 2 ) * epsilon - hamiltonian + selfEnergy);
	singleParticleGreensFunctionKet01 = lambda epsilon: np.linalg.inv( np.linalg.inv(singleParticleGreensFunctionKet00(epsilon)) - interactionKet01);
	singleParticleGreensFunctionKet10 = lambda epsilon: np.linalg.inv( np.linalg.inv(singleParticleGreensFunctionKet00(epsilon)) - interactionKet10);
	singleParticleGreensFunctionKet11 = lambda epsilon: np.linalg.inv( np.linalg.inv(singleParticleGreensFunctionKet00(epsilon)) - interactionKet11);

	# inverse temperature 
	#beta = betaInverse*capacitive;
	 
	if doInv:
		beta = (betaInverse)**(-1.); 

	# Fermi-Dirac distribution

	boltzmann = lambda epsilon: np.exp(-beta*epsilon);
	fd = lambda epsilon: 1.0 / (1.0+np.exp(-beta*epsilon)); 
	occupancy = lambda epsilon: 0.5 * (fd(epsilon - bias/2.) + fd(epsilon + bias/2.));


	#built in as a control switch
	m0 = 0;
	m1 = 0;



	if 1==1:
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
 			
 			if np.sum( nullSpaceList[i]) > 1.0:
				nullSpaceList[i] /= np.sum( nullSpaceList[i]);  

			i += 1;

		nullSpaceList = np.array(nullSpaceList);
		nullSpaceShape = nullSpaceList.shape; 

		if nullSpaceShape[0] == 0:
			raise Exception("Abort: The single-particle occupation expectations do not converge.");

		n = np.dot( kappaMatrix, nullSpaceList[0]);

		print >> sys.stderr, "Found n_0=%.9f, n_1=%.9f" % (n[0], n[1]);
 

		m0 = n[0];
		m1 = n[1];

	mbGreensFunction = lambda epsilon: (1-m0)*(1-m1) * singleParticleGreensFunctionKet00(epsilon) + m0 * (1-m1) *singleParticleGreensFunctionKet01(epsilon) + m1 * (1-m0) *singleParticleGreensFunctionKet10(epsilon) + m0 * m1 *singleParticleGreensFunctionKet11(epsilon);

	leftMatrix = lambda epsilon: np.dot(gammaLeft, mbGreensFunction(epsilon));
	rightMatrix = lambda epsilon: np.dot(gammaRight, mbGreensFunction(epsilon).conj().transpose());

	T = lambda epsilon: np.dot( leftMatrix(epsilon), rightMatrix(epsilon));

	epsilonArray = np.linspace(-bias/2, bias/2, 1e2);

	transport = np.array([np.real(complex(np.trace(T(eps)))) for eps in epsilonArray]);
 	 
 	if 1==debugFlag:
	 	plt.plot(epsilonArray, transport, 'g-');
	 	plt.plot(epsilonArray, -fd(epsilonArray - bias/2.) + fd(epsilonArray + bias/2.), 'b--')
	 	plt.title('Bias = %.3f' % bias);
	 	plt.xlim(-1.0, 1.0);
		plt.show();

	current = np.trapz(transport, epsilonArray);
	current = np.real(complex(current)); #current is -2e/hbar Int(T(e))


	
	print '%.9e\t%.9e\t%.9e\t%.9e\t%.9e' % (bias, current, betaInverse, m0, m1);
	print >> sys.stderr, '%.9e\t%.9e\t%.9e\t%.9e\t%.9e' % (bias, current, betaInverse, m0, m1);

	#print >> sys.stderr, '%.3e \t\t %.3e' % (bias, current);

	#print '%.9e\t%.9e\t%.9e' % (betaInverse[i], beta[i], np.sum(transport)/(8*capacitive));
 	#print "%.9e\t%.9e\t%.9e\t%.9e\t%.9e\t" % (betaInverse, beta, bias, n[0], n[1]);

#toc
global_time_end = time.time ()
print >> sys.stderr, "\n Time spent %.6f seconds. \n " % (global_time_end - global_time_start)