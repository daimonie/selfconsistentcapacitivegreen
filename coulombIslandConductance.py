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
# Zero-bias level.   
levels = .1*capacitive;
# Integration interval for the self-consistent calculation.
intervalW = np.linspace( -10.0, 10.0, 1e4);
# bias
bias = 0.0
# Temperature (units U); number, min, max
betaNumber = 250-179;
betaMin = 7.151;
betaMax = 10.0;
betaArray = np.linspace(betaMin, betaMax, betaNumber);  

doInv = 1; 
###    
print >> sys.stderr, "Setting system matrices.\n";
#system matrices
hamiltonian = np.zeros((2,2));
hamiltonian[0][0] = -levels;
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



print >> sys.stderr, "Setting Green's function lambda's.\n";
selfEnergy = 0.5j*(gammaLeft + gammaRight);
# single-particle Green's Functions. The numbers denote the state, ket{n_1 n_2}.
# G^{lambda+}
singleParticleGreensFunctionKet00 = lambda epsilon: np.linalg.inv( np.eye( 2 ) * epsilon - hamiltonian + selfEnergy);
singleParticleGreensFunctionKet01 = lambda epsilon: np.linalg.inv( np.linalg.inv(singleParticleGreensFunctionKet00(epsilon)) - interactionKet01);
singleParticleGreensFunctionKet10 = lambda epsilon: np.linalg.inv( np.linalg.inv(singleParticleGreensFunctionKet00(epsilon)) - interactionKet10);
singleParticleGreensFunctionKet11 = lambda epsilon: np.linalg.inv( np.linalg.inv(singleParticleGreensFunctionKet00(epsilon)) - interactionKet11);

parser  = argparse.ArgumentParser(prog="Transport Plot",
  description = "Filename for Transport Plot.")
parser.add_argument(
    '-f',
    '--filename',
    help='File to plot',
    action='store',
    type = str,
    default = ''
)   
args    = parser.parse_args() 
filename = args.filename
file_handler = open( filename, "r" );

data = np.genfromtxt(file_handler, skip_header=0, dtype=None, usecols=range(0,4));

betaFraction = data[:,0];
beta = data[:,1];
n0 = data[:,2];
n1 = data[:,3]; 

epsilon = np.linspace(-4*capacitive, 4*capacitive, 1000);

for i in range(len(n0)):
	print >> sys.stderr, 'Working on beta=%.3f. Progress %d/%d' % (beta[i], i, len(n0));
	m0 = n0[i];
	m1 = n1[i];
	mbGreensFunction = lambda epsilon: (1-m0)*(1-m1) * singleParticleGreensFunctionKet00(epsilon) + m0 * (1-m1) *singleParticleGreensFunctionKet01(epsilon) + m1 * (1-m0) *singleParticleGreensFunctionKet10(epsilon) + m0 * m1 *singleParticleGreensFunctionKet11(epsilon);

	leftMatrix = lambda epsilon: np.dot(gammaLeft, mbGreensFunction(epsilon));
	rightMatrix = lambda epsilon: np.dot(gammaLeft, mbGreensFunction(epsilon).conj());

	T = lambda epsilon: np.dot( leftMatrix(epsilon), rightMatrix(epsilon));


	transport = np.array([np.real(np.trace(T(eps))) for eps in epsilon]);
 
	print '%.9e\t%.9e\t%.9e' % (betaFraction[i], beta[i], np.sum(transport)/(8*capacitive));
#toc
global_time_end = time.time ()
print  >> sys.stderr, "\n Time spent %.6f seconds. \n " % (global_time_end - global_time_start)