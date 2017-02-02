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


#griddata to format data
from matplotlib.mlab import griddata 

#3d surf plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

### parameters
import time
global_time_start = time.time()
###
gridsize = 20;

tolerance = 1e-7  

maxBias = 0.25; 

realscale   = pc["elementary charge"][0] / pc["Planck constant"][0] * pc["electron volt"][0]
epsilon_res = 250

epsilon_window = np.linspace(-1.0, 1.0, epsilon_res)

#parameters from fit_0 are default
tau = 0.010
gamma = 0.010
alpha = 0.40
capacitive = 0.300
levels = -capacitive -1e-8 

intervalW = np.linspace( -1.0, 1.0, 1000);

numPoints = 10;

biasArray = np.linspace( -.25, .25, gridsize);
betaArray = np.linspace(0, 1000, gridsize);
  
for bias in biasArray:
    for beta in betaArray:
        spinlessHamiltonian = np.zeros((2,2))
        spinlessHamiltonian[0][0] = levels + 0.5 * alpha * bias
        spinlessHamiltonian[1][1] = levels - 0.5 * alpha * bias

        spinlessTunnellingMatrix = np.zeros((2,2))
        spinlessTunnellingMatrix[0][1] = -tau
        spinlessTunnellingMatrix[1][0] = -tau

        spinlessInteraction = np.zeros((2,2))
        spinlessInteraction[0][1] = capacitive
        spinlessInteraction[1][0] = capacitive


        spinlessGammaLeft = np.zeros((2,2))
        spinlessGammaLeft[0][0] = gamma

        spinlessGammaRIght = np.zeros((2,2))
        spinlessGammaRIght[1][1] = gamma

        spinlessCalculation = igfwl(
            spinlessHamiltonian, 
            spinlessTunnellingMatrix,
            spinlessInteraction, 
            spinlessGammaLeft,
            spinlessGammaRIght, 
            beta
        )  

        spinlessCalculation.label = "P stability at bias %.3f, beta %.3f" % (bias, beta);

        spinlessCalculation.calculate_number_matrix_k()
        spinlessCalculation.calculate_number_matrix_w( -bias, intervalW)

        ### self-consistency loop
        newP = spinlessCalculation.selfconsistent_distribution(tolerance)

        print "%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t" % (bias, beta, newP[0], newP[1], newP[2], newP[3]);

        print >>sys.stderr, "%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t" % (bias, beta, newP[0], newP[1], newP[2], newP[3]);
 
global_time_end = time.time ()
print >> sys.stderr,   "\n Time spent %.6f seconds. \n " % (global_time_end - global_time_start)
