#Regular calculation modules.
import numpy as np 
import scipy as sp 
#Allows a debug-output stream.
import sys as sys 
#Physical constants list.
from scipy.constants import *
#Time differences.
import time as time  
#Command line arguments.
import argparse as argparse  

#griddata to format data
from matplotlib.mlab import griddata 

#cycler will cause a colour cycle automatically
from cycler import cycler 
#pyplot is simple plotting
import matplotlib.pyplot as plt 
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
#Commandline arguments instruction.

parser  = argparse.ArgumentParser(prog="Contour Plot",
  description = "Filename for contour plot.")
parser.add_argument(
    '-f',
    '--filename',
    help='File to plot',
    action='store',
    type = str,
    default = 'transport.txt'
)   
args    = parser.parse_args() 
filename = args.filename


file_handler = open( filename, "r" );

data = np.genfromtxt(file_handler, dtype=None, usecols=range(0,4)); #excluding the symtype col


betaFraction = data[:,0];

beta = data[:,0]; #betaFraction, actually. kT/U
epsilon = data[:,2];
transport = data[:,3]; 


lin_b = np.linspace(min(beta), max(beta), 10)
#lin_e = np.linspace(min(epsilon), max(epsilon), 2500)
lin_e = np.linspace(-.15, .5, 2500);

x, y = np.meshgrid(lin_b, lin_e)
z = griddata(beta, epsilon, transport, lin_b, lin_e, interp='linear')

epsilonArray = y[:, 0];

plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'c', 'm','k'])));
fig, ax = plt.subplots();
ax.plot(y, z + x/10.00); 
ax.set_xlabel( "epsilon" ,fontsize=30); 
ax.set_ylabel( "T(e) + beta/10" ,fontsize=30); 
#ax.legend(['kT=%.3f [U]' % b for b in x[0, :]], loc='upper center', ncol=10, bbox_to_anchor=(.5, 1.15));
plt.show (); 