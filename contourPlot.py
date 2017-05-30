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

#3d surf plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
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
    default = ''
)   
args    = parser.parse_args() 
filename = args.filename


file_handler = open( filename, "r" );

data = np.genfromtxt(file_handler, dtype=None, usecols=range(0,4)); #excluding the symtype col


betaFraction = data[:,0];

beta = data[:,0];
epsilon = data[:,2];
transport = data[:,3]; 


lin_b = np.linspace(min(beta), max(beta))
lin_e = np.linspace(min(epsilon), max(epsilon))

x, y = np.meshgrid(lin_b, lin_e)
z = griddata(beta, epsilon, transport, lin_b, lin_e, interp='linear')

z = np.log(z/2.)

fig, ax = plt.subplots()


cmap = plt.get_cmap('afmhot') 

levels = MaxNLocator(nbins=100).tick_values(-10, 0) 

cf = ax.contourf(y, np.log(x), z, cmap=cmap, levels=levels)
fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.15)    
 
plt.rc('font', family='serif')

ax.set_facecolor('black'); 

ax.set_ylabel( "log(temperature [U])" ,fontsize=30);
ax.set_xlabel( "epsilon" ,fontsize=30); 
plt.title( "Transport contour versus inverse temperature and chemical potential" ,fontsize=20); 
plt.gca().invert_xaxis();  

plt.show()  