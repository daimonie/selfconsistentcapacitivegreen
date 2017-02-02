Regular calculation modules.
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



parser	= argparse.ArgumentParser(prog="Surface Plot",
  description = "Surface plot of data file")  
parser.add_argument('-z', '--zcolumn', help='z column', action='store', type = int, default = 0)  
parser.add_argument('-g', '--gridsize', help='z column', action='store', type = int, default = 25);

parser.add_argument('-f', '--filename', help='Data file.', action='store', type = str, default = 'distribution_data.txt')    

args	= parser.parse_args() 
zColumn = args.zcolumn;
filename    = args.filename
gridsize    = args.gridsize
 
file_handler = open( filename, "r" ); 
data = np.genfromtxt(file_handler, dtype=None, usecols=range(0,cols)); #excluding the symtype col

arrayBias = data[:, 0]
arrayBeta = data[:, 1]
arrayP0 = data[:, 2]
arrayP1 = data[:, 3]
arrayP2 = data[:, 4]
arrayP3 = data[:, 5]

biasArray = np.linspace( np.min(arrayBias), np.max(arrayBias), gridsize);
arrayBeta = np.linspace( np.min(arrayBeta), np.max(arrayBeta), gridsize);

x = arrayBias;
y = arrayBeta;  

bias, beta = np.meshgrid( biasArray, betaArray);

z = griddata( arrayBias, arrayBeta, arrayP0, biasArray, betaArray, interp='linear');

fig, ax = plt.subplots();

cmap = plt.get_cmap('afmhot');


levels = MaxNLocator(nbins=100).tick_values(z.min(), z.max()) 

cf = ax.contourf(bias, beta, z, cmap=cmap, levels=levels)
fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.15)    

 
ax.set_facecolor('black'); 

ax.set_xlabel('Bias (V)')
ax.set_ylabel('Beta (eV)') 

plt.show() 