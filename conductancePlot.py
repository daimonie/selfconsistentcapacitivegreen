import numpy as np
import matplotlib.pyplot as plt
# Argument parsing
import argparse as argparse    

parser  = argparse.ArgumentParser(prog="Simple Plot",
  description = "Filename for simple plot.")
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

data = np.genfromtxt(file_handler, skip_header=0, dtype=None, usecols=range(0,3));
 

betaFraction = data[:,0];
beta = data[:,1];
conductance = data[:,2];
 
horizontal = betaFraction;
horizontalLabel = 'betaFraction [U]';

plt.semilogx(horizontal, conductance, 'r-', label='average conductance'); 
plt.title('Inspecting self-consistent conductance results at capactive=%.3f [eV]' % 0.3);
plt.legend();
#plt.rc('text', usetex=True);
plt.rc('font', family='serif');
plt.xlabel(horizontalLabel);
plt.ylabel('conductance [2e^2/h]');
plt.xlim( 1e-3, 10);
plt.show();