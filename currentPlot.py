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
 

bias = data[:,0];
current = data[:,1];
beta = np.mean(data[:,2]);
 

plt.plot(bias, current, 'r-', label='current'); 
plt.title('Inspecting self-consistent current results at beta=%.3f [U]' % beta);
plt.legend();
#plt.rc('text', usetex=True);
plt.rc('font', family='serif');
plt.xlabel("Bias [eV]");
plt.ylabel('Current [e/hbar/pi]');
plt.xlim( 1e-3, 10);
plt.show();