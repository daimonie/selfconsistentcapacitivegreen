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

data = np.genfromtxt(file_handler, skip_header=0, dtype=None, usecols=range(0,5));
 

bias = data[:,0];
current = data[:,1];
beta = np.mean(data[:,2]);
n0 = data[:,3];
n1 = data[:,4];
 
#plt.rc('text', usetex=True);
plt.rc('font', family='serif');
plt.title('Inspecting current results at beta=%.3f [U]' % beta);
plt.subplot(2,1,1);
plt.plot(bias, current, 'r-', label='current'); 
plt.ylabel('Current [e/hbar/pi]');
plt.xlim( -1, 1);


plt.subplot(2,1,2);
plt.plot(bias, n0, 'g-', label='n_0'); 
plt.plot(bias, n1, 'b-', label='n_1'); 
plt.ylabel('Occupance');
plt.legend();

plt.xlabel("Bias [eV]");
plt.xlim( -1, 1);
plt.show();