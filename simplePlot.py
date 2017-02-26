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

data = np.genfromtxt(file_handler, skip_header=0, dtype=None, usecols=range(0,7));

betaFraction = data[:,0];
beta = data[:,1];
P00 = data[:,2];
P01 = data[:,3];
P10 = data[:,4];
P11 = data[:,5];
sepLength = data[:,6];

horizontal = beta;
print beta
horizontalLabel = 'beta';

plt.plot(horizontal, P00, 'r-', label='P00');
plt.plot(horizontal, P01, 'g^', label='P01');
plt.plot(horizontal, P10, 'bv', label='P10');
plt.plot(horizontal, P11, 'm-', label='P11'); 
plt.plot(horizontal, sepLength, 'y--', label='sepLength');  
plt.title('Inspecting self-consistent distribution results at capactive=%.3f [eV]' % 0.3);
plt.legend();
#plt.rc('text', usetex=True);
plt.rc('font', family='serif');
plt.xlabel(horizontalLabel);
plt.show();