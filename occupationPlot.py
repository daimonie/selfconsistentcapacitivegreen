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

data = np.genfromtxt(file_handler, skip_header=0, dtype=None, usecols=range(0,4));

print data.shape

betaFraction = data[:,0];
beta = data[:,1];
n0 = data[:,2];
n1 = data[:,3]; 

horizontal = beta;

horizontalLabel = 'beta [eV^{-1}]';

horizontal = betaFraction;
horizontalLabel = 'betaFraction [U]';

plt.semilogx(horizontal, n0, 'r-', label='n0');
plt.semilogx(horizontal, n1, 'g-', label='n1');  
plt.title('Inspecting self-consistent occupation results at capactive=%.3f [eV]' % 0.3);
plt.legend();
#plt.rc('text', usetex=True);
plt.rc('font', family='serif');
plt.xlabel(horizontalLabel);
plt.ylabel('average occupation');
plt.show();