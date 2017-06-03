import numpy as np
import matplotlib.pyplot as plt
# Argument parsing
import argparse as argparse    
from experiment import *
# SP.Constants contains physical constants.
from scipy.constants import physical_constants as pc

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

differentialConductance = np.gradient(current, np.mean(np.diff(bias)));

#read experimental data
data_bias, data_current = readExperiment();
physicalCurrentUnit   = 2.0 * pc["elementary charge"][0]**2.0 / pc["Planck constant"][0];

experimentalBias    = data_bias;
experimentalCurrent = data_current / physicalCurrentUnit;

maxExperiment = np.max ( experimentalCurrent);

experimentFactor = np.max(current) / maxExperiment;
experimentalCurrent = experimentalCurrent * experimentFactor;

#
 
#plt.rc('text', usetex=True);
plt.rc('font', family='serif');
plt.subplot(3,1,1);
plt.title('Factor %.3e, beta=%.3e [U]' % (experimentFactor, beta));
plt.plot(bias, current, 'r-', label='current'); 
plt.plot(experimentalBias, experimentalCurrent, 'g-', label='current'); 
plt.ylabel('Current [e/hbar/pi]');
plt.xlim( np.min(bias), np.max(bias));

plt.subplot(3,1,2); 
plt.plot(bias, differentialConductance, 'r-', label='G'); 
plt.ylabel('Differential Conductance [2e^2/h]');
plt.xlim( np.min(bias), np.max(bias));


plt.subplot(3,1,3);
plt.plot(bias, n0, 'g-', label='n_0'); 
plt.plot(bias, n1, 'b-', label='n_1'); 
plt.ylabel('Occupance');
plt.ylim(0, 1);
plt.legend();

plt.xlabel("Bias [eV]");
plt.xlim( np.min(bias), np.max(bias));
plt.show();