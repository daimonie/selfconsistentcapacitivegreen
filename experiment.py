import numpy as np
import scipy.interpolate as si
def read_experiment_data( filename ):
    
    file_handler = open( filename, "r" );
    
    data = np.genfromtxt(file_handler, skip_header=9, dtype=None, usecols=range(0,3)); #excluding the symtype col
    exp_bias = data[:,0]
    exp_current = data[:,1] 
    
    truebias = np.linspace(-0.2, 0.2, int(exp_bias.shape[0]/2))
    
    truecurrent = si.griddata(exp_bias, exp_current, truebias, method='nearest')
    file_handler.close()
    
    return truebias, truecurrent
def read_experiment( filename ):
    exp_bias, exp_current = read_experiment_data(filename)
    
    dI = np.max(exp_current) - np.min(exp_current)
    dV = np.max(exp_bias) - np.min(exp_bias)
    
    exp_background = lambda V: dI/dV * V
    
    exp_current -= exp_background(exp_bias)
    
    return exp_bias, exp_current 
def calculate_error( param_bias, param_current, param_exp ):
    error_func = 0.0
    param_bias = np.array( param_bias )
    param_current = np.array( param_current )
    param_exp = np.array( param_exp )
    
    if param_bias.shape[0] == param_current.shape[0] and param_current.shape[0] == param_exp.shape[0]:
        peak_current = param_current.max()
        peak_exp = param_exp.max()
         
        param_current /= peak_current
        param_exp /= peak_exp 
        
        squares = np.square( param_exp - param_current)
        sum_least_squares = squares.sum()
        
        scaler = 1.0
        if peak_current > peak_exp:
            scaler = peak_current / peak_exp
            #print peak_current, peak_exp, scaler
        else:
            scaler = peak_exp / peak_current
            #print peak_current, peak_exp, scaler
            
        error_func = sum_least_squares
        #print scaler, error_func, sum_least_squares
        return scaler, error_func, error_func * scaler
    else:
        raise Exception("Calculate Error: Arguments should have the same shape.")
     