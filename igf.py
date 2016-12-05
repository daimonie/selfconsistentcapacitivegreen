import numpy as np
import scipy as sc
import scipy.misc as scmisc
import scipy.special as scspecial
import sys as sys
from scipy.optimize import minimize

class igfwl(object): 
    """The igfwl class will compute the transport function using the many-body interacting Green's function in the wide-band limit."""
    def __init__(self, 
        param_epsilon, 
        param_tau,
        param_u, 
        param_gamma_left,
        param_gamma_right,
        param_beta): 
            """The init function takes the single-particle energy matrix (diagonal), the inter-orbital hopping(tunneling), the interaction energy, and the coupling to the leads (Gamma). Finally, it holds the inverse temperature (initial) and the fermi energies of the leads."""
            self.epsilon     = param_epsilon
            self.tau         = param_tau
            self.u           = param_u
            self.gamma_left  = param_gamma_left
            self.gamma_right = param_gamma_right
             
            self.gamma = (np.sum( self.gamma_left) + np.sum(self.gamma_right ))/2.             
            
            self.sigma_retarded = 1j * (self.gamma_left + self.gamma_right) / 2.0
            self.sigma_advanced = - self.sigma_retarded;
            
            self.dim    = len(self.u)
            self.rho = np.zeros((2**self.dim))
            
            self.beta = param_beta
    
            self.cutoff_chance = 0.0001
            self.external_distribution = False
            self.external_distribution_array = self.distribution()
            self.external_distribution = True
            
            self.k_matrix = None
            self.w_matrix = None
    def singleparticlebackground(self, background):
        """This gives us the single-particle Green's function with some background."""
        
        mu_background = np.diag([self.epsilon[i][i] for i in range(0,self.dim)])
         
        ket_background = self.ket( background )
        #Formal, as equation 3.10
        for i in range(mu_background.shape[0]):
            for j in range(mu_background.shape[1]):
                for p in range( self.dim ): 
                    if ket_background[p] == 1.0:
                        mu_background[i,j] += self.u[i, p]
        #Green's function with a single-particle background
        single_retarded = lambda energy: np.linalg.inv( np.eye( self.dim) * energy - mu_background - self.tau - self.sigma_retarded)
        single_advanced = lambda energy: np.linalg.inv( np.eye( self.dim) * energy - mu_background - self.tau - self.sigma_advanced)
        
        return single_retarded, single_advanced
    def generate_superset(self, number):
        """ This function returns a list of the integers that are in the superset of k. The reason this is a seperate function is because I think this algorithm can be a lot smoother/faster."""
        
        superset = []
        for i in range(0, 2**(self.dim)):
            if self.bitwise_and( number, i)==number:
                superset.append(i)
        return superset
    def number(self, ket):
        """Returns the number of the ket"""
        
        final = 0.0
        q = 0
        for i in ket:
            if i != 0:
                final += 2**q
            q += 1 
        return final
    def ket(self, number):
        """Turns an integer number into a ket.""" 
        final_ket = np.array( [(2**i==self.bitwise_and( number, 2**i))*1.0 for i in range(0,self.dim)] ) 
        return final_ket 
    def set_distribution(self, P):
        
        if np.sum(np.abs(P)) < 1.:
            raise TypeError("set_distribution error: P is not normal.")
        self.external_distribution = True
        self.external_distribution_array = P
        
        #print "new distribution set. ", P
    def distribution(self):
        """Sets the boltzmann distribution function and generates the density-matrix."""
        
        #external_distribution serves both the purpose of external setting of distribution and the caching of distribution()
        if self.external_distribution:
            return self.external_distribution_array;
        else:
            energy_vector = []
            superset = self.generate_superset(0) 
            
            for i in superset:
                state           = self.ket(i)
                
                norm_squared    = np.dot(state.T, state)
                
                if norm_squared > 0: #zero is appended at the end
                    energy          = np.dot(state.T, np.dot( self.epsilon, state))
                    interaction     = np.dot(state.T, np.dot( self.u, state))/2.0 #divide by two. Otherwise, <l r| U |l r > =  U_LR + U_RL = 2U
                    #print state, np.dot(self.u, state) 
                    #print interaction
                    energy_vector.append( energy + interaction )
                    
            energy_vector.insert(0, 0.0) 
            probability = np.exp( np.multiply(-self.beta, energy_vector)) 
            probability /= probability.sum() 
            return probability
    
    def transport_channel_ij(self, i, j, chances, epsilon): 
        __, ad_gf = self.singleparticlebackground( i ) 
        ret_gf, __ = self.singleparticlebackground( j ) 
        
        transport_k_ij = np.real([np.trace(np.dot(self.gamma_left, ( np.dot(
            ret_gf(ee),  np.dot(self.gamma_right, ad_gf(ee)))))) for ee in epsilon])
        
        
        #print "%d\t%d\t%.3f\t%.3f\t%2.3f" % (i,j,np.min(transport_k_ij), np.max(transport_k_ij), chances[i]*chances[j])
        transport_k_ij *= chances[i] * chances[j]
        
        return transport_k_ij
    def transport_channel(self, k, epsilon):
        """Returns the transmission function for the many body state k."""
        transport_k = 0 * epsilon
        
        chances = self.distribution() 
        
        for i in self.generate_superset(k): 
            if chances[i] > self.cutoff_chance:
                for j in self.generate_superset(k): 
                    if chances[j] > self.cutoff_chance:  
                        transport_k += np.real( self.transport_channel_ij(i, j, chances, epsilon)) 
        return transport_k * chances[k]
    def full_transmission(self, epsilon):
        transport = 0 * epsilon
        for k in self.generate_superset(0):
            transport += self.transport_channel(k, epsilon)
        scale = self.scaler()
        #print >> sys.stderr, "Scaler found to be %2.3f, scaling T(E)." % scale 
        transport /= scale
        return transport
    def scaler(self):
        scale = 0.0
        chances = self.distribution()
        for k in self.generate_superset(0):
            for l in self.generate_superset(k):
                for ll in self.generate_superset(k):
                    scale += chances[k] * chances[l] * chances[ll]
        return scale
    def bitwise_and(self, a, b):
        return np.bitwise_and( np.uint64(a), np.uint64(b));
    
    def spectral_channel(self, k, epsilon):
        """Returns the spectral function for the many body state k."""
        raise Exception("Needs to be redone") 
    def calculate_number_matrix_k(self):
        """Calculates the K matrix, required for self consistency"""
        number_rows = self.epsilon.shape[0]
        number_cols = self.distribution().shape[0]
        
        self.k_matrix =  np.zeros((number_rows, number_cols))
        #Find the supersets of the ground state (i.e. all many-body eigenvectors)
        #Returns an array of numbers of each state!
        superset = self.generate_superset(0)
        
        #for each number operator
        for i in range(number_rows): 
            #for each number operator
            for j in superset: 
                #remember that the numbered many-body state j is binary encoded!
                #if (i+1) & j == (i+1):
                if self.bitwise_and( i+1, j) == i+1:
                    self.k_matrix[i, j] = 1.0  
        pass
    def calculate_number_matrix_w(self, bias, epsilon_range):
        """Calculates the W matrix, required for self consistency. Warning; assumes left/right coupled lead!!!"""
        number_rows = self.epsilon.shape[0]
        number_cols = self.distribution().shape[0]
        
        self.w_matrix =  np.zeros((number_rows, number_cols))
        superset = self.generate_superset(0)
        
        fd = lambda epsilon: 1.0 / ( 1 + np.exp( np.longfloat(epsilon * self.beta )))
        
        left_green_integral = []
        right_green_integral = []
         
        for i in superset:
            retarded_lambda,  advanced_lambda = self.singleparticlebackground(i) 
            left_w_lambda = []
            right_w_lambda = []
            
            state = self.ket(i)
            for j in range(self.dim):
                if state[j] == 1.0:
                    left_w_lambda.append(lambda epsilon: np.real(fd( epsilon + 0.5*bias ) * retarded_lambda(epsilon).item(j,j)* self.gamma* advanced_lambda(epsilon).item(j,j)))
                    right_w_lambda.append(lambda epsilon: np.real(fd( epsilon - 0.5*bias ) * retarded_lambda(epsilon).item(j,j)* self.gamma* advanced_lambda(epsilon).item(j,j)))
             
            left_green_integral.append(left_w_lambda)
            right_green_integral.append(right_w_lambda)
        #edit this loop so that it creates the integrals required for W
        #should be clear how to do this (for epsi -> correct linspace, etc)
        
        #Upon reconsideration of the derivation leading to 3.13, a factor \frac{\hbar}{\imath} seems to be missing.
        #However, that factor hbar is then absorbed into the fourier -> energy integral substitution
        factor_w = 1./(2.*np.pi) 
        
        for i in superset: 
            left_value = 0.0
            for lambda_fun in left_green_integral[i]:
                left_value += factor_w*np.trapz( [lambda_fun( er ) for er in epsilon_range], epsilon_range)
            right_value = 0.0
            for lambda_fun in right_green_integral[i]:
                left_value += factor_w*np.trapz( [lambda_fun( er ) for er in epsilon_range], epsilon_range)
            #Now, we need to iterate over the filled states in ket(i). Afterwards, we go over the superset again
            #   and add the contribution for each seperate state.
            state = self.ket(i)
            for j in range( self.dim ):
                if state[j] == 1.0:
                    for beta in self.generate_superset(i):
                        betaState = self.ket(beta)
                        if betaState[j] == 1.0: 
                            self.w_matrix[j, beta] += left_value + right_value 
        pass 
    def selfconsistent_distribution(self, tolerance):
        difference = 0.0
        
        initial_guess = self.external_distribution_array
        initial_guess[initial_guess < 1e-9] = 0.0
        
        difference = 1.00
        generation = 0.0
        
        P = initial_guess
        while difference > tolerance:
            number_vector_error = lambda pp: np.sum( np.square( np.dot(self.w_matrix, np.abs(pp))-np.dot(self.k_matrix, np.abs(pp))))
            
            #Find minimum using Nelder-Mead method. Note that both the self-consistency and the Nelder-Mead are
            #   vulnerable for bistability (more than one saddle point)
            res = minimize( number_vector_error, P, method='nelder-mead')
            
            P_previous = P
            
            
            absoluteResultX = np.abs( res.x ) #NB: Minimisation function takes absolute of P
            
            absoluteResultX[ absoluteResultX < self.cutoff_chance] = 0.0      
            P = absoluteResultX / np.sum(absoluteResultX)# Make sure it is normalised
            
            
            
            difference = np.sum( np.square( P-P_previous))
            
            generation += 1
        
        if np.sum(np.square(P)) < 0:
            raise TypeError("P did not converge?")
        P0 = initial_guess
        print "Self-consistency converged after %d generations. \n" % generation
        print "\t Initial probability vector:"
        print "\t %.4e\t%.4e\t%.4e\t%.4e" % (P0[0], P0[1], P0[2], P0[3])
        print "\t Final probability vector:"
        print "\t %.4e\t%.4e\t%.4e\t%.4e" % (P[0], P[1], P[2], P[3]) 
        self.set_distribution( P ) 
        
        return P
        