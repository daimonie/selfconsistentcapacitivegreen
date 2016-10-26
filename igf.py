import numpy as np
import scipy as sc
import scipy.misc as scmisc
import scipy.special as scspecial
import sys as sys

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
            
            self.sigma_retarded = 1j * (self.gamma_left + self.gamma_right) / 2.0
            self.sigma_advanced = - self.sigma_retarded;
            
            self.dim    = len(self.u)
            self.rho = np.zeros((2**self.dim))
            
            self.beta = param_beta
    
            self.cutoff_chance = 0.0001
            self.external_distribution = False
            self.external_distribution_array = self.distribution()
            self.external_distribution = True
    def singleparticlebackground(self, background):
        """This gives us the single-particle Green's function with some background."""
        
        mu_background = np.diag([self.epsilon[i][i] + np.dot( self.u[i], background) for i in range(0,self.dim)])
        single_retarded = lambda energy: np.linalg.inv( np.eye( self.dim) * energy - mu_background - self.tau - self.sigma_retarded)
        single_advanced = lambda energy: np.linalg.inv( np.eye( self.dim) * energy - mu_background - self.tau - self.sigma_advanced)
        
        return single_retarded, single_advanced
    def generate_superset(self, number):
        """ This function returns a list of the integers that are in the superset of k. The reason this is a seperate function is because I think this algorithm can be a lot smoother/faster."""
        
        superset = []
        for i in range(0, 2**(self.dim)):
            if  (number & i)==number:
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
        
        final_ket = np.array( [(2**i==number&2**i)*1.0 for i in range(0,self.dim)] )
        return final_ket 
    def set_distribution(self, P):
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
        state_i = self.ket( i ) 
        state_j = self.ket( j ) 
                 
        __, ad_gf = self.singleparticlebackground( state_i ) 
        ret_gf, __ = self.singleparticlebackground( state_j ) 
        
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
    def spectral_channel(self, k, epsilon):
        """Returns the spectral function for the many body state k."""
        raise Exception("Needs to be redone")
    def scaler(self):
        scale = 0.0
        chances = self.distribution()
        for k in self.generate_superset(0):
            for l in self.generate_superset(k):
                for ll in self.generate_superset(k):
                    scale += chances[k] * chances[l] * chances[ll]
        return scale
