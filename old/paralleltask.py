import numpy as np
import multiprocessing as mp

def taskManagerGroup( arguments ):
    task = arguments[0]
    param_group = arguments[1]
    
    return [task(params) for params in param_group]
class taskManager(object):
    def __init__(self, cores, task):
        """Initiate a parallel task manager with a specified number of cores"""
        self.cores = cores
        self.pool = mp.Pool(processes=cores) #automatically uses all cores 
        self.task = task
        self.params = []
        for i in range(self.cores):
            self.params.append([])
        self.param_count = 0
    def add_params(self, params):
        self.params[ self.param_count%self.cores ].append( params )
        self.param_count += 1
    def print_params(self):
        i = 0
        j = 0
        for group in self.params:
            for params in group:
                print i, j, params, "\n"
                j += 1
            i += 1 
    def execute(self):
        prepared_params = []
        for group in self.params:
            prepared_params.append([ self.task, group]) 
        self.results = self.pool.map(taskManagerGroup, prepared_params)
    def final(self):
        final_results = range( self.param_count )
        
        i = 0
        for group in self.results:
            j = i
            for task in group:
                final_results[j] = task
                j += self.cores
            i += 1
        return final_results
    def get(self, i):
        group_number = i % self.cores
        number = (i - i % self.cores) / self.cores
        
        return self.params[group_number][number]
def test(x): 
    return [x[0], x[0]**2, x[1]]
def parallel_example():
    manager = taskManager( 4, test )
    for i in range(20):
        manager.add_params( [i, "bla"] ) 
    manager.execute()
    results = manager.final()
    print results
    