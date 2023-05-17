import numpy as np

# load all kinds of data
def load_data_file(filename, reduce=False, type='int'):
    # reduce subtracts 1 from node indices to account for zero-indexing in python
    data = []
    with open(filename) as file:
        if type == 'int':
            # node read (-1)
            for line in file:
                if reduce:
                    data.append([int(i)-1 for i in line.rstrip().split()])
                else:
                    data.append([int(i) for i in line.rstrip().split()])
        else:
            # coordinate read
            for line in file:
                data.append([float(i) for i in line.rstrip().split()])
    return np.array(data)

class Dataloader:
    def __init__(self, filename_coordinates, filename_dirichlet='', filename_neumann='', reduce=True):
        
        self.coordinates = load_data_file(filename_coordinates, type='float')
        self.num_coordinates = self.coordinates.shape[0]
        
        if filename_dirichlet != '':
            self.dirichlet = load_data_file(filename_dirichlet, reduce=reduce, type='int')
            self.num_dirichlet_boundary = self.dirichlet.shape[0]
        if filename_neumann != '':
            self.neumann = load_data_file(filename_neumann, reduce=reduce, type='int')
            self.num_neumann_boundary = self.neumann.shape[0]
    
        
        