from fem_funcs import load_data_file
import numpy as np
import matplotlib.pyplot as plt
from mesh_generation import mesh, plot_mesh
from scipy.spatial import Delaunay




coordinates = load_data_file('ada_data/coordinates.txt',type='float')
elements = load_data_file('ada_data/elements.txt',reduce=True)
dirichlet = load_data_file('ada_data/dirichlet.txt',reduce=True)
neumann = load_data_file('ada_data/neumann.txt',reduce=True)


# print(elements)

# plt.triplot(coordinates[:,0],coordinates[:,1],elements)
# plt.plot(coordinates[:,0],coordinates[:,1], 'o')
# plt.show()


# coordinates, simpl = mesh(coordinates, True)
# print(coordinates[simpl[0]])
# plot_mesh(coordinates, simpl)

def num_bound_cond(d,n):
    if type(d) == list and type(n) == list:
        return 0
    if type(d) == list or type(n) == list:
        return 1
    return 2


def provide_geometric_data(elements, dirichlet=[], neumann=[]):
    nE = elements.shape[0]
    nB = num_bound_cond(dirichlet, neumann)
    I = elements.ravel(order='F').reshape(-1,1)
    
    J = elements[:, [1, 2, 0]].reshape(3*nE, 1)
    pointer = np.hstack(([1,3*nE],np.zeros((nB,))))
    bounds = [dirichlet,neumann]
    for j in range(nB):
        boundary = bounds[j]
        if not type(boundary) == list:
            I = np.vstack((I,boundary[:,1].reshape(-1,1)))
            J = np.vstack((J,boundary[:,0].reshape(-1,1)))
        pointer[j+2] = pointer[j+1] + boundary.shape[0]
    

    # Create numbering of edges
    idxIJ = np.where(I < J)[0]
    edgeNumber = np.zeros(len(I))
    edgeNumber[idxIJ] = np.arange(len(idxIJ))
    idxJI = np.where(I > J)[0]
    
    number2edges = np.zeros((np.max(I), np.max(J)))  # create number2edges with max dimensions
    print(number2edges.shape)
    np.add.at(number2edges, (I[idxIJ], J[idxIJ]), np.arange(len(idxIJ)))  # update with indices
    numberingIJ = np.where(number2edges != 0)
    idxJI2IJ = np.where(number2edges[J[idxJI], I[idxJI]] != 0)[0]
    edgeNumber[idxJI[idxJI2IJ]] = numberingIJ[1]




    
            
    
provide_geometric_data(elements, dirichlet, neumann)
    
    






