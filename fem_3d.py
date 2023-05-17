import numpy as np
from fem_funcs import assemble_stiffness_matrix_A, assemble_right_hand_side_b_3d, enforce_neumann_boundary_cond_3d, enforce_dirichlet_boundary_cond, solve_reduced_system_of_equations
from mesh_generation import mesh
from problemfuncs import ProblemFunctions
from dataloader import Dataloader


# problem specific funcs
class D3(ProblemFunctions):
    # define laplace specific funcs
    # right hand side
    def f(self, u):
        #return np.ones((u.shape[0],1))
        return np.ones((1,1))

    # values at neumann boundary
    def g(self, u):
        #return np.zeros((u.shape[0],1))
        return np.zeros((1,1))

    # values at dirichlet boundary
    def u_d(self, u):
        return np.zeros((u.shape[0],1))
D3 = D3()


#data loading
D = Dataloader(filename_coordinates='coord3d.txt',filename_dirichlet='diri3d.txt',filename_neumann='neu3d.txt')
coordinates = D.coordinates
# elements3 -> tetraeders/triangles
coordinates, elements3 = mesh(coordinates, True)
dirichlet = D.dirichlet
neumann = D.neumann

# matrices and vectors
num_coordinates = coordinates.shape[0]
# start assembling the matrices and vectors
free_nodes = np.setdiff1d(range(1,num_coordinates), np.unique(dirichlet))
A = np.zeros((num_coordinates,num_coordinates))
b = np.zeros((num_coordinates,1))

A = assemble_stiffness_matrix_A(A, coordinates, elements3)



b = assemble_right_hand_side_b_3d(b, D3, coordinates, elements3)


b = enforce_neumann_boundary_cond_3d(b, D3, coordinates, neumann)


u = enforce_dirichlet_boundary_cond(D3, coordinates, dirichlet)

# linear problem
b = b - A @ u

u = solve_reduced_system_of_equations(A, b, u, coordinates, dirichlet)
print(u)


#ADD 3D PLOTTING WITH COLORINDICATION
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def showsurface(surface, coordinates, u):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(coordinates[:,0], coordinates[:,1], coordinates[:,2],
                    triangles=surface, cmap=plt.cm.Spectral, linewidth=0.2, edgecolor='grey', alpha=1, antialiased=True,
                    facecolors=plt.cm.Spectral(u))
    ax.set_axis_off()
    ax.view_init(160, -30)
    plt.show()


showsurface(np.vstack((neumann,dirichlet)), coordinates, u)
# or : showsurface(elements3, coordinates, u)