import numpy as np
from fem_funcs import assemble_stiffness_matrix_A, assemble_right_hand_side_b, enforce_neumann_boundary_cond, enforce_dirichlet_boundary_cond, solve_reduced_system_of_equations, show
from mesh_generation import mesh
from problemfuncs import ProblemFunctions
from dataloader import Dataloader


# problem specific funcs
class Laplace(ProblemFunctions):
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
L = Laplace()


# data loading
D = Dataloader(filename_coordinates='coordinates.txt',filename_dirichlet='dirichlet.txt',filename_neumann='neumann.txt')
coordinates = D.coordinates
coordinates, elements3 = mesh(coordinates, True)
dirichlet = D.dirichlet
neumann = D.neumann

# matrices and vectors
num_coordinates = coordinates.shape[0]
# start assembling the matrices and vectors
A = np.zeros((num_coordinates,num_coordinates))
b = np.zeros((num_coordinates,1))


# assembling
# matrix A
A = assemble_stiffness_matrix_A(A, coordinates=coordinates, triangle_elements=elements3)

# volume forces (vector b)
b = assemble_right_hand_side_b(b, PF=L, coordinates=coordinates, triangle_elements=elements3)

# neumann cond
b = enforce_neumann_boundary_cond(b, PF=L, coordinates=coordinates, neumann_boundary_cond=neumann)

# dirichlet cond
u = enforce_dirichlet_boundary_cond(PF=L, coordinates=coordinates, dirichlet_boundary_cond=dirichlet)
b = b - A @ u

# solving
# calc reduced system solution
u = solve_reduced_system_of_equations(A, b, u, coordinates, dirichlet)
# energy of soultion
# very negative value -> low energy -> small perturbation
# very positive value -> high energy -> larger perturbation
energy = (-u.T @ A @ u)[0][0]
# showing
print(energy)
# show solution
show(elements3, coordinates, u)
