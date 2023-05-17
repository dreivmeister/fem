import numpy as np
from fem_funcs import assemble_stiffness_matrix_A, assemble_mass_matrix_B, time_assemble_right_hand_side_b, time_enforce_neumann_boundary_cond, time_enforce_dirichlet_boundary_cond, solve_reduced_system_of_equations, animate
from mesh_generation import mesh
from problemfuncs import ProblemFunctions
from dataloader import Dataloader


# problem specific funcs
class Heat(ProblemFunctions):
    # define laplace specific funcs
    # right hand side
    def f(self, u, t):
        #return np.ones((u.shape[0],1))
        return np.ones((1,1))

    # values at neumann boundary
    def g(self, u, t):
        #return np.zeros((u.shape[0],1))
        return np.zeros((1,1))

    # values at dirichlet boundary
    def u_d(self, u, t):
        return np.zeros((u.shape[0],1))
H = Heat()


# data loading
D = Dataloader(filename_coordinates='coord1.txt',filename_dirichlet='dirichlet.txt',filename_neumann='neumann.txt', reduce=True)
coordinates = D.coordinates
coordinates, elements3 = mesh(coordinates, return_coords=True)
dirichlet = D.dirichlet
neumann = D.neumann

# matrices and vectors
num_coordinates = coordinates.shape[0]


# define vars
A = np.zeros((num_coordinates,num_coordinates))
B = np.zeros((num_coordinates,num_coordinates))

free_nodes = np.setdiff1d(range(1,num_coordinates), np.unique(dirichlet))
T = 1
dt = 0.01
N = int(T/dt)
U = np.zeros((num_coordinates,N+1))


# assembly
# assemble A
A = assemble_stiffness_matrix_A(A, coordinates, elements3)

# assemble B
B = assemble_mass_matrix_B(B, coordinates, elements3)


U[:,0] = np.zeros((num_coordinates,))

# we start at time step 1 and move until N+1 (non inclusive)
for n in range(1, N+1):
    b = np.zeros((num_coordinates,1))
    
    # volume forces
    b = time_assemble_right_hand_side_b(b, H, coordinates, elements3, n*dt, dt)   
        
    # neumann coditions
    b = time_enforce_neumann_boundary_cond(b, H, coordinates, neumann, n*dt, dt)
    
    # previous timestep
    b = b + B.dot(U[:,n-1]).reshape(-1,1)



    # dirichlet
    u = time_enforce_dirichlet_boundary_cond(H, coordinates, dirichlet, n*dt)
    b = b - (dt * A + B).dot(u).reshape(-1,1)
    
    # solve system
    u = solve_reduced_system_of_equations(dt*A+B,b,u,coordinates,dirichlet)
    U[:,n] = u.reshape((-1,))

#print(u)
animate(elements3,coordinates,U)