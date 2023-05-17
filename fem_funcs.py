import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def show(elements, coordinates, u):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Display the information associated with triangular elements.
    if len(u.shape) == 2:
        ax.plot_trisurf(coordinates[:,0], coordinates[:,1], u[:,0], triangles=elements, cmap='viridis')
    elif len(u.shape) == 1:
        ax.plot_trisurf(coordinates[:,0], coordinates[:,1], u[:], triangles=elements, cmap='viridis')

    # Define the initial viewing angle.
    ax.view_init(elev=30, azim=-67.5)

    ax.set_title('Solution')

    plt.show()
    
def animate(elements, coordinates, U):
    fig = plt.figure()
    num_frames = U.shape[1]-1
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Solution')
    #ax.view_init(elev=30, azim=-67.5)
    
    def show_frame(frame, coordinates, elements):
        ax.clear()
        ax.plot_trisurf(coordinates[:,0], coordinates[:,1], U[:,frame], triangles=elements, cmap='viridis')
        
    ani = FuncAnimation(fig, show_frame, frames=num_frames, fargs=(coordinates, elements))
    
    plt.show()


# assemble stiffness over triangles
def stima3(vertices):
    d = len(vertices[0])
    if d == 2:
        pr = 2
    elif d == 3:
        pr = 6
    else:
        print(f'dim has to be 2 or 3 but is: {d}')
        return
        
    T = np.ones((d+1,d+1))
    for j in range(d):
        c = np.array([v[j] for v in vertices])
        T[j+1,:] = c[:]

    E = np.concatenate((np.expand_dims(np.zeros(d), axis=0), np.eye(d)), axis=0)
    
    G = np.linalg.solve(T, E)
    
    M = (np.linalg.det(T)/pr) * G @ G.T
    
    return M


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



# assemble and solve funcs (time invariant)
def assemble_stiffness_matrix_A(A, coordinates, triangle_elements):
    num_triangles = triangle_elements.shape[0]
    for j in range(num_triangles):
        A[np.ix_(triangle_elements[j], triangle_elements[j])] += stima3(coordinates[triangle_elements[j], :])
    return A

def assemble_right_hand_side_b(b, PF, coordinates, triangle_elements):
    num_triangles = triangle_elements.shape[0]
    for j in range(num_triangles):
        centroid = np.sum(coordinates[triangle_elements[j], :], axis=0) / 3
        det_val = np.linalg.det(np.hstack((np.ones((3, 1)), coordinates[triangle_elements[j], :])))
        b[triangle_elements[j]] += det_val * PF.f(np.sum(centroid) / 3) / 6
    return b


def assemble_right_hand_side_b_3d(b, PF, coordinates, triangle_elements):
    num_triangles = triangle_elements.shape[0]
    for j in range(num_triangles):
        centroid = np.sum(coordinates[triangle_elements[j], :], axis=0) / 4
        det_val = np.linalg.det(np.hstack((np.ones((4, 1)), coordinates[triangle_elements[j], :])))
        b[triangle_elements[j]] += det_val * PF.f(np.sum(centroid) / 4) / 24
    return b

def time_assemble_right_hand_side_b(b, PF, coordinates, triangle_elements, t, dt):
    # t is n*dt (i*dt) current time
    num_triangles = triangle_elements.shape[0]
    for j in range(num_triangles):
        centroid = np.sum(coordinates[triangle_elements[j],:], axis=0)
        det_val = np.linalg.det(np.hstack((np.ones((3, 1)), coordinates[triangle_elements[j], :])))
        b[triangle_elements[j]] += det_val * dt * PF.f(centroid / 3, t) / 6    
    return b


def time_enforce_neumann_boundary_cond(b, PF, coordinates, neumann_boundary_cond, t, dt):
        num_neumann_boundary = neumann_boundary_cond.shape[0]
        for j in range(num_neumann_boundary):
            b[neumann_boundary_cond[j]] += np.linalg.norm(coordinates[neumann_boundary_cond[j,0],:] - coordinates[neumann_boundary_cond[j,1],:]) * \
                dt*PF.g(np.sum(coordinates[neumann_boundary_cond[j],:], axis=0)/2, t)/2
        return b



def enforce_neumann_boundary_cond(b, PF, coordinates, neumann_boundary_cond):
    if neumann_boundary_cond.size == 0:
        return b
    
    num_neumann_boundary = neumann_boundary_cond.shape[0]
    for j in range(num_neumann_boundary):
        b[neumann_boundary_cond[j]] += np.linalg.norm(coordinates[neumann_boundary_cond[j,0],:] - coordinates[neumann_boundary_cond[j,1],:]) * \
            PF.g(np.sum(coordinates[neumann_boundary_cond[j],:], axis=0)/2)/2
    return b


def enforce_neumann_boundary_cond_3d(b, PF, coordinates, neumann_boundary_cond):
    if neumann_boundary_cond.size == 0:
        return b
    
    for j in range(neumann_boundary_cond.shape[0]):
        b[neumann_boundary_cond[j]] += np.linalg.norm(np.cross(coordinates[neumann_boundary_cond[j,2],:]-coordinates[neumann_boundary_cond[j,0],:], 
                                                               coordinates[neumann_boundary_cond[j,1],:]-coordinates[neumann_boundary_cond[j,0],:])) \
                                                               * PF.g(np.sum(coordinates[neumann_boundary_cond[j],:], axis=0)/3) / 6
    return b


def enforce_dirichlet_boundary_cond(PF, coordinates, dirichlet_boundary_cond):
    num_coordinates = coordinates.shape[0]
    
    u = np.zeros((num_coordinates, 1))
    bound_nodes = np.unique(dirichlet_boundary_cond)
    u[bound_nodes] = PF.u_d(coordinates[bound_nodes])
    
    return u


def time_enforce_dirichlet_boundary_cond(PF, coordinates, dirichlet_boundary_cond, t):
        num_coordinates = coordinates.shape[0]
        
        u = np.zeros((num_coordinates, 1))
        bound_nodes = np.unique(dirichlet_boundary_cond)
        u[bound_nodes] = PF.u_d(coordinates[bound_nodes],t)
        
        return u


def solve_reduced_system_of_equations(A, b, u, coordinates, dirichlet_boundary_cond):
    num_coordinates = coordinates.shape[0]
    bound_nodes = np.unique(dirichlet_boundary_cond)
    
    free_nodes = np.setdiff1d(range(1,num_coordinates), bound_nodes)
    u[free_nodes] = np.linalg.solve(A[free_nodes,:][:,free_nodes], b[free_nodes])
    
    return u


def assemble_mass_matrix_B(B, coordinates, triangle_elements):
    m = np.array([[2,1,1],
                  [1,2,1],
                  [1,1,2]])
    num_triangles = triangle_elements.shape[0]
    for j in range(num_triangles):
        det_val = np.linalg.det(np.hstack((np.ones((3, 1)), coordinates[triangle_elements[j], :])))
        B[np.ix_(triangle_elements[j], triangle_elements[j])] += (det_val/24) * m    
    return B
