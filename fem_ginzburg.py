import numpy as np
from fem_funcs import show, load_data_file
from mesh_generation import mesh, plot_mesh


# define problem specific funcs
def f(u):
    #return np.ones((u.shape[0],1))
    return np.ones((1,1))


def g(u):
    #return np.zeros((u.shape[0],1))
    return np.zeros((1,1))


def u_d(u):
    return np.zeros((u.shape[0],1))



# local integrals
def localj(vertices, U):
    eps = 1/100
    G = np.linalg.solve(np.vstack([np.ones(3), vertices.T]), np.vstack([np.zeros(2), np.eye(2)]))
    area = np.linalg.det(np.vstack([np.ones(3), vertices.T])) / 2
    b = area * ((eps * np.dot(G, G.T) - np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12) @ U + 
                np.array([4*U[0]**3 + U[1]**3 + U[2]**3 + 3*U[0]**2*(U[1]+U[2]) + 2*U[0]*(U[1]**2+U[2]**2)+U[1]*U[2]*(U[1]+U[2]) + 2*U[0]*U[1]*U[2],
                          4*U[1]**3 + U[0]**3 + U[2]**3 + 3*U[1]**2*(U[0]+U[2]) + 2*U[1]*(U[0]**2+U[2]**2)+U[0]*U[2]*(U[0]+U[2]) + 2*U[0]*U[1]*U[2],
                          4*U[2]**3 + U[1]**3 + U[0]**3 + 3*U[2]**2*(U[1]+U[0]) + 2*U[2]*(U[1]**2+U[0]**2)+U[1]*U[0]*(U[1]+U[0]) + 2*U[0]*U[1]*U[2]]) / 60)
    return b



def localdj(vertices, U):
    eps = 1/100
    G = np.linalg.solve(np.vstack([np.ones(3), vertices.T]), np.vstack([np.zeros(2), np.eye(2)]))
    area = np.linalg.det(np.vstack([np.ones(3), vertices.T])) / 2
    a = np.array(       [[12*U[0]**2+2*(U[1]**2+U[2]**2+U[1]*U[2])+6*U[0]*(U[1]+U[2]), 
                          3*(U[0]**2+U[1]**2)+U[2]**2+4*U[0]*U[1]+2*U[2]*(U[0]+U[1]), 
                         3*(U[0]**2+U[2]**2)+U[1]**2+4*U[0]*U[2]+2*U[1]*(U[0]+U[2])], 
                         [3*(U[0]**2+U[1]**2)+U[2]**2+4*U[0]*U[1]+2*U[2]*(U[0]+U[1]), 
                          12*U[1]**2+2*(U[0]**2+U[2]**2+U[0]*U[2])+6*U[1]*(U[0]+U[2]), 
                          3*(U[1]**2+U[2]**2)+U[0]**2+4*U[1]*U[2]+2*U[0]*(U[1]+U[2])], 
                         [3*(U[0]**2+U[2]**2)+U[1]**2+4*U[0]*U[2]+2*U[1]*(U[0]+U[2]), 
                          3*(U[1]**2+U[2]**2)+U[0]**2+4*U[1]*U[2]+2*U[0]*(U[1]+U[2]), 
                          12*U[2]**2+2*(U[0]**2+U[1]**2+U[0]*U[1])+6*U[2]*(U[0]+U[1])]]
                         ).reshape((3,3))
    M = area * (eps * np.dot(G,G.T) - np.array([[2,1,1], [1,2,1], [1,1,2]]) / 12 + a / 60)
    return M




# set reduce=True if you read from one-indexing sources
# else reduce=False

coordinates, elements3 = mesh('coordinates.txt', reduce=True, return_coords=True)
#plot_mesh(coordinates,elements3)
# coordinates = load_data_file('coordinates.txt', type='float')
# elements3 = load_data_file('elements3.txt', reduce=True)
num_coordinates = len(coordinates)
num_triangles = elements3.shape[0]

dirichlet = load_data_file('dirichlet.txt', reduce=True)
num_dirichlet_boundary = len(dirichlet)

neumann = load_data_file('neumann.txt', reduce=True)
num_neumann_boundary = len(neumann)


bound_nodes = np.unique(dirichlet)
free_nodes =  np.setdiff1d(range(1,num_coordinates), bound_nodes)
U = -np.ones((num_coordinates,1))
U[bound_nodes] = u_d(coordinates[bound_nodes])


n = 50
for i in range(n):
    # assembly of DJ(U)
    A = np.zeros((num_coordinates,num_coordinates))
    for j in range(num_triangles):    
        A[np.ix_(elements3[j], elements3[j])] += localdj(coordinates[elements3[j],:],U[elements3[j],:])
        
    
    b = np.zeros((num_coordinates,1))
    for j in range(num_triangles):
        b[elements3[j]] += localj(coordinates[elements3[j],:],U[elements3[j],:])

    
    for j in range(num_triangles):
        b[elements3[j]] += np.linalg.det(np.vstack([np.ones(3), coordinates[elements3[j],:].T])) * f(np.sum(coordinates[elements3[j],:])/3)/6

    
    for j in range(num_neumann_boundary):
        b[neumann[j]] -= np.linalg.norm(coordinates[neumann[j,0],:]-coordinates[neumann[j,1],:]) * g(np.sum(coordinates[neumann[j],:])/2)/2
    
    W = np.zeros((num_coordinates,1))
    W[bound_nodes] = 0
    
    
    W[free_nodes] = np.linalg.solve(A[free_nodes, :][:, free_nodes], b[free_nodes])
    U = U - W
    eps = 10**(-5)
    if np.linalg.norm(W) < eps:
        break

show(elements3,coordinates,U)
    
    
    
    
    
    
    
    
        
    







