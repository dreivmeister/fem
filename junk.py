
# for j in range(num_quadrilaterals):
#     A[np.ix_(elements4[j], elements4[j])] += stima4(coordinates[elements4[j], :])


# for j in range(num_quadrilaterals):
#     centroid = np.sum(coordinates[elements4[j], :], axis=0)
#     det_val = np.linalg.det(np.hstack((np.ones((3, 1)), coordinates[elements4[j,:3], :])))
#     b[elements4[j]] += det_val * f(np.sum(centroid) / 4) / 4






# general fem problem:
"""
- functions: f (rhs of pde), g (neumann boundary), u_d (dirichlet boundary)
- specifics regarding: node coordinates, at which edges which boundary condition applies
- mesh algorithm generates triangles from node coordinates
"""


"""
coordinates, elements3 = mesh('coordinates.txt', reduce=True, return_coords=True)
num_coordinates = len(coordinates)
num_triangles = elements3.shape[0]

dirichlet = load_data_file('dirichlet.txt', reduce=True, type='int')
num_dirichlet_boundary = len(dirichlet)

neumann = load_data_file('neumann.txt', reduce=True, type='int')
num_neumann_boundary = len(neumann)
"""


# # assemble stiffness over quadrilaterals
# def stima4(vertices):
#     vertices = np.array(vertices)
#     D_Phi = np.array([vertices[1,:]-vertices[0,:],vertices[3,:]-vertices[0,:]]).T
    
#     B = np.linalg.inv(D_Phi.T @ D_Phi)
    
#     C1 = B[0,0]*np.array([[2,-2],[-2,2]]) + B[0,1]*np.array([[3,0],[0,-3]]) + \
#          B[1,1]*np.array([[2,1],[1,2]])
#     C2 = B[0,0]*np.array([[-1,1],[1,-1]]) + B[0,1]*np.array([[-3,0],[0,3]]) + \
#          B[1,1]*np.array([[-1,-2],[-2,-1]])
    
#     M = (np.linalg.det(D_Phi)/6) * np.block([[C1,C2],
#                                              [C2,C1]])
    
#     return M


"""

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

"""