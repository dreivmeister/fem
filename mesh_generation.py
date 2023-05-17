import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial import Delaunay
from fem_funcs import load_data_file

# generating triangle elements using Delaunay triangulation
def mesh(coordinates, return_coords=True):
    simplices = Delaunay(coordinates).simplices
    if return_coords:
        return coordinates, simplices
    else:
        return simplices

def plot_mesh(coordinates, simplices):
    if coordinates.shape[1] == 2:
        plt.triplot(coordinates[:,0],coordinates[:,1],simplices)
        plt.plot(coordinates[:,0],coordinates[:,1], 'o')
        plt.show()
    elif coordinates.shape[1] == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        for tr in simplices:
            pts = coordinates[tr,:]
            ax.plot3D(pts[[0,1],0], pts[[0,1],1], pts[[0,1],2], color='g', lw='0.1')
            ax.plot3D(pts[[0,2],0], pts[[0,2],1], pts[[0,2],2], color='g', lw='0.1')
            ax.plot3D(pts[[0,3],0], pts[[0,3],1], pts[[0,3],2], color='g', lw='0.1')
            ax.plot3D(pts[[1,2],0], pts[[1,2],1], pts[[1,2],2], color='g', lw='0.1')
            ax.plot3D(pts[[1,3],0], pts[[1,3],1], pts[[1,3],2], color='g', lw='0.1')
            ax.plot3D(pts[[2,3],0], pts[[2,3],1], pts[[2,3],2], color='g', lw='0.1')
        ax.scatter(coordinates[:,0], coordinates[:,1], coordinates[:,2], color='b')
        plt.show()


    
if __name__=="__main__":
    fn = 'coord3d.txt'
    coords = load_data_file(fn, type='float')
    coords, simpl = mesh(coords, True)
    
    print(simpl)
    
    plot_mesh(coords, simpl)
