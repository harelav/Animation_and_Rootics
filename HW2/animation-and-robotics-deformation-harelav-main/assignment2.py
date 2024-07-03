#%%
import vedo as vd
vd.settings.default_backend= 'vtk'

from vedo import show
import numpy as np

from abc import ABC, abstractmethod
import numdifftools as nd
from scipy.sparse import coo_matrix
import triangle as tr # pip install triangle  
      

#%% Energy functions
# Abstract element energy class that implements finite differences for the gradient and hessian
# of the energy function. The energy function is defined in the derived classes, which must implement
# the energy method. The gradient and hessian methods should override the finite differences implementation.
# X is the undeformed configuration, x is the deformed configuration in a nx3 matrix, where n is the number
# of vertices in the element.
# the variable ordering for the gradient and hessian methods be in the x1, x2, x3, ..., y1, y2, y3, ... z1, z2, z3 format
class ElementEnergy(ABC):
    # constructor
    def __init__(self,X):
        self.X = X
    
    @abstractmethod
    def energy(x):
        return 0

    # should be overridden by the derived class, otherwise the finite difference implementation will be used
    def gradient(self, x):
        return self.gradient_fd(x)

    def hessian(self, x):
        return self.hessian_fd(x)
    
    # finite difference gradient and hessian
    def gradient_fd(self, x):
        return nd.Gradient(lambda x: self.energy(x))

    def hessian_fd(self, x):
        return nd.Hessian(lambda x: self.energy(x))
    
    # check that the gradient is correct by comparing it to the finite difference gradient
    def check_gradient(self, x):
        grad = self.gradient(x)
        h = 1e-6
        for i in range(3*x.shape[0]):
            x1 = x.copy()
            x2 = x.copy()
            x1[i] += h
            x2[i] -= h
            f1 = self.energy(x1)
            f2 = self.energy(x2)
            fd = (f1 - f2)/(2*h)
            print(grad[i], fd)

# Spring energy function for a zero-length spring, defined as E = 0.5*||x||^2, regardless of the undeformed configuration
class ZeroLengthSpringEnergy(ElementEnergy):
    def energy(x):
        return 0.5*np.linalg.norm(x)**2

    def gradient(x):
        return x.flatten()

    def hessian(x):
        return np.eye(6)
    
# Spring energy function for a spring with a rest length, defined as E = 0.5*(||x|| - l(X))^2, where l is the rest length
class SpringEnergy(ElementEnergy):
    def energy(self, x):
        return 0.5*(np.linalg.norm(x) - np.linalg.norm(self.X))**2
    # def gradient(x):
    #     TODO
    #
    # def hessian(X,x):
    #     TODO


#%% Mesh class
class FEMMesh:
    def __init__(self, mesh, energy, element_type = "edge"):
        self.mesh = mesh
        self.element_type = element_type
        self.energy = energy
        self.elements = np.array(self.mesh.edges if element_type == "edge" else self.mesh.faces)
        self.X = self.mesh.points()
        self.nV = self.X.shape[0]
        self.nE = self.elements.shape[0]
    def compute_energy(self,x):
        energy = 0
        for element in self.elements:
            energy += self.energy.energy(self.X[element], self.x[element])
        return energy
    
    def compute_gradient(self,x):
        grad = np.zeros_like(self.x)
        for element in self.elements:
            g = self.energy.gradient(self.X[element], self.x[element])
            g = g.reshape((-1, 3)) # reshape the gradient to a matrix with columns, one for each axis
            # the gradient is a vector with 3xnV components, where nV is the number of vertices
            
            grad[element] += g[-1, 0]
            grad[element + self.nV] += g[-1, 1]
            grad[element + 2*self.nV] += g[-1, 2]
            
        return grad
    
    def compute_hessian(self,x):
        # create arrays to store the sparse hessian matrix
        I = []
        J = []
        S = []
        for element in self.elements:
            hess = self.energy.hessian(self.X[element], self.x[element])
            for i in range(3):
                for j in range(3):
                    I.append(element[i])
                    J.append(element[j])
                    S.append(hess[i,j])

        H = coo_matrix((S, (I, J)), shape=(3*self.X.shape[0], 3*self.X.shape[0]))

            
#%% Optimization
class MeshOptimizer:
    def __init__(self, femMesh):
        self.femMesh = femMesh
        self.SearchDirection = self.GradientDescent
        self.LineSearch = self.BacktrackingLineSearch

    def BacktrackingLineSearch(self, x, d, alpha=1):
        x0 = x.copy()
        while self.mesh.compute_energy() > self.mesh.compute_energy():
            alpha /= 2
            x = x0 + alpha*d
        return alpha, x

    def GradientDescent(self, x):
        d = self.mesh.compute_gradient(x)
        return -d

    def Newton(self, x):
        grad = self.mesh.compute_gradient(x)
        hess = self.mesh.compute_hessian(x)
        d -= np.linalg.solve(hess, grad)
        return d
    
    def step(self, x):
        d = self.SearchDirection(x)
        new_x, alpha = self.LineSearch(x,d)

    def optimize(self, x, max_iter=100, tol=1e-6):
        for i in range(max_iter):
            x = self.step(x)
            if self.mesh.compute_gradient(x) < tol:
                break
            break
        return x

#%% Main program
vertices = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) # square
tris = tr.triangulate({"vertices":vertices[:,0:2]}, f'qa0.01') # triangulate the square
V = tris['vertices'] # get the vertices of the triangles
F = tris['triangles'] # get the triangles

pinned_vertices = []

def redraw():
    plt.remove("Mesh")
    mesh = vd.Mesh([V,F]).linecolor('black')
    plt.add(mesh)
    plt.remove("Points")
    plt.add(vd.Points(V[pinned_vertices,:],r=10))
    plt.render()

def OnLeftButtonPress(event):
    if event.object is None:          # mouse hits nothing, return.
        print('Mouse hits nothing')
    if isinstance(event.object,vd.mesh.Mesh):          # mouse hits the mesh
        Vi = mesh.closest_point(event.picked3d, return_point_id=True)
        print('Mouse hits the mesh')
        print('Coordinates:', event.picked3d)
        print('Point ID:', Vi)
        if Vi not in pinned_vertices:
            pinned_vertices.append(Vi)
        else:
            pinned_vertices.remove(Vi)
    redraw()

plt = vd.Plotter()

plt.add_callback('LeftButtonPress', OnLeftButtonPress) # add Keyboard callback
mesh = vd.Mesh([V,F]).linecolor('black')
plt += mesh
plt += vd.Points(V[pinned_vertices,:])
plt.user_mode('2d').show().close()

# %%
