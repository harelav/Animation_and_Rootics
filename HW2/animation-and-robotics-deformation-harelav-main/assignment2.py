#%%
import vedo as vd
vd.settings.default_backend= 'vtk'

from vedo import show
import numpy as np

from abc import ABC, abstractmethod
import numdifftools as nd
from scipy.sparse import coo_matrix
import triangle as tr # pip install triangle    

#%% Stencil class
# Abstract class for a stencil, which is a local description of the elements in a mesh. The stencil
# is used to extract the elements from the mesh and to extract the different variables from the vectors
# that represent the elements. 
# The ExtractElementsFromMesh method take a Vedo Mesh object as input and returns a list of elements
# The ExtractVariblesFromVectors method takes a vector as input and returns the variables that represent the element
class Stencil(ABC):
    @abstractmethod
    def ExtractElementsFromMesh(F):
        return 0

    @abstractmethod
    def ExtractVariblesFromVectors(x):
        return 0

class EdgeStencil(Stencil):
    # Extract the edges from a mesh
    @staticmethod
    def ExtractElementsFromMesh(F):
        edges = {tuple(sorted((F[i, j], F[i, (j+1) % 3]))) for i in range(F.shape[0]) for j in range(3)}
        return list(edges)
    
    # Extract x1, x2, the two vertices that define the edge, from the vector x, assuming that the variables are stored in the order x1, x2, x3, y1, y2, y3, z1, z2, z3
    # or as a 3x2 matrix, where the columns are the vertices
    @staticmethod
    def ExtractVariblesFromVectors(x):
        return x.flat[0:3], x.flat[3:6]

#%% Energy functions
# Abstract element energy class that implements finite differences for the gradient and hessian
# of the energy function. The energy function is defined in the derived classes, which must implement
# the energy method. The gradient and hessian methods should override the finite differences implementation.
# X is the undeformed configuration and x is the deformed configuration. The order of the variables in X and x
# is implementation dependant, but should be consistent between the energy, gradient and hessian methods.
# The current implementation assumes that the variables are stored in a 1D array, in and x1, x2, x3,..., y1, y2, y3, ..., z1, z2, z3 order.

class ElementEnergy(ABC):    
    @abstractmethod
    def energy(X, x):
        return 0

    # should be overridden by the derived class, otherwise the finite difference implementation will be used
    def gradient(self, X, x):
        return self.gradient_fd(X, x)

    def hessian(self, X, x):
        return self.hessian_fd(X, x)
    
    # finite difference gradient and hessian
    def gradient_fd(self, X, x):
        return nd.Gradient(lambda X, x: self.energy(X, x.flatten()))

    def hessian_fd(self, X, x):
        return nd.Hessian(lambda X, x: self.energy(X, x.flatten()))
    
    # check that the gradient is correct by comparing it to the finite difference gradient
    def check_gradient(self, X, x):
        grad = self.gradient(X, x)
        grad_fd = self.gradient_fd(X, x)
        return np.linalg.norm(grad - grad_fd)


# Spring energy function for a zero-length spring, defined as E = 0.5*||x1-x2||^2, regardless of the undeformed configuration
class ZeroLengthSpringEnergy(ElementEnergy):
    def __init__(self):
        self.stencil = EdgeStencil()
    
    def energy(self, X, x):
        x1,x2 = self.stencil.ExtractVariblesFromVectors(x)
        return 0.5*np.linalg.norm(x1 - x2)**2

    def gradient(self, X, x):
        x1,x2 = self.stencil.ExtractVariblesFromVectors(x)
        return np.array([x1 - x2, x2 - x1])

    def hessian(self, X, x):
        # The hessian is constant and is shapes like [I -I; -I I], where I is the identity matrix
        I = np.eye(3)
        return np.block([[I, -I], [-I, I]])
    
# Spring energy function for a spring with a rest length, defined as E = 0.5*(||x1-x2|| - l)^2, where l is the rest length
class SpringEnergy(ElementEnergy):
    def __init__(self):
        self.stencil = EdgeStencil()
    
    def energy(self, X, x):
        x1,x2 = self.stencil.ExtractVariblesFromVectors(x)
        X1,X2 = self.stencil.ExtractVariblesFromVectors(X)
        return 0.5*(np.linalg.norm(x1-x2) - np.linalg.norm(X1-X2))**2
    # def gradient(self, X, x):
    #     TODO
    #
    # def hessian(self, X, x):
    #     TODO


#%% Mesh class
class FEMMesh:
    def __init__(self, V, F, energy, stencil):
        self.V = V
        self.F = F
        self.energy = energy
        self.stencil = stencil
        self.elements = self.stencil.ExtractElementsFromMesh(V,F)
        self.X = self.V.copy()
        self.nV = self.V.shape[0]

    def compute_energy(self,x):
        energy = 0
        for element in self.elements:
            Xi = self.X[element,:]
            xi = x[element,:]
            energy += self.energy.energy(Xi, xi)
        return energy
    
    def compute_gradient(self,x):
        grad = np.zeros(3*x.shape[0])
        for element in self.elements:
            Xi = self.X[element,:]
            xi = x[element,:]
            gi = self.energy.gradient(Xi, xi)

            grad[element] += gi[0:1]
            grad[element + self.nV] += gi[2:3]
            grad[element + 2*self.nV] += gi[4:5]
            
        return grad
    
    def compute_hessian(self,x):
        # create arrays to store the sparse hessian matrix
        I = []
        J = []
        S = []
        for element in self.elements:
            Xi = self.X[element,:]
            xi = x[element,:]
            hess = self.energy.hessian(Xi,xi) # The hessian is a 6x6 matrix
            for i in range(6):
                for j in range(6):
                    I.append(element[i%2]+self.nV*(i//2))
                    J.append(element[j%2]+self.nV*(j//2))
                    S.append(hess[i,j])
        H = coo_matrix((S, (I, J)), shape=(3*self.nV, 3*self.nV))

            
#%% Optimization
class MeshOptimizer:
    def __init__(self, femMesh):
        self.femMesh = femMesh
        self.SearchDirection = self.GradientDescent
        self.LineSearch = self.BacktrackingLineSearch

    def BacktrackingLineSearch(self, x, d, alpha=1):
        x0 = x.copy()
        f0 = self.femMesh.compute_energy(x0)
        while self.femMesh.compute_energy(x0 + alpha*d) > f0:
            alpha *= 0.5
        return x0 + alpha*d, alpha
    

    def GradientDescent(self, x):
        d = self.femMesh.compute_gradient(x)
        return -d

    def Newton(self, x):
        grad = self.femMesh.compute_gradient(x)
        hess = self.femMesh.compute_hessian(x)
        d -= np.linalg.solve(hess, grad)
        return d
    
    def step(self, x):
        d = self.SearchDirection(x)
        new_x, alpha = self.LineSearch(x,d)
        return new_x

    def optimize(self, x, max_iter=100, tol=1e-6):
        for i in range(max_iter):
            x = self.step(x)
            if self.femMesh.compute_gradient(x) < tol:
                break
        return x

#%% Main program
# Define the boundary vertices for a hexagon
vertices = np.array([
    [0, 2], [2, 1], [2, -1], [0, -2], [-2, -1], [-2, 1]
])

# Create segments to connect boundary vertices
segments = np.array([[i, (i + 1) % len(vertices)] for i in range(len(vertices))])

# Create a boundary description for Triangle
boundary = {
    'vertices': vertices,
    'segments': segments
}

# Triangulate with options to ensure only the provided vertices are used
# 'p' ensures the vertices and segments are used as provided
triangulated = tr.triangulate(boundary, 'p')

# Extract vertices and triangles from the result
V = triangulated['vertices']
F = triangulated['triangles']

print(f"Number of vertices: {len(V)}")
print(f"Number of triangles: {len(F)}")

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
        Vi = vdmesh.closest_point(event.picked3d, return_point_id=True)
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
vdmesh = vd.Mesh([V,F]).linecolor('black')
plt += vdmesh
plt += vd.Points(V[pinned_vertices,:])
plt.user_mode('2d').show().close()

# %%
