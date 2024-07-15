import vedo as vd
vd.settings.default_backend= 'vtk'

from vedo import show
import numpy as np
from abc import ABC, abstractmethod
import numdifftools as nd
from scipy.sparse import coo_matrix
import triangle as tr # pip install triangle    

#%% Stencil class
class Stencil(ABC):
    @abstractmethod
    def ExtractElementsFromMesh(F):
        return 0

    @abstractmethod
    def ExtractVariblesFromVectors(x):
        return 0

class EdgeStencil(Stencil):
    @staticmethod
    def ExtractElementsFromMesh(F):
        edges = {tuple(sorted((F[i, j], F[i, (j+1) % 3]))) for i in range(F.shape[0]) for j in range(3)}
        return list(edges)
    
    @staticmethod
    def ExtractVariblesFromVectors(x):
        return x.flat[0:3], x.flat[3:6]

#%% Energy functions
class ElementEnergy(ABC):    
    @abstractmethod
    def energy(X, x):
        return 0

    def gradient(self, X, x):
        return self.gradient_fd(X, x)

    def hessian(self, X, x):
        return self.hessian_fd(X, x)
    
    def gradient_fd(self, X, x):
        return nd.Gradient(lambda X, x: self.energy(X, x.flatten()))

    def hessian_fd(self, X, x):
        return nd.Hessian(lambda X, x: self.energy(X, x.flatten()))
    
    def check_gradient(self, X, x):
        grad = self.gradient(X, x)
        grad_fd = self.gradient_fd(X, x)
        return np.linalg.norm(grad - grad_fd)

class ZeroLengthSpringEnergy(ElementEnergy):
    def __init__(self):
        self.stencil = EdgeStencil()
    
    def energy(self, X, x):
        x1, x2 = self.stencil.ExtractVariblesFromVectors(x)
        return 0.5 * np.linalg.norm(x1 - x2) ** 2

    def gradient(self, X, x):
        x1, x2 = self.stencil.ExtractVariblesFromVectors(x)
        return np.array([x1 - x2, x2 - x1])

    def hessian(self, X, x):
        I = np.eye(3)
        return np.block([[I, -I], [-I, I]])

class SpringEnergy(ElementEnergy):
    def __init__(self):
        self.stencil = EdgeStencil()
    
    def energy(self, X, x):
        x1, x2 = self.stencil.ExtractVariblesFromVectors(x)
        X1, X2 = self.stencil.ExtractVariblesFromVectors(X)
        return 0.5 * (np.linalg.norm(x1 - x2) - np.linalg.norm(X1 - X2)) ** 2
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
        self.elements = self.stencil.ExtractElementsFromMesh(F)
        self.X = self.V.copy()
        self.nV = self.V.shape[0]

    def compute_energy(self, x):
        energy = 0
        for element in self.elements:
            Xi = self.X[element, :]
            xi = x[element, :]
            energy += self.energy.energy(Xi, xi)
        return energy
    
    def compute_gradient(self, x):
        grad = np.zeros(3 * x.shape[0])
        for element in self.elements:
            Xi = self.X[element, :]
            xi = x[element, :]
            gi = self.energy.gradient(Xi, xi)

            grad[element] += gi[0:1]
            grad[element + self.nV] += gi[2:3]
            grad[element + 2 * self.nV] += gi[4:5]
            
        return grad
    
    def compute_hessian(self, x):
        I = []
        J = []
        S = []
        for element in self.elements:
            Xi = self.X[element, :]
            xi = x[element, :]
            hess = self.energy.hessian(Xi, xi)
            for i in range(6):
                for j in range(6):
                    I.append(element[i % 2] + self.nV * (i // 2))
                    J.append(element[j % 2] + self.nV * (j // 2))
                    S.append(hess[i, j])
        H = coo_matrix((S, (I, J)), shape=(3 * self.nV, 3 * self.nV))
        return H

#%% Optimization
class MeshOptimizer:
    def __init__(self, femMesh):
        self.femMesh = femMesh
        self.SearchDirection = self.GradientDescent
        self.LineSearch = self.BacktrackingLineSearch

    def BacktrackingLineSearch(self, x, d, alpha=1):
        x0 = x.copy()
        f0 = self.femMesh.compute_energy(x0)
        while self.femMesh.compute_energy(x0 + alpha * d) > f0:
            alpha *= 0.5
        return x0 + alpha * d, alpha

    def GradientDescent(self, x):
        d = self.femMesh.compute_gradient(x)
        return -d

    def Newton(self, x):
        grad = self.femMesh.compute_gradient(x)
        hess = self.femMesh.compute_hessian(x)
        d = -np.linalg.solve(hess, grad)
        return d
    
    def step(self, x):
        d = self.SearchDirection(x)
        new_x, alpha = self.LineSearch(x, d)
        return new_x

    def optimize(self, x, max_iter=100, tol=1e-6):
        for i in range(max_iter):
            x = self.step(x)
            if np.linalg.norm(self.femMesh.compute_gradient(x)) < tol:
                break
        return x

#%% Main program
vertices = np.array([
    [0, 2], [2, 1], [2, -1], [0, -2], [-2, -1], [-2, 1]
])

segments = np.array([[i, (i + 1) % len(vertices)] for i in range(len(vertices))])

boundary = {
    'vertices': vertices,
    'segments': segments
}

hex_area = 1.5 * np.sqrt(3) * (2 ** 2)
desired_triangles = 100
max_triangle_area = hex_area / desired_triangles

triangulated = tr.triangulate(boundary, f'pa{max_triangle_area:.8f}')

V = triangulated['vertices']
F = triangulated['triangles']

pinned_vertices = []
moving_vertex = None
message = vd.Text2D("", pos='bottom-left', c='black', font='Courier', s=1)

def redraw():
    plt.remove("Mesh")
    mesh = vd.Mesh([V, F]).linecolor('black')
    plt.add(mesh)
    plt.remove("Points")
    plt.add(vd.Points(V[pinned_vertices, :], r=10))
    
    plt.remove(message)
    plt.add(message)
    plt.render()

def OnLeftButtonPress(event):
    print("In OnLeftButtonPress")
    global moving_vertex, message
    print(f"isinstance(event.object, vd.mesh.Mesh)", isinstance(event.object, vd.mesh.Mesh))
    if event.object is None:  # mouse hits nothing, return.
        message.text("Mouse hits nothing")
    elif isinstance(event.object, vd.mesh.Mesh):  # mouse hits the mesh
        message.text("Mouse hits the mesh")
        Vi = vdmesh.closest_point(event.picked3d, return_point_id=True)
        if Vi not in pinned_vertices:
            pinned_vertices.append(Vi)
        else:
            pinned_vertices.remove(Vi)
        print(f"Vi: {Vi}")
        moving_vertex = Vi
        message.text(f'Moving vertex: {Vi}')
    redraw()

def OnRightButtonPress(event):
    print("In OnRightButtonPress")
    global moving_vertex
    if moving_vertex is None:  # No vertex is currently being moved
        message.text("No vertex is being moved")
    else:
        message.text("Mouse hits the mesh")
        print(f"Moving vertex: {moving_vertex}, new coordinates: {event.picked3d[:2]}")  # Debugging line
        V[moving_vertex] = event.picked3d[:2]  # Update the vertex position with new coordinates
        moving_vertex = None
        redraw()


def OnMouseMove(event):
    global moving_vertex
    if moving_vertex is not None:
        V[moving_vertex] = event.picked3d[:2]  # Update the vertex position with new coordinates
        
        redraw()

def OnLeftButtonRelease(event):
    global moving_vertex
    moving_vertex = None

plt = vd.Plotter()

plt.add_callback('LeftButtonPress', OnLeftButtonPress)  # add Left Button Press callback
plt.add_callback('MouseMove', OnMouseMove)  # add Mouse Move callback
plt.add_callback('LeftButtonRelease', OnLeftButtonRelease)  # add Left Button Release callback
plt.add_callback('RightButtonPress', OnRightButtonPress)  # add Right Button Press callback

vdmesh = vd.Mesh([V, F]).linecolor('black')
plt += vdmesh
plt += vd.Points(V[pinned_vertices, :])
plt += message
plt.user_mode('2d').show().close()

# Capture a screenshot for the report
plt.screenshot("vedo_debug_view.png")
   