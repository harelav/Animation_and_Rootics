import vedo as vd
vd.settings.default_backend = 'vtk'

from vedo import show
import numpy as np
from abc import ABC, abstractmethod
import numdifftools as nd
from scipy.sparse import coo_matrix
import triangle as tr  # pip install triangle

#%% Stencil class
class Stencil(ABC):
    @abstractmethod
    def ExtractElementsFromMesh(F):
        edges = {tuple(sorted((F[i, j], F[i, (j + 1) % 3]))) for i in range(F.shape[0]) for j in range(3)}
        return list(edges)

    #@abstractmethod
    @staticmethod
    def ExtractVariblesFromVectors(x):
        return x[:2], x[2:4]

class EdgeStencil(Stencil):
    @staticmethod
    def ExtractElementsFromMesh(F):
        edges = {tuple(sorted((F[i, j], F[i, (j + 1) % 3]))) for i in range(F.shape[0]) for j in range(3)}
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
        # print("grad", grad)
        # print("grad shape", grad.shape)
        grad_fd = self.gradient_fd(X, x)
        return np.linalg.norm(grad - grad_fd)

class ZeroLengthSpringEnergy(ElementEnergy):
    def __init__(self):
        self.stencil = EdgeStencil()
    
    def energy(self, X, x):
        x1, x2 = self.stencil.ExtractVariblesFromVectors(x)
        return 0.5 * np.linalg.norm(x1 - x2) ** 2

# I modified the code here.
    def gradient(self, X, x):
        x1, x2 = self.stencil.ExtractVariblesFromVectors(x)
        g1 = x1 - x2
        g2 = x2 - x1
        return np.array([g1, g2])
        #return np.array([x1 - x2, x2 - x1])

    def hessian(self, X, x):
        #I = np.eye(3)
        I = np.eye(2)
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
        print("In compute_gradient")
        #grad = np.zeros(3 * x.shape[0])
        grad = np.zeros_like(x)
        # print("grad", grad)
        # print("grad shape", grad.shape)
        print("after grad")
        for element in self.elements:
            print("in loop")
            Xi = self.X[element, :]
            xi = x[element, :]
            gi = self.energy.gradient(Xi, xi)
            print("in middle of loop")

            print(f"element: {element}, grad[element[0]] shape: {grad[element[0]].shape}, gi[0] shape: {gi[0].shape}")
            print(f"grad[element[0]]: {grad[element[0]]}, gi[0]: {gi[0]}")
            
            grad[element[0]] += gi[0]
            print("in almost end of loop")
            grad[element[1]] += gi[1]

            # grad[element] += gi[0:1]
            # grad[element + self.nV] += gi[2:3]
            # grad[element + 2 * self.nV] += gi[4:5]
        print("grad computed")    
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
        print("In GradientDescent")
        d = self.femMesh.compute_gradient(x)
        return -d

    def Newton(self, x):
        grad = self.femMesh.compute_gradient(x)
        hess = self.femMesh.compute_hessian(x)
        d = -np.linalg.solve(hess, grad)
        return d
    
    def step(self, x):
        print("In step")
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

mesh = vd.Mesh([V, F]).linecolor('black')

# Initialize the optimizer with the starting positions
#x = np.copy(V)
# Create x with shape (105, 3) by adding a column of zeros
x = np.hstack((V, np.zeros((V.shape[0], 1))))
print("x", x)
print("x shape", x.shape)
iteration = 0

femMesh = FEMMesh(V, F, ZeroLengthSpringEnergy(), EdgeStencil())
optimizer = MeshOptimizer(femMesh)

print("optimizer", optimizer)

def optimization_callback(mesh, iteration, x):
    mesh.points(x)
    message.text(f"Iteration: {iteration}")
    plt.render()

def on_button_click(*args, **kwargs):
    global iteration, x, optimizer, mesh
    x = optimizer.step(x)
    print("x shape", x.shape)
    optimization_callback(mesh, iteration, x)
    iteration += 1

# Setup the plotter and callbacks
plt = vd.Plotter()

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
    global moving_vertex, message
    if event.object is None:  # mouse hits nothing, return.
        message.text("Mouse hits nothing")
    elif isinstance(event.object, vd.mesh.Mesh):  # mouse hits the mesh
        message.text("Mouse hits the mesh")
        Vi = vdmesh.closest_point(event.picked3d, return_point_id=True)
        if Vi not in pinned_vertices:
            pinned_vertices.append(Vi)
        else:
            pinned_vertices.remove(Vi)
        moving_vertex = Vi
        message.text(f'Moving vertex: {Vi}')
    redraw()

def OnRightButtonPress(event):
    global moving_vertex, V, x, message
    if moving_vertex is None:  # No vertex is currently being moved
        message.text("No vertex is being moved")
    else:
        message.text("Mouse hits the mesh")
        print(f"Moving vertex: {moving_vertex}, new coordinates: {event.picked3d[:2]}")  # Debugging line

        V[moving_vertex] = event.picked3d[:2]  # Update the vertex position with new coordinates
        print(f"V[moving_vertex]: {V[moving_vertex]}")
        print(f"x[moving_vertex]: {x[moving_vertex]}")
        x[moving_vertex, :2] = event.picked3d[:2]  # Update the x variable as well
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

# Add callbacks
plt.add_callback('LeftButtonPress', OnLeftButtonPress)
plt.add_callback('MouseMove', OnMouseMove)
plt.add_callback('LeftButtonRelease', OnLeftButtonRelease)
plt.add_callback('RightButtonPress', OnRightButtonPress)
plt.add_button(on_button_click, pos=(0.7, 0.05), states=["Next Step"])

vdmesh = vd.Mesh([V, F]).linecolor('black')
plt += vdmesh
plt += vd.Points(V[pinned_vertices, :])
plt += message

plt.user_mode('2d').show()

