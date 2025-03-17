import vedo as vd
vd.settings.default_backend = 'vtk'

import time
from vedo import show
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
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

    def gradient(self, X, x):
        x1, x2 = self.stencil.ExtractVariblesFromVectors(x)
        g1 = x1 - x2
        g2 = x2 - x1
        return np.concatenate((g1, g2))  # connect the gradients

    def hessian(self, X, x):
        #I = np.eye(3)
        I = np.eye(2)
        return np.block([[I, -I], [-I, I]])

class SpringEnergy(ElementEnergy):
    def __init__(self):
        self.stencil = EdgeStencil()
    
    def energy(self, X, x):
        x1, x2 = self.stencil.ExtractVariblesFromVectors(x)
        return 0.5 * np.linalg.norm(x1 - x2)**2
    
    def gradient(self, X, x):
        x1, x2 = self.stencil.ExtractVariblesFromVectors(x)        
        grad_x1 = x1 - x2
        grad_x2 = x2 - x1
        return np.concatenate((grad_x1, grad_x2))  # connect the gradients

    
    def hessian(self, X, x):
        I = np.eye(2)  # create a 2x2 identity matrix
        return np.block([[I, -I], [-I, I]])


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
        #print("In compute_energy")
        energy = 0
        x = x.reshape(-1, 2)  # reshape to a 2D with 2 columns
        for element in self.elements:
            Xi = self.X[list(element), :]
            xi = x[list(element), :]
            energy += self.energy.energy(Xi, xi)
        return energy
    
    def compute_gradient(self, x):
        print("In compute_gradient")
        grad = np.zeros_like(self.V)
        x = x.reshape(-1, 2) # lets reshape x to a 2D array with 2 columns
        for element in self.elements:
            Xi = self.X[list(element), :]
            xi = x[list(element), :]
            gi = self.energy.gradient(Xi, xi)
            for idx, elem in enumerate(element):
                grad[elem] += gi[2 * idx: 2 * (idx + 1)]
        return grad.flatten()
    
    def compute_hessian(self, x):
        print("In compute_hessian")
        I = []
        J = []
        S = []
        x = x.reshape(-1, 2)  
        for element in self.elements:
            Xi = self.X[element, :]
            xi = x[element, :]
            hess = self.energy.hessian(Xi, xi)
            for i in range(6):
                for j in range(6):
                    I.append(element[i % 2] + self.nV * (i // 2))
                    J.append(element[j % 2] + self.nV * (j // 2))
                    S.append(hess[i, j])
        H = coo_matrix((S, (I, J)), shape=(2 * self.nV, 2 * self.nV)).tocsc()
        return H

#%% Optimization
class MeshOptimizer:
    def __init__(self, femMesh):
        self.femMesh = femMesh
        self.SearchDirection = self.GradientDescent
        self.LineSearch = self.BacktrackingLineSearch

    def BacktrackingLineSearch(self, x, d, alpha=1):
        print("In BacktrackingLineSearch")
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
        d = -np.dot(np.linalg.pinv(hess.toarray()), grad)  # we use pseudo-inverse:
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
vertices = np.array([[0, 2], [2, 1], [2, -1], [0, -2], [-2, -1], [-2, 1]])
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

femMesh = FEMMesh(V, F, ZeroLengthSpringEnergy(), EdgeStencil())
optimizer = MeshOptimizer(femMesh)

# lets start with an initial configuration
x = V.copy().flatten()
x_optimized = x.copy()

iterations_num = 15
for i in tqdm(range(iterations_num), desc="Some Direction"):
    plt = vd.Plotter() # Create a plotter object
    x_optimized  = optimizer.step(x_optimized )

    current_mesh = vd.Mesh([x_optimized.reshape(-1, 2), F]).linecolor('black')
    plt.add(current_mesh)
    plt.camera.SetPosition([0, 0, 4])  # Set the camera position
    plt.render()  # Render the plot
    time.sleep(1) # in order to see the result
    plt.close() 

# lets compare to Newton's method gradient descent
optimizer.SearchDirection = optimizer.Newton
x_newton = x.copy()
for i in tqdm(range(iterations_num), desc="Newton's Method"):
    x_newton = optimizer.step(x_newton)
x_gradient = x.copy()
for i in tqdm(range(iterations_num), desc="Gradient Descent"):
    x_gradient = optimizer.step(x_gradient)

# GD plot
gd_mesh = vd.Mesh([x_gradient.reshape(-1, 2), F]).linecolor('blue').legend('Gradient Descent')
title_gd = vd.Text2D("GD", pos="button", c="blue")

# NM plot
newton_mesh = vd.Mesh([x_newton.reshape(-1, 2), F]).linecolor('brown').legend('Newton')
title_newton = vd.Text2D("Newton's", pos="button", c="brown")

#BLS plot
initial_mesh = vd.Mesh([x_optimized.reshape(-1, 2), F]).linecolor('green').legend('Random Direction Solution')
title_initial = vd.Text2D("BLS", pos="button", c="green")

# set plotter
plt = vd.Plotter(shape=(3, 1), size=(1800, 600))  

plt.show(initial_mesh, title_initial, at=0, viewup='2d', resetcam=True)
plt.show(gd_mesh, title_gd, at=1, viewup='2d', resetcam=True)
plt.show(newton_mesh, title_newton, at=2, viewup='2d', resetcam=True)

plt.camera.SetPosition([0, 0, 3])  

# combined plot
plt.show(interactive=True)

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
        #print(f"V[moving_vertex]: {V[moving_vertex]}")
        #print(f"x[moving_vertex]: {x[moving_vertex]}")
        x[moving_vertex, :2] = event.picked3d[:2]  # Update the x variable as well
        moving_vertex = None
        redraw()

def mouse_move(event):
    #print("evt", event)
    #print("evt type", type(event))
    # print("evt.__dict__", evt.__dict__)
    # print("evt.__dict__ type", type(evt.__dict__))
    # print("evt.__dict__['button']", evt.__dict__['button'])
    #print("evt shape", event.shape)
    #print("In OnMouseMove")
    global moving_vertex
    if not event.object:
        return
    if moving_vertex is not None:
        print("moving_vertex is not None")
        V[moving_vertex] = event.picked3d[:2]  # Update the vertex position with new coordinates
    redraw()

def OnLeftButtonRelease(event):
    global moving_vertex
    moving_vertex = None

# Add callbacks
plt.add_callback('LeftButtonPress', OnLeftButtonPress)
plt.add_callback('MouseMove', mouse_move)
plt.add_callback('LeftButtonRelease', OnLeftButtonRelease)
plt.add_callback('RightButtonPress', OnRightButtonPress)
# plt.add_button(on_button_click, pos=(0.7, 0.05), states=["Next Step"], size=20, c="w", bc="green")

vdmesh = vd.Mesh([V, F]).linecolor('black')
plt += vdmesh
plt += vd.Points(V[pinned_vertices, :])
plt += message

plt.user_mode('2d').show()

