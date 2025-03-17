#%% T3 Q1 Test the optimization pipeline
import vedo as vd
vd.settings.default_backend = 'vtk'

from vedo import show
import numpy as np
from abc import ABC, abstractmethod
import numdifftools as nd
from scipy.sparse import coo_matrix
import triangle as tr  # pip install triangle
import time
from tqdm import tqdm

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


vertices = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) # square
tris = tr.triangulate({"vertices":vertices[:,0:2]}, f'qa0.01') # triangulate the square
V = tris['vertices'] # get the vertices of the triangles
F = tris['triangles'] # get the triangles

# Create a global Plotter object
plt = vd.Plotter()


class EdgeStencil(Stencil):
    @staticmethod
    def ExtractElementsFromMesh(F):
        edges = {tuple(sorted((F[i, j], F[i, (j+1) % 3]))) for i in range(F.shape[0]) for j in range(3)}
        return list(edges)
    
    @staticmethod
    def ExtractVariblesFromVectors(x):
        return x.flat[0:2], x.flat[2:4]  # Adjust to 2D

class ElementEnergy(ABC):    
    @abstractmethod
    def energy(X, x):
        pass

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
        return 0.5 * np.linalg.norm(x1 - x2)**2

    def gradient(self, X, x):
        x1, x2 = self.stencil.ExtractVariblesFromVectors(x)
        grad_x1 = x1 - x2
        grad_x2 = x2 - x1
        return np.concatenate((grad_x1, grad_x2))  # Concatenate the gradients

    def hessian(self, X, x):
        I = np.eye(2)  # Identity matrix of size 2x2
        return np.block([[I, -I], [-I, I]])

class MeshOptimizer:
    def __init__(self, femMesh):
        self.femMesh = femMesh
        self.SearchDirection = self.Newton
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
        d = -np.dot(np.linalg.pinv(hess.toarray()), grad)  # Use pseudo-inverse
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
        x = x.reshape(-1, 2)  # Reshape x to a 2D array with 2 columns
        for element in self.elements:
            Xi = self.X[list(element), :]
            xi = x[list(element), :]
            energy += self.energy.energy(Xi, xi)
        return energy
    
    def compute_gradient(self, x):
        grad = np.zeros_like(self.V)
        x = x.reshape(-1, 2)  # Reshape x to a 2D array with 2 columns
        for element in self.elements:
            Xi = self.X[list(element), :]
            xi = x[list(element), :]
            gi = self.energy.gradient(Xi, xi)
            for idx, elem in enumerate(element):
                grad[elem] += gi[2 * idx: 2 * (idx + 1)]
        return grad.flatten()

    
    def compute_hessian(self, x):
        I = []
        J = []
        S = []
        x = x.reshape(-1, 2)  # Reshape x to a 2D array with 2 columns
        for element in self.elements:
            Xi = self.X[list(element), :]
            xi = x[list(element), :]
            hess = self.energy.hessian(Xi, xi)
            for i in range(4):
                for j in range(4):
                    I.append(element[i % 2] + self.nV * (i // 2))
                    J.append(element[j % 2] + self.nV * (j // 2))
                    S.append(hess[i, j])
        H = coo_matrix((S, (I, J)), shape=(2 * self.nV, 2 * self.nV)).tocsc()
        return H

#%% T3 Q1 Run
# Create the FEMMesh with the ZeroLengthSpringEnergy
femMesh = FEMMesh(V, F, ZeroLengthSpringEnergy(), EdgeStencil())

# Create the optimizer
optimizer = MeshOptimizer(femMesh)
# Initial deformed configuration (same as the original for now)
x = V.copy().flatten()
x_bls = x.copy()

# Run the optimization step by step and show the result after each iteration
n_iterations = 10
for i in tqdm(range(n_iterations), desc="Random Direction"):
    plt = vd.Plotter()
    x_bls = optimizer.step(x_bls)
    
    # Redraw the updated mesh
    updated_mesh = vd.Mesh([x_bls.reshape(-1, 2), F]).linecolor('black')
    plt.add(updated_mesh)

    # Set the camera position to be farther away
    plt.camera.SetPosition([0, 0, 3])  # Adjust the third value to set the distance

    plt.render()  # Render the updated plot without blocking

    # Wait for 1 second
    time.sleep(1)

    # Close the plot
    plt.close()
 

# Compare gradient descent and Newton's method
optimizer.SearchDirection = optimizer.GradientDescent
x_gd = x.copy()
for i in tqdm(range(n_iterations), desc="Gradient Descent"):
    x_gd = optimizer.step(x_gd)

optimizer.SearchDirection = optimizer.Newton
x_newton = x.copy()
for i in tqdm(range(n_iterations), desc="Newton's Method"):
    x_newton = optimizer.step(x_newton)

# Create the Backtracking Line Search solution plot
initial_mesh = vd.Mesh([x_bls.reshape(-1, 2), F]).linecolor('black').legend('Random Direction Solution')
initial_text = vd.Text2D("Random Direction Solution", pos="top-left", c="black")

# Create the Gradient Descent solution plot
gd_mesh = vd.Mesh([x_gd.reshape(-1, 2), F]).linecolor('red').legend('Gradient Descent')
gd_text = vd.Text2D("Gradient Descent Solution", pos="top-left", c="red")

# Create the Newton's Method solution plot
newton_mesh = vd.Mesh([x_newton.reshape(-1, 2), F]).linecolor('blue').legend('Newton')
newton_text = vd.Text2D("Newton's Method Solution", pos="top-left", c="blue")

# Set up the plotter
plt = vd.Plotter(shape=(1, 3), size=(1800, 600))  # 1 row, 3 columns

# Add the plots and titles to the plotter
plt.show(initial_mesh, initial_text, at=0, viewup='2d', resetcam=True)
plt.show(gd_mesh, gd_text, at=1, viewup='2d', resetcam=True)
plt.show(newton_mesh, newton_text, at=2, viewup='2d', resetcam=True)

# Set the camera farther away for all subplots
plt.camera.SetPosition([0, 0, 3])  # Adjust the third value to set the distance

# Show the combined plot
plt.show(interactive=True)
