# Animation and Robotics - Assignment 1: <br> Optimization and visualization basics

## Introduction
Both animation and robotics heavily rely on optimization algorithms. In order to understand what is happening inside the optimizer, and to debug efficiently, we must emply *interactive* visualization technique. Interactive means that it is possible to change parameters during runtime and see the change in result immediately, without having to stop, edit, and run again.
In this introductory assignment you will experiment with basic optimization and visualization techniques. The goal is to introduce you an important, different way of coding that is geared toward interactive techniques. This will be important in the rest of the class.

## Instructions
Preliminary steps:
1. Make a Github account and get your Github student benefits at:
   
   `https://education.github.com/discount_requests/application`

   You will need to use your University email and have a scan of an up-to-date student card.
2. Install Chocolatey, and use it to install Python and VS Code.
3. Open VS Code and install the Python extension (`CTRL-SHIFT-X` and search `python` and then `install`), and Jupyter extension.
4. Install the Github Copilot extension and log in to it.

Setup steps
1. Create a folder with no spaces and no non-english characters (Preferably in `C:\Courses\AnimationAndRobotics\Assignments\`) and clone the assignment repository with `git clone`:

    `git clone https://github.com/HaifaGraphicsCourses/animation-and-robotics-basics-[your github id]`
    
    This will create a new folder that contains this repository.
2. Open the folder with VS Code.
3. Create a new Python environment (`CTRL-SHIFT-P`, type `python env` and select `Python: Create Environment`). Follow the steps. VS Code should create a new folder called `.venv`.
4. Open a new terminal (`` CTRL-SHIFT-` ``). If VS Code detected the Python environment correcly, the prompt should begin with `(.venv)`. If not, restart VS Code and try again. If it still doesn't make sure the default terminal is `cmd` or `bash` (use `CTRL-SHIFT-P` and then `Terminal: Select Default Profile` to change it) and start a new terminal. If it still fails, ask for help.
5. Install Vedo, a scientific visualization package for python, using `pip install vedo` in the terminal.
6. Open `Assignment1.py`. The file is divided into cell, where each cell is defined by `#%%`. Run the first cell, which has this code, but pressing `CTRL-ENTER`.
   ```python
    #%% Imports
    import vedo as vd
    import numpy as np
    from vedo.pyplot import plot
    from vedo import Latex

    vd.settings.default_backend= 'vtk'
    ```
    On the first run, VS Code will tell you that it needs to install the ipykernel runtime.
7.  Run the whole file, cell-by-cell, by pressing `SHIFT-ENTER`. Running the last cell should result in a new window appearing, with a surface on it.
8.  Congrats! You can start working now!
## Tasks
You are required to write a report in a markdown format (`.md`). The report will be check semi-automatically, so it is important you follow these steps:
1. Create an .md file with the work `report` in its name (e.g. `report.md`) and put it in the root path of the repository.
2. Put your full name and ID number somewhere in the report.

Before you begin, take a moment to interact with the GUI. Try to change the view with the mouse. As you move the mouse on the surface or under it, you will create a path. You can also observe the text flag and the text on the bottom left. The bottom right has a slider that changes the surface's alpha, also known as transparancy. Pressing `c` on the keyboard will reset the path. Try to press other buttons and see what happens. For further information, check the Vedo documentation.
### Task 1: Understand the code
The code is divided to several cells, each with its own purpose. We will use [Vedo](https://vedo.embl.es/) to create the GUI. The Imports cell does the package imports for python. The Callbacks cell contains GUI callbacks. These functions are called in a responce to a predefined user action, such as moving the mouse or pressing a button on the keyboard. The Optimization cell contains functions the will be required later for optimization. The Plotting cell contains calls to plotting functions.

To check your understanding, add the following to the code:
1. A callback for a mouse right button press. Choose what will happen and report on it. Be creative.
2. Add a graph of the function values on the path (the numpy array `Xi`) you made with the mouse. The graph should update as the path extends. Use `vedo.plot` and report what happened. Use `vedo.plot.clone2d` to fix it.

### Task 2: Optimize
The code includes rudimentary optimization functions. Follow these steps:
1. Change the code such that the mouse does not cause a path to form anymore. Do not remove the path code yet, as it will be used to plot the progression of the optimization.
2. Add a left mouse button click callback that, when clicked on the surface or the plane under it, clears the path and sets the current optimization candidate to that position
3. Add a button that runs a single gradient descent step and updates the path (the last point in the path should be the last candidate)
4. Add a button that runs a single Newton's step.

Don't forget the graph from the previous task. It should still be visible and show the values of the objective along the path. 
Attach a few pictures to the report and explain what they show.

### Task 3: Evaluate
We regularly need to know which method performs better. To test this, we need to compare methods in terms of speed and convergence rate.
1. Change the code such that is maintains two paths, one for gradient descent steps and one for Newton steps. when pressing with the gradient or Newton's button, it will only advance the appropriate one. In addition, the plot of the objective should present both paths.
2. Select several points, and plot the value of the objective after enough iterations are performed such that the is no visible difference between iterations. Put the results in your report.
3. (Optional) How many iterations are needed? Devise an automatic way to decide when to stop iterating and report on your approach.
4. (Optional) The gradients and Hessians in the code are computed *numerically* using finite differences. This is a slow but simple way to obtain them. The alternative is to compute them *analytically* by hand, using basic calculus (or to use automatic differentiation). Write two new functions that compute the gradient and Hessian of the objective analytically and copy the code to your report.
5. (Optional) Measure the time it takes to run the numerical vs. analytical computation.
6. (Optional) The finite different approximation relies on a finite epsilon. Compare the values of the finite difference gradients for different epsilons with the analytical (true) value. 

## Submission
Place the report in a file names `report.md` in the root directory of the repository.

## Grading
Gradient will be done in a special Github classroom issue. Do not change that issue.
