#%% imports
import vedo as vd
import numpy as np
from vedo.pyplot import plot

vd.settings.default_backend = 'vtk'

#%% Callbacks


def OnMouseMove(evt):                ### called every time mouse moves!
    global Xi, Xi_grad, Xi_newton

    if evt.object is None:          # mouse hits nothing, return.
        return                       

    pt  = evt.picked3d               # 3d coords of point under mouse
    X   = np.array([pt[0],pt[1],objective([pt[0],pt[1]])])  # X = (x,y,e(x,y))
    Xi_grad = np.append(Xi_grad,[X],axis=0)             # append to the list of points for gradient step
    Xi_newton = np.append(Xi_newton,[X],axis=0)             # append to the list of points for newton step
    
    if len(Xi_grad) > 1:               # need at least two points to compute a distance
        txt =(
            f"X:  {vd.precision(X,2)}\n"
            f"dX: {vd.precision(Xi_grad[-1,0:2] - Xi_grad[-2,0:2],2)}\n"
            f"dE: {vd.precision(Xi_grad[-1,2] - Xi_grad[-2,2],2)}\n"
        )
        ar = vd.Arrow(Xi_grad[-2,:], Xi_grad[-1,:], s=0.001, c='orange5')
        plt.add(ar) # add the arrow
    else:
        txt = f"X: {vd.precision(X,2)}"
    msg.text(txt)                    # update text message

    c = vd.Cylinder([np.append(Xi_grad[-1,0:2], 0.0), Xi_grad[-1,:]], r=0.01, c='orange5')
    plt.remove("Cylinder")    
    fp = fplt3d[0].flagpole(txt, point=X,s=0.08, c='k', font="Quikhand")
    fp.follow_camera()                 # make it always face the camera
    plt.remove("FlagPole") # remove the old flagpole

    plt.add(fp, c) # add the new flagpole and new cylinder
    plt.render()   # re-render the scene

def OnLeftClick(evt):
    if evt.actor:
        click_pos = evt.picked3d
        if click_pos is not None:
            print("Left click event")
            global Xi, Xi_newton, Xi_grad, newton_prev_pathplot, grad_prev_pathplot
            
            # Convert clicked position to array with objective value
            new_point = np.array([click_pos[0], click_pos[1], objective([click_pos[0], click_pos[1]])])
            if Xi.size > 0:  # Check if Xi is not empty
                Xi = np.empty((0, 3))  # Clear the path points for gradient step
            if Xi_newton.size > 0:
                Xi_newton = np.empty((0, 3))
            if Xi_grad.size > 0:
                Xi_grad = np.empty((0, 3))
                
            # Insert the new point into Xi arrays
            Xi = np.append(Xi, [new_point], axis=0)
            Xi_newton = np.append(Xi_newton, [new_point], axis=0)
            Xi_grad = np.append(Xi_grad, [new_point], axis=0)

            # Remove all arrows
            plt.remove("Arrow").render()
            # Remove the previous path plots if they exist
            if newton_prev_pathplot is not None:
                plt.remove(newton_prev_pathplot).render()
            if grad_prev_pathplot is not None:
                plt.remove(grad_prev_pathplot).render()
            
            # Set the current optimization candidate to the clicked position
            global current_optimization_candidate
            current_optimization_candidate = click_pos[:2]
            print(f"Optimization candidate set to: {current_optimization_candidate}")
            
            if len(Xi) > 1:
                print(len(Xi))

            update_path_plot()  #update the path plot on the mesh

def update_path_plot():     #update the mesh plot
    global Xi
    print("Updating path plot")
    txt = f"X: {vd.precision(Xi[-1], 2)}"
    msg.text(txt)  # update text message
    c = vd.Cylinder([np.append(Xi[-1, 0:2], 0.0), Xi[-1, :]], r=0.01, c='orange5')
    plt.remove("Cylinder")
    fp = fplt3d[0].flagpole(txt, point=Xi[-1], s=0.08, c='k', font="Quikhand")
    fp.follow_camera()  # make it always face the camera
    plt.remove("FlagPole")  # remove the old flagpole
    plt.add(fp, c)  # add the new flagpole and new cylinder
    plt.render()  # re-render the scene

def update_path_plot_grad():
    print("Updating gradient descent plot")
    global Xi_grad, grad_prev_pathplot
    if len(Xi_grad) > 1:  # need at least two points to compute a distance
        txt = (
            f"X:  {vd.precision(Xi_grad[-1], 2)}\n"
            f"dX: {vd.precision(Xi_grad[-1, 0:2] - Xi_grad[-2, 0:2], 2)}\n"
            f"dE: {vd.precision(Xi_grad[-1, 2] - Xi_grad[-2, 2], 2)}\n"
        )
        ar = vd.Arrow(Xi_grad[-2, :], Xi_grad[-1, :], s=0.001, c='orange5')
        plt.add(ar)  # add the arrow to the plot
        if grad_prev_pathplot is not None:
            plt.remove(grad_prev_pathplot)
        grad_pathplot = plot(np.arange(len(Xi_grad)), Xi_grad[:, 2], lw=3, c='orange5', title="Gradient_Step", xtitle="Point", ytitle="Value", marker='o')
        grad_pathplot = grad_pathplot.clone2d(pos='top-left')
        grad_prev_pathplot = grad_pathplot
        plt.add(grad_pathplot)
    else:
        txt = f"X: {vd.precision(Xi_grad[-1], 2)}"
    msg.text(txt)  # update text message

    c = vd.Cylinder([np.append(Xi_grad[-1, 0:2], 0.0), Xi_grad[-1, :]], r=0.01, c='orange5')
    plt.remove("Cylinder")
    fp = fplt3d[0].flagpole(txt, point=Xi_grad[-1], s=0.08, c='k', font="Quikhand")
    fp.follow_camera()  # make it always face the camera
    plt.remove("FlagPole")  # remove the old flagpole

    plt.add(fp, c)  # add the new flagpole and new cylinder
    plt.render()  # re-render the scene

def update_path_plot_newton():
    global Xi_newton, newton_prev_pathplot
    if len(Xi_newton) > 1:  # need at least two points to compute a distance
        txt = (
            f"X:  {vd.precision(Xi_newton[-1], 2)}\n"
            f"dX: {vd.precision(Xi_newton[-1, 0:2] - Xi_newton[-2, 0:2], 2)}\n"
            f"dE: {vd.precision(Xi_newton[-1, 2] - Xi_newton[-2, 2], 2)}\n"
        )
        ar = vd.Arrow(Xi_newton[-2, :], Xi_newton[-1, :], s=0.001, c='red5')
        plt.add(ar)  # add the arrow to the plot
        if newton_prev_pathplot is not None:
            plt.remove(newton_prev_pathplot)
        newton_pathplot = plot(np.arange(len(Xi_newton)), Xi_newton[:, 2], lw=3, c='red5', title="Newton_Step", xtitle="Point", ytitle="Value", marker='o')   
        newton_pathplot = newton_pathplot.clone2d(pos='top-right')
        newton_prev_pathplot = newton_pathplot
        plt.add(newton_pathplot)
    else:
        txt = f"X: {vd.precision(Xi_newton[-1], 2)}"
    msg.text(txt)  # update text message
    c = vd.Cylinder([np.append(Xi_newton[-1, 0:2], 0.0), Xi_newton[-1, :]], r=0.01, c='orange5')
    plt.remove("Cylinder")
    fp = fplt3d[0].flagpole(txt, point=Xi_newton[-1], s=0.08, c='k', font="Quikhand")
    fp.follow_camera()  # make it always face the camera
    plt.remove("FlagPole")  # remove the old flagpole

    plt.add(fp, c)  # add the new flagpole and new cylinder
    plt.render()  # re-render the scene


def OnSliderAlpha(widget, event):  ### called every time the slider is moved
    val = widget.value  # get the slider value
    fplt3d[0].alpha(val)  # set the alpha (transparency) value of the surface
    fplt3d[1].alpha(val)  # set the alpha (transparency) value of the isolines

def OnKeyPress(evt):  ### called every time a key is pressed
    if evt.keypress in ['c', 'C']:  # reset Xi_grad, Xi_newton and the arrows
        global Xi_grad, Xi_newton
        Xi_grad = np.empty((0, 3))
        Xi_newton = np.empty((0, 3))
        plt.remove("Arrow").render()
        
def OnGradButtonPress(widget, event):  ### Define the button callback function with the correct signature
    global current_optimization_candidate, Xi_grad, grad_clicked
    if Xi_grad.size > 0:  # Ensure there is a point to perform gradient descent
        new_point = step(objective, current_optimization_candidate, gradient_direction, bounds)
        new_point = np.array([new_point[0], new_point[1], objective(new_point)])
        current_optimization_candidate = new_point[:2]
        Xi_grad = np.append(Xi_grad, [new_point], axis=0)
        update_path_plot_grad()

def OnNewtonButtonPress(widget, event):  ### Define the button callback function with the correct signature
    global current_optimization_candidate, Xi_newton, newton_clicked
    #newton_clicked = True     # Set the Newton step button clicked flag to true 
    print("Newton step button pressed")
    if Xi_newton.size > 0:  # Ensure there is a point to perform Newton step
        new_point = step(objective, current_optimization_candidate, Newton_direction, bounds)
        new_point = np.array([new_point[0], new_point[1], objective(new_point)])
        current_optimization_candidate = new_point[:2]
        Xi_newton = np.append(Xi_newton, [new_point], axis=0)
        update_path_plot_newton()

#%% Optimization functions
#calculates the gradient of a given function at a specific point
def gradient_fd(func, X, h=0.001):  # finite difference gradient
    x, y = X[0], X[1]
    gx = (func([x + h, y]) - func([x - h, y])) / (2 * h)
    gy = (func([x, y + h]) - func([x, y - h])) / (2 * h)
    return gx, gy

def Hessian_fd(func, X, h=0.001):  # finite difference Hessian
    x, y = X[0], X[1]
    gxx = (func([x + h, y]) - 2 * func([x, y]) + func([x - h, y])) / h ** 2
    gyy = (func([x, y + h]) - 2 * func([x, y]) + func([x, y - h])) / h ** 2
    gxy = (func([x + h, y + h]) - func([x + h, y - h]) - func([x - h, y + h]) + func([x - h, y - h])) / (4 * h ** 2)
    H = np.array([[gxx, gxy], [gxy, gyy]])
    return H

def gradient_direction(func, X):  # compute gradient step direction
    g = gradient_fd(func, X)
    return -np.array(g)

def Newton_direction(func, X):  # compute Newton step direction
    g = gradient_fd(func, X)
    H = Hessian_fd(func, X)
    d = -np.linalg.solve(H, np.array(g))
    return np.array(d[0], d[1])

def boundary_check(X, bounds):
    X[0] = max(min(X[0], bounds[0][1]), bounds[0][0])
    X[1] = max(min(X[1], bounds[1][1]), bounds[1][0])
    if len(X) > 2:
        X[2] = max(min(X[2], bounds[2][1]), bounds[2][0])  # z-axis boundary check
    return X

def line_search(func, X, d):
    alpha = 0.5
    while func(X + d * alpha) > func(X):  # If the function value does not decrease, reduce alpha
        alpha *= 0.5  # by half and try again
    return alpha

def step(func, X, search_direction_function, bounds):
    d = search_direction_function(func, X)
    alpha = line_search(func, X, d)
    new_X = X + d * alpha
    new_X = boundary_check(new_X, bounds)
    return new_X

def optimize(func, X, search_direction_function, tol=1e-6, iter_max=10, bounds=None):
    for i in range(iter_max):
        X = step(func, X, search_direction_function, bounds)
        if np.linalg.norm(gradient_fd(func, X)) < tol:
            break
    return X
#%% Callbacks

# Function to create a Magen David shape
def create_magen_david():
    points1 = np.array([[0, 1, 0], [np.sin(np.pi/3), -0.5, 0], [-np.sin(np.pi/3), -0.5, 0]])
    points2 = np.array([[0, -1, 0], [-np.sin(np.pi/3), 0.5, 0], [np.sin(np.pi/3), 0.5, 0]])
    
    triangle1 = vd.Mesh([points1, [[0, 1, 2]]]).color('blue')
    triangle2 = vd.Mesh([points2, [[0, 1, 2]]]).color('blue')
    
    magen_david = vd.Assembly([triangle1, triangle2])
    return magen_david

magen_david = create_magen_david()

#%% Callbacks


#%% Callbacks
def OnRightClick(evt):
    if evt.actor:
        # Place the Magen David shape at the top-left corner of the screen
        magen_david.scale(0.1).pos(-0.8, 0.8, 0)  # Adjust the scale and position if necessary
        plt.add(magen_david)
        plt.render()
        print(f"Magen David shape added at top-left corner")


def objective(X):
    x, y = X[0], X[1]
    return np.sin(2 * x * y) * np.cos(3 * y) / 2 + 1 / 2

msg = vd.Text2D(pos='bottom-left', font="VictorMono")  # an empty text

#bounds of the mesh
x_min, x_max = 0, 3  
y_min, y_max = 0, 3  
z_min, z_max = 0, 1  
bounds = [(x_min, x_max), (y_min, y_max), (z_min, z_max)]

Xi = np.empty((0, 3))
Xi_grad = np.empty((0, 3))
Xi_newton = np.empty((0, 3))
grad_prev_pathplot = None
grad_pathplot = None
newton_pathplot = None
newton_prev_pathplot = None
# test the optimization functions
X = optimize(objective, [0.6, 0.6], Newton_direction, tol=1e-6, iter_max=100, bounds=bounds)
current_optimization_candidate = np.array([0.6, 0.6])

#%% Plotting

plt = vd.Plotter(bg2='lightblue')  # Create the plotter
fplt3d = plot(lambda x, y: objective([x, y]), c='terrain')  # create a plot from the function e. fplt3d is a list containing surface mesh, isolines, and axis
fplt2d = fplt3d.clone()  # clone the plot to create a 2D plot

fplt2d[0].lighting('off')  # turn off lighting for the 2D plot
fplt2d[0].vertices[:, 2] = 0  # set the z-coordinate of the mesh to 0
fplt2d[1].vertices[:, 2] = 0  # set the z-coordinate of the isolines to 0

#plt.add_callback('mouse move', OnMouseMove)  # add Mouse move callback
# Register the callback for right mouse button click
plt.add_callback('RightButtonPressEvent', OnRightClick)
plt.add_callback('key press', OnKeyPress)  # add Keyboard callback
plt.add_slider(OnSliderAlpha, 0., 1., 1., title="Alpha")  # add a slider for the alpha value of the surface
plt.add_callback('LeftButtonPressEvent', OnLeftClick)  # add left mouse button click callback
plt.add_button(OnGradButtonPress, pos=(0.8, 0.9), states=["Gradient Step"], size=20, c="w", bc="blue")
plt.add_button(OnNewtonButtonPress, pos=(0.8, 0.8), states=["Newton Step"], size=20, c="w", bc="red")
plt.show([fplt3d, fplt2d], msg, __doc__, viewup='z')
plt.close()

