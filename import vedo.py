import vedo

def print_mouse_button(evt):
    # Assuming evt has an attribute 'button' revealed by the above diagnostic
    if hasattr(evt, 'button'):
        print("Mouse button pressed:", evt.button)
    else:
        print("Event object does not contain 'button' attribute")

plt = vedo.Plotter()
plt.add_callback('mouse click', print_mouse_button)
plt.show()
