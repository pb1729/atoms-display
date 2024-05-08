# Atoms Display

Simple program to visualize molecular dynamics simulations.

Atomic numbers and atom positions are taken as simple numpy arrays of shape `[n]` and `[n, 3]` respectively.

### Features:

* Automatically launch the visualization in a new thread: `display = launch_atom_display(atomic_numbers, positions)`
* Update atom positions currently showing in the display: `display.update_pos(new_positions)`
* Re-center the view on the average atom position: `display.center_pos()`
* Rotate the view: Click and drag in the display, rotation will update on releasing the mouse button.





