# Atoms Display

Simple program to visualize molecular dynamics simulations.

Atomic numbers and atom positions are taken as simple numpy arrays of shape `[n]` and `[n, 3]` respectively. Atomic numbers are integers, while positions are floats.

### Features:

* Automatically launch the visualization in a new thread: `display = launch_atom_display(atomic_numbers, positions)`
* Update atom positions currently showing in the display: `display.update_pos(new_positions)`
* Re-center the view on the average atom position: `display.center_pos()`
* Rotate the view: Click and drag in the display, rotation will update on releasing the mouse button.
* Zoom in and out: use the `io` keys
* Move through space: use `wasd` to move in plane of the screen and `ex` to move forward and backward.

### Install

Clone the repo and run: `pip install .`

If you want to make changes to the package: `pip install -e .`

You might need to run `sudo apt-get install freeglut3-dev` to get things working.

## ACKNOWLEDGEMENTS:
* We thank Rick Muller for the following recipe for using OpenGL in python:
    -> https://code.activestate.com/recipes/325391-open-a-glut-window-and-draw-a-sphere-using-pythono/
* We thank the authors of the Mogli project for their work:
    -> https://github.com/sciapp/mogli

Code in this project is partially derived from these programs.

