from threading import Thread
from queue import Queue
import time

import numpy as np

from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *


"""
      ATOMS DISPLAY

Very simple and bare-bones code for displaying molecules. Key distinguishing feature is that molecules
can have their configurations updated in approximately real time, and the display will be correspondingly
animated.

ACKNOWLEDGEMENTS:
* We thank Rick Muller for the following recipe for using OpenGL in python:
    -> https://code.activestate.com/recipes/325391-open-a-glut-window-and-draw-a-sphere-using-pythono/
* We thank the authors of the Mogli project for their work:
    -> https://github.com/sciapp/mogli
Code in this file is partially derived from these programs.
"""


# Constants:
WINDOWNAME = "atoms display"
WIDTH = 600
HEIGHT = 600
LIGHT_BRIGHTNESS = 1.0
CAM_STEP_BACK_DIST = 100.
UPDATE_INTERVAL = 0.01 # [s]

MISSING_GLUT_ERRMSG = """ Null Function Exception:
Could not find the function glutInit(). This error is usually caused by not having Freeglut
installed. To obtain this software, go to their website: https://freeglut.sourceforge.net/
Or on Ubuntu you can run: sudo apt-get install freeglut3-dev
"""

# Atom color rgb tuples (used for rendering, may be changed by users)
ATOM_COLORS = np.array([(0, 0, 0),  # No element 0
                        (255, 255, 255), (217, 255, 255), (204, 128, 255),
                        (194, 255, 0), (255, 181, 181), (144, 144, 144),
                        (48, 80, 248), (255, 13, 13), (144, 224, 80),
                        (179, 227, 245), (171, 92, 242), (138, 255, 0),
                        (191, 166, 166), (240, 200, 160), (255, 128, 0),
                        (255, 255, 48), (31, 240, 31), (128, 209, 227),
                        (143, 64, 212), (61, 225, 0), (230, 230, 230),
                        (191, 194, 199), (166, 166, 171), (138, 153, 199),
                        (156, 122, 199), (224, 102, 51), (240, 144, 160),
                        (80, 208, 80), (200, 128, 51), (125, 128, 176),
                        (194, 143, 143), (102, 143, 143), (189, 128, 227),
                        (225, 161, 0), (166, 41, 41), (92, 184, 209),
                        (112, 46, 176), (0, 255, 0), (148, 255, 255),
                        (148, 224, 224), (115, 194, 201), (84, 181, 181),
                        (59, 158, 158), (36, 143, 143), (10, 125, 140),
                        (0, 105, 133), (192, 192, 192), (255, 217, 143),
                        (166, 117, 115), (102, 128, 128), (158, 99, 181),
                        (212, 122, 0), (148, 0, 148), (66, 158, 176),
                        (87, 23, 143), (0, 201, 0), (112, 212, 255),
                        (255, 255, 199), (217, 225, 199), (199, 225, 199),
                        (163, 225, 199), (143, 225, 199), (97, 225, 199),
                        (69, 225, 199), (48, 225, 199), (31, 225, 199),
                        (0, 225, 156), (0, 230, 117), (0, 212, 82),
                        (0, 191, 56), (0, 171, 36), (77, 194, 255),
                        (77, 166, 255), (33, 148, 214), (38, 125, 171),
                        (38, 102, 150), (23, 84, 135), (208, 208, 224),
                        (255, 209, 35), (184, 184, 208), (166, 84, 77),
                        (87, 89, 97), (158, 79, 181), (171, 92, 0),
                        (117, 79, 69), (66, 130, 150), (66, 0, 102),
                        (0, 125, 0), (112, 171, 250), (0, 186, 255),
                        (0, 161, 255), (0, 143, 255), (0, 128, 255),
                        (0, 107, 255), (84, 92, 242), (120, 92, 227),
                        (138, 79, 227), (161, 54, 212), (179, 31, 212),
                        (179, 31, 186), (179, 13, 166), (189, 13, 135),
                        (199, 0, 102), (204, 0, 89), (209, 0, 79),
                        (217, 0, 69), (224, 0, 56), (230, 0, 46),
                        (235, 0, 38), (255, 0, 255), (255, 0, 255),
                        (255, 0, 255), (255, 0, 255), (255, 0, 255),
                        (255, 0, 255), (255, 0, 255), (255, 0, 255),
                        (255, 0, 255)], dtype=np.float32)/255.0
# Atomic numbers mapped to their symbols
ATOMIC_NUMBERS = {"H": 1, "HE": 2, "LI": 3, "BE": 4, "B": 5, "C": 6, "N": 7,
                "O": 8, "F": 9, "NE": 10, "NA": 11, "MG": 12, "AL": 13,
                "SI": 14, "P": 15, "S": 16, "CL": 17, "AR": 18, "K": 19,
                "CA": 20, "SC": 21, "TI": 22, "V": 23, "CR": 24, "MN": 25,
                "FE": 26, "CO": 27, "NI": 28, "CU": 29, "ZN": 30, "GA": 31,
                "GE": 32, "AS": 33, "SE": 34, "BR": 35, "KR": 36, "RB": 37,
                "SR": 38, "Y": 39, "ZR": 40, "NB": 41, "MO": 42, "TC": 43,
                "RU": 44, "RH": 45, "PD": 46, "AG": 47, "CD": 48, "IN": 49,
                "SN": 50, "SB": 51, "TE": 52, "I": 53, "XE": 54, "CS": 55,
                "BA": 56, "LA": 57, "CE": 58, "PR": 59, "ND": 60, "PM": 61,
                "SM": 62, "EU": 63, "GD": 64, "TB": 65, "DY": 66, "HO": 67,
                "ER": 68, "TM": 69, "YB": 70, "LU": 71, "HF": 72, "TA": 73,
                "W": 74, "RE": 75, "OS": 76, "IR": 77, "PT": 78, "AU": 79,
                "HG": 80, "TL": 81, "PB": 82, "BI": 83, "PO": 84, "AT": 85,
                "RN": 86, "FR": 87, "RA": 88, "AC": 89, "TH": 90, "PA": 91,
                "U": 92, "NP": 93, "PU": 94, "AM": 95, "CM": 96, "BK": 97,
                "CF": 98, "ES": 99, "FM": 100, "MD": 101, "NO": 102,
                "LR": 103, "RF": 104, "DB": 105, "SG": 106, "BH": 107,
                "HS": 108, "MT": 109, "DS": 110, "RG": 111, "CN": 112,
                "UUB": 112, "UUT": 113, "UUQ": 114, "UUP": 115, "UUH": 116,
                "UUS": 117, "UUO": 118}

# Atom valence radii in Å (used for bond calculation)
ATOM_VALENCE_RADII = np.array([0,  # No element 0
                               230, 930, 680, 350, 830, 680, 680, 680, 640,
                               1120, 970, 1100, 1350, 1200, 750, 1020, 990,
                               1570, 1330, 990, 1440, 1470, 1330, 1350, 1350,
                               1340, 1330, 1500, 1520, 1450, 1220, 1170, 1210,
                               1220, 1210, 1910, 1470, 1120, 1780, 1560, 1480,
                               1470, 1350, 1400, 1450, 1500, 1590, 1690, 1630,
                               1460, 1460, 1470, 1400, 1980, 1670, 1340, 1870,
                               1830, 1820, 1810, 1800, 1800, 1990, 1790, 1760,
                               1750, 1740, 1730, 1720, 1940, 1720, 1570, 1430,
                               1370, 1350, 1370, 1320, 1500, 1500, 1700, 1550,
                               1540, 1540, 1680, 1700, 2400, 2000, 1900, 1880,
                               1790, 1610, 1580, 1550, 1530, 1510, 1500, 1500,
                               1500, 1500, 1500, 1500, 1500, 1500, 1600, 1600,
                               1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600,
                               1600, 1600, 1600, 1600, 1600, 1600],
                              dtype=np.float32)/1000.0
ATOM_VALENCE_RADII.flags.writeable = False


def look_at(eye, center, up):
  gluLookAt(*eye, *center, *up)

def draw_sphere(rad, pos, color):
  glPushMatrix()
  glTranslated(*pos)
  glMaterialfv(GL_FRONT,GL_DIFFUSE,color)
  glutSolidSphere(rad, 9, 6)
  glPopMatrix()

def create_rotation_matrix(angle, x, y, z):
    """ Creates a 3x3 rotation matrix. """
    if np.linalg.norm((x, y, z)) < 0.0001:
        return np.eye(3, dtype=np.float32)
    x, y, z = np.array((x, y, z))/np.linalg.norm((x, y, z))
    matrix = np.zeros((3, 3), dtype=np.float32)
    cos = np.cos(angle)
    sin = np.sin(angle)
    matrix[0, 0] = x*x*(1-cos)+cos
    matrix[1, 0] = x*y*(1-cos)+sin*z
    matrix[0, 1] = x*y*(1-cos)-sin*z
    matrix[2, 0] = x*z*(1-cos)-sin*y
    matrix[0, 2] = x*z*(1-cos)+sin*y
    matrix[1, 1] = y*y*(1-cos)+cos
    matrix[1, 2] = y*z*(1-cos)-sin*x
    matrix[2, 1] = y*z*(1-cos)+sin*x
    matrix[2, 2] = z*z*(1-cos)+cos
    return matrix

class AtomDisplay:
  """ A class representing a window displaying a molecule! """
  EVENT_UPDATE_POS = 1
  EVENT_CENTER_POS = 2
  def __init__(self, atomic_numbers, positions, radii_scale=1.0):
    self._positions = np.copy(positions)
    self._radii = radii_scale*self._get_radii(atomic_numbers)
    self._colors = self._get_colors(atomic_numbers)
    self._previous_mouse_position = None
    self._camera = np.array([0., 0., CAM_STEP_BACK_DIST]), np.array([0., 0., 0.]), np.array([0., 1., 0.])
    self._window_id = None
    self._events = Queue()
    self._active = True
  def _get_colors(self, atomic_numbers):
    return ATOM_COLORS[atomic_numbers]
  def _get_radii(self, atomic_numbers):
    return ATOM_VALENCE_RADII[atomic_numbers]
  def _make_window(self):
    assert self._window_id is None
    try:
      glutInit("dummy_script_name")
    except OpenGL.error.NullFunctionError as e:
      raise RuntimeError(MISSING_GLUT_ERRMSG) from e
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(WIDTH, HEIGHT)
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION) # allow user to close display without ending program
    self._window_id = glutCreateWindow(WINDOWNAME)
  def _setup_graphics(self):
    glClearColor(0.,0.,0.,1.)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
  def _setup_lighting(self):
    lightZeroPosition = [2*CAM_STEP_BACK_DIST, 0., 2*CAM_STEP_BACK_DIST, 1.]
    lightZeroColor = 3*[LIGHT_BRIGHTNESS] + [1.0]
    glLightfv(GL_LIGHT0, GL_POSITION, lightZeroPosition)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightZeroColor)
    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 1.0)
    glEnable(GL_LIGHT0)
  def _setup_camera(self):
    glMatrixMode(GL_PROJECTION)
    gluPerspective(40., 1., 1., 4*CAM_STEP_BACK_DIST)
    glMatrixMode(GL_MODELVIEW)
  def _register_callbacks(self):
    glutMouseFunc(self._get_mouse_click_func())
    glutKeyboardFunc(self._get_keyboard_func())
    glutCloseFunc(self._get_cleanup())
  def _get_display(self):
    def display():
      glutSetWindow(self._window_id)
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
      glPushMatrix()
      look_at(*self._camera)
      for rad, pos, col in zip(self._radii, self._positions, self._colors):
        draw_sphere(rad, pos, col)
      glutSwapBuffers()
      glPopMatrix()
    return display
  def _get_keyboard_func(self):
    active_keys = "exwsdaio"
    key_axes = [0, 0, 1, 1, 2, 2, 0, 0]
    key_signs = [1, -1, 1, -1, 1, -1, 0, 0]
    zoom_factors = [1.]*6 + [0.9, 1/0.9]
    def keyboard(keycode, x, y):
      key = active_keys.find(chr(ord(keycode)))
      if key != -1:
        eye, center, up = self._camera
        forward = center - eye
        right = np.cross(forward, up)
        forward_mag = np.linalg.norm(forward)
        unit_directions = [forward/forward_mag,
          up/np.linalg.norm(up),
          right/np.linalg.norm(right)]
        scale = forward_mag/10
        center += scale*unit_directions[key_axes[key]]*key_signs[key]
        forward *= zoom_factors[key]
        eye = center - forward
        self._camera = eye, center, up
        self._redisplay()
    return keyboard
  def _get_mouse_click_func(self):
    def mouse_click(btn, state, x, y):
      if state == GLUT_DOWN:
        if btn == GLUT_LEFT_BUTTON:
          self._previous_mouse_position = x, y
      elif state == GLUT_UP:
        if btn == GLUT_LEFT_BUTTON:
          self._drag_rotate(*self._previous_mouse_position, x, y)
          self._redisplay()
    return mouse_click
  def _get_cleanup(self):
    def cleanup():
      self._active = False
      glutSetWindow(self._window_id)
      glutHideWindow()
    return cleanup
  def _drag_rotate(self, x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    rotation_intensity = np.linalg.norm((dx, dy)) / WIDTH
    eye, center, up = self._camera
    camera_distance = np.linalg.norm(center-eye)
    forward = (center-eye)/camera_distance
    right = np.cross(forward, up)
    rotation_axis = (up*dx+right*dy)
    rotation_matrix = create_rotation_matrix(-rotation_intensity,
      rotation_axis[0], rotation_axis[1], rotation_axis[2])
    forward = np.dot(rotation_matrix, forward)
    up = np.dot(rotation_matrix, up)
    eye = center-forward*camera_distance
    self._camera = eye, center, up
  def _redisplay(self):
    glutSetWindow(self._window_id)
    glutPostRedisplay()
  def _pop_events(self):
    while not self._events.empty():
      event_type, event_data = self._events.get()
      if   event_type == self.EVENT_UPDATE_POS: self._do_update_pos(event_data)
      elif event_type == self.EVENT_CENTER_POS: self._do_center_pos(event_data)
      else: assert False
  def _do_center_pos(self, _):
    eye, center, up = self._camera
    forward = center - eye
    center = self._positions.mean(0)
    eye = center - forward
    self._camera = eye, center, up
    self._redisplay()
  def _do_update_pos(self, newpos):
    self._positions = newpos
    self._redisplay()
  def launch(self):
    self._make_window()
    self._setup_graphics()
    self._setup_lighting()
    self._setup_camera()
    self._register_callbacks()
    # main loop:
    glutDisplayFunc(self._get_display())
    i = 0
    while self._active:
      glutMainLoopEvent()
      self._pop_events()
      time.sleep(UPDATE_INTERVAL)
  def update_pos(self, newpos):
    assert self._active, "window has already been closed!"
    self._events.put((self.EVENT_UPDATE_POS, np.copy(newpos)))
  def center_pos(self):
    assert self._active, "window has already been closed!"
    self._events.put((self.EVENT_CENTER_POS, None))



def parse_xyz(xyz_lines):
  natoms = int(xyz_lines[0])
  xyz_lines = xyz_lines[2:] # omit the "comment line" of xyz file
  assert natoms >= len(xyz_lines)
  xyz_lines = xyz_lines[:natoms]
  atomic_numbers = np.zeros((natoms,), dtype=int)
  positions = np.zeros((natoms, 3), dtype=float)
  for i in range(natoms):
    data = xyz_lines[i].split()
    atomic_numbers[i] = ATOMIC_NUMBERS[data[0]]
    for j in range(3):
      positions[i, j] = float(data[1+j])
  return atomic_numbers, positions


def read_xyz(path):
  with open(path, "r") as f:
    return parse_xyz(f.readlines())


def launch_atom_display(atomic_numbers, positions, **kwargs):
  disp = AtomDisplay(atomic_numbers, positions, **kwargs)
  t = Thread(target=(lambda: disp.launch()))
  t.start()
  disp.center_pos()
  return disp


if __name__ == "__main__":
  from sys import argv
  atomic_numbers, positions = read_xyz(argv[1])
  launch_atom_display(atomic_numbers, positions)
