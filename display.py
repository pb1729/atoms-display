import threading

import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective

from quaternion import Quat


# ------------------ define Display base class ------------------

class Display:
  """ Display base class """
  def __init__(self, width:int, height:int, title:str,
      move_step=0.02, start_args=None):
    # create window:
    self.win = glfw.create_window(width, height, title, None, None)
    if not self.win:
        glfw.terminate(); raise SystemExit("Window create failed")
    # world state:
    self.pos = np.array([0.0, 0.0, 5.0])    # player position in world
    self.view = Quat()            # camera orientation (mouse look)
    self.cam_back = 6.0           # camera distance behind the player
    self.dragging = False
    self.last_xy = (0.0, 0.0)
    self.aspect = width / height
    self.dirty = True             # redraw needed
    self.reset_proj = True        # need to reset the projection
    # store kwargs:
    self.move_step = move_step
    # do any user-defined initialization we need:
    if start_args is None: start_args = []
    glfw.make_context_current(self.win) # make the context current so we can create buffers inside start() if needed
    self.start(*start_args)
    # rest of setup:
    glfw.make_context_current(self.win)
    glfw.swap_interval(1)
    self._set_callbacks()
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.05, 0.07, 0.10, 1.0)
  def _get_wh(self):
    return glfw.get_window_size(self.win)
  def _set_callbacks(self):
    def on_resize(window, w, h):
      h = max(1, h)
      glViewport(0, 0, w, h)
      self.aspect = w / float(h)
      self.reset_proj = True
      self.dirty = True
    def on_mouse_button(window, button, action, mods):
      if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
          self.dragging = True
          self.last_xy = glfw.get_cursor_pos(window)
        elif action == glfw.RELEASE:
          self.dragging = False
        self.dirty = True
    def on_cursor_pos(window, x, y):
      if not self.dragging: return
      lx, ly = self.last_xy
      dx, dy = x - lx, y - ly
      self.last_xy = (x, y)
      magnitude = (dx**2 + dy**2)**0.5
      drag_dir = self.view.rotate_vec3((-dy, -dx, 0)) # pre-emptively cross (dx, -dy, 0) with (0, 0, 1)
      rotate = Quat.from_axis_angle(drag_dir, magnitude*0.01)
      self.view = rotate * self.view                  # multiply to update state
      self.dirty = True
    def on_key(window, key, sc, action, mods):
      if action in (glfw.PRESS, glfw.REPEAT) and key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)
    glfw.set_window_size_callback(self.win, on_resize)
    glfw.set_mouse_button_callback(self.win, on_mouse_button)
    glfw.set_cursor_pos_callback(self.win, on_cursor_pos)
    glfw.set_key_callback(self.win, on_key)
  def _handle_continuous_input(self):
    move = np.zeros(3)
    zoom = 0.
    if glfw.get_key(self.win, glfw.KEY_W) == glfw.PRESS: move[2] -= self.move_step   # forward
    if glfw.get_key(self.win, glfw.KEY_S) == glfw.PRESS: move[2] += self.move_step   # back
    if glfw.get_key(self.win, glfw.KEY_A) == glfw.PRESS: move[0] -= self.move_step   # left
    if glfw.get_key(self.win, glfw.KEY_D) == glfw.PRESS: move[0] += self.move_step   # right
    if glfw.get_key(self.win, glfw.KEY_E) == glfw.PRESS: move[1] += self.move_step   # up
    if glfw.get_key(self.win, glfw.KEY_X) == glfw.PRESS: move[1] -= self.move_step   # down
    if glfw.get_key(self.win, glfw.KEY_I) == glfw.PRESS: zoom -= 0.05 # zoom in
    if glfw.get_key(self.win, glfw.KEY_O) == glfw.PRESS: zoom += 0.05 # zoom out
    if np.linalg.norm(move) > 0.0:
      dpos = self.view.rotate_vec3(move*self.cam_back) # scale movement amount to current zoom level
      self.pos += dpos
      self.dirty = True
    if zoom != 0:
      self.cam_back *= 1.0 + zoom
      self.dirty = True
  def _set_projection(self):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity() # this line seems to be necessary for some reason
    gluPerspective(60.0, self.aspect, 0.1, 1000.0)
    glMatrixMode(GL_MODELVIEW)
  def _draw_all(self):
    # reset canvas
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    # setup transforms
    glLoadIdentity()
    forward = self.view.rotate_vec3((0, 0, -1)) # forward direction for the player
    cam_pos = self.pos - self.cam_back*forward
    view_conj = self.view.conj()
    glMultMatrixf(view_conj.to_mat4())
    glTranslatef(-cam_pos[0], -cam_pos[1], -cam_pos[2])
    # call subclass draw method
    self.draw()
    # draw player marker
    glDisable(GL_LIGHTING) # turn off lighting for drawing player marker!
    glPointSize(8)
    glBegin(GL_POINTS)
    glColor3f(1.0, 1.0, 1.0)  # white marker
    glVertex3f(self.pos[0], self.pos[1], self.pos[2])
    glEnd()
  def main_loop(self):
    glfw.make_context_current(self.win)
    while not glfw.window_should_close(self.win):
      self._handle_continuous_input()
      self.continuous_update()
      if self.reset_proj:
        self._set_projection()
        self.reset_proj = False
      if self.dirty:
        self._draw_all()
        glfw.swap_buffers(self.win)
        self.dirty = False
      glfw.wait_events_timeout(0.05)
    glfw.terminate()
  # Methods that should be overridden by the subclass:
  def draw(self):
    raise RuntimeError("Base Display class does not implement draw() method. Subclasses should override.")
  def start(self, *args):
    raise RuntimeError("Base Display class does not implement start() method. Subclasses should override.")
  def continuous_update(self):
    pass # can override in subclass

def launch_display_thread(display_cls, width:int, height:int, title:str, **kwargs):
  def launch():
    display = display_cls(width, height, title, **kwargs)
    display.main_loop()
  threading.Thread(target=launch, daemon=True).start()
    



# ------------------ Initialize GLFW ------------------
if not glfw.init():
  raise SystemExit("GLFW init failed")


