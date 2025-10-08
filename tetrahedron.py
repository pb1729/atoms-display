from OpenGL.GL import *

from display import Display, launch_display_thread


# ------------------ Example Display: Tetrahedron ------------------

class TetrahedronDisplay(Display):
  def start(self):
    self.vertices = [
      ( 1.0,  1.0,  1.0),
      (-1.0, -1.0,  1.0),
      (-1.0,  1.0, -1.0),
      ( 1.0, -1.0, -1.0),
    ]
    self.faces = [(0,1,2),(0,3,1),(0,2,3),(1,3,2)] 
    self.colors = [
      (0.90, 0.30, 0.30),
      (0.30, 0.90, 0.30),
      (0.30, 0.50, 0.95),
      (0.95, 0.80, 0.30),
    ]
  def draw(self):
    # draw world geometry
    glBegin(GL_TRIANGLES)
    for fi, (i, j, k) in enumerate(self.faces):
      glColor3f(*self.colors[fi % len(self.colors)])
      glVertex3f(*self.vertices[i]); glVertex3f(*self.vertices[j]); glVertex3f(*self.vertices[k])
    glEnd()


# ------------------ Main ------------------
def main():
  import time
  launch_display_thread(TetrahedronDisplay, 800, 600, "tetrahedron display")
  for i in range(0, 100):
    time.sleep(1)
    print(i)

if __name__ == "__main__":
  main()

