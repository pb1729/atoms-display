import math
import numpy as np
from OpenGL.GL import GLfloat


class Quat:
  __slots__ = ("w", "x", "y", "z")
  def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
    self.w, self.x, self.y, self.z = w, x, y, z
  @staticmethod
  def from_axis_angle(a, angle):
    ax, ay, az = a
    th = angle * 0.5
    s = math.sin(th)
    # normalize axis (assume non-zero)
    n = math.sqrt(ax*ax + ay*ay + az*az)
    ax, ay, az = ax/n, ay/n, az/n
    return Quat(math.cos(th), ax*s, ay*s, az*s)
  @staticmethod
  def from_frames(u, v, up, vp):
    u, v = u/np.linalg.norm(u), v/np.linalg.norm(v)
    up, vp = up/np.linalg.norm(up), vp/np.linalg.norm(vp)
    w = np.cross(u, v)
    wp = np.cross(up, vp)
    w, wp = w/np.linalg.norm(w), wp/np.linalg.norm(wp)
    s = 1.0 + np.dot(u, up) + np.dot(v, vp) + np.dot(w, wp)
    r = (np.cross(u, up) + np.cross(v, vp) + np.cross(w, wp))
    norm = np.sqrt(s*s + np.dot(r, r))
    s /= norm
    r /= norm
    return Quat(s, r[0], r[1], r[2])
  def __add__(self, o):  # quaternion addition
    return Quat(self.w + o.w, self.x + o.x, self.y + o.y, self.z + o.z)
  def __neg__(self):
    return self.scale(-1)
  def scale(self, a):
    return Quat(a*self.w, a*self.x, a*self.y, a*self.z)
  def __mul__(self, o):  # quaternion multiply
    w, x, y, z = self.w, self.x, self.y, self.z
    W = w*o.w - x*o.x - y*o.y - z*o.z
    X = w*o.x + x*o.w + y*o.z - z*o.y
    Y = w*o.y - x*o.z + y*o.w + z*o.x
    Z = w*o.z + x*o.y - y*o.x + z*o.w
    return Quat(W, X, Y, Z)
  def conj(self):
    return Quat(self.w, -self.x, -self.y, -self.z)
  def rotate_vec3(self, v):
    # rotate vector v by this quaternion
    qv = Quat(0.0, v[0], v[1], v[2])
    qi = self.conj() # inverse for unit q
    r = self * qv * qi
    return np.stack([r.x, r.y, r.z], axis=-1)
  def to_mat4(self):
    # Column-major 4x4 for OpenGL
    w,x,y,z = self.w, self.x, self.y, self.z
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return (GLfloat * 16)(
      1-2*(yy+zz), 2*(xy+wz),   2*(xz-wy),   0,
      2*(xy-wz),   1-2*(xx+zz), 2*(yz+wx),   0,
      2*(xz+wy),   2*(yz-wx),   1-2*(xx+yy), 0,
      0,           0,           0,           1
    )





if __name__ == "__main__":
  def nvecs():
    u = np.random.randn(3)
    v = np.cross(u, np.random.randn(3))
    return u/np.linalg.norm(u), v/np.linalg.norm(v)
  (u, v), (up, vp) = nvecs(), nvecs()
  q = Quat.from_frames(u, v, up, vp)
  print(q.rotate_vec3(u), up)
  print(q.rotate_vec3(v), vp)




