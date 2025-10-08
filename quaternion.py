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
    return np.array([r.x, r.y, r.z])
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
    


