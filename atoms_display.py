import ctypes
from queue import Queue

import numpy as np
from OpenGL.GL import *

from display import Display, launch_display_thread
from utils import must_be


# ------------------ Element Parameters ------------------

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



# ------------------ Mesh Construction Code ------------------

def make_icosphere(subdivisions: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a unit icosphere mesh with smooth normals.

    Returns:
        pos : (N, 3) float32 array of vertex positions (on the unit sphere)
        nrm : (N, 3) float32 array of vertex normals (== positions, normalized)
        idx : (M,)   uint32 array of triangle indices (triplets)

    Notes:
        - Keep 'subdivisions' small (e.g., 0..4). Each step quadruples triangle count.
        - Positions are unit-length; use uniform instance scaling for radius.
    """
    # Golden ratio and base icosahedron (unit sphere after normalization)
    t = (1.0 + 5.0 ** 0.5) / 2.0

    base_vertices = np.array([
        [-1,  t,  0],
        [ 1,  t,  0],
        [-1, -t,  0],
        [ 1, -t,  0],

        [ 0, -1,  t],
        [ 0,  1,  t],
        [ 0, -1, -t],
        [ 0,  1, -t],

        [ t,  0, -1],
        [ t,  0,  1],
        [-t,  0, -1],
        [-t,  0,  1],
    ], dtype=np.float64)

    # Normalize to unit sphere
    base_vertices /= np.linalg.norm(base_vertices, axis=1, keepdims=True)

    base_faces = [
        (0, 11, 5),  (0, 5, 1),   (0, 1, 7),   (0, 7, 10),  (0, 10, 11),
        (1, 5, 9),   (5, 11, 4),  (11, 10, 2), (10, 7, 6),  (7, 1, 8),
        (3, 9, 4),   (3, 4, 2),   (3, 2, 6),   (3, 6, 8),   (3, 8, 9),
        (4, 9, 5),   (2, 4, 11),  (6, 2, 10),  (8, 6, 7),   (9, 8, 1),
    ]

    verts = base_vertices.tolist()
    faces = [tuple(face) for face in base_faces]

    # Cache for midpoints to avoid duplicating vertices
    midpoint_cache: dict[tuple[int, int], int] = {}

    def midpoint(i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        if key in midpoint_cache:
            return midpoint_cache[key]
        vi = np.array(verts[i], dtype=np.float64)
        vj = np.array(verts[j], dtype=np.float64)
        vm = vi + vj
        vm /= np.linalg.norm(vm)  # project back to unit sphere
        idx = len(verts)
        verts.append(vm.tolist())
        midpoint_cache[key] = idx
        return idx

    # Subdivide
    for _ in range(subdivisions):
        new_faces = []
        midpoint_cache.clear()
        for a, b, c in faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            # 4 new faces, preserving CCW winding on a sphere
            new_faces.extend([
                (a,  ab, ca),
                (b,  bc, ab),
                (c,  ca, bc),
                (ab, bc, ca),
            ])
        faces = new_faces

    pos = np.asarray(verts, dtype=np.float32)
    # Smooth normals = normalized positions
    nrm = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    idx = np.asarray([i for tri in faces for i in tri], dtype=np.uint32)

    return pos, nrm, idx


def create_mesh(sphere_verts, sphere_norms, sphere_indxs, atomic_numbers, positions):
  nv,          must_be[3] = sphere_verts.shape
  must_be[nv], must_be[3] = sphere_norms.shape
  nidx, = sphere_indxs.shape
  natom, = atomic_numbers.shape
  must_be[natom], must_be[3] = positions.shape
  ans_verts = sphere_verts[None, :, :]*ATOM_VALENCE_RADII[atomic_numbers][:, None, None] + positions[:, None, :]
  ans_norms = sphere_norms[None, :, :] + np.zeros((natom, nv, 3))
  ans_indxs = sphere_indxs[None, :] + (np.arange(natom)*nv)[:, None]
  ans_cols = (ATOM_COLORS[atomic_numbers])[:, None, :] + np.zeros((natom, nv, 3))
  return ans_verts.reshape(natom*nv, 3), ans_norms.reshape(natom*nv, 3), ans_indxs.reshape(natom*nidx), ans_cols.reshape(natom*nv, 3)

def update_mesh(sphere_verts, output_verts, atomic_numbers, positions):
  """ MUTATES output_verts """
  nv, must_be[3] = sphere_verts.shape
  natom, must_be[3] = positions.shape
  must_be[natom], = atomic_numbers.shape
  must_be[natom*nv], must_be[3] = output_verts.shape
  output_verts[:, :] = (
      sphere_verts[None, :, :]*ATOM_VALENCE_RADII[atomic_numbers][:, None, None] + positions[:, None, :]
    ).reshape(natom*nv, 3)



# ------------------ Define Display subclass AtomsDisplay ------------------

class AtomsDisplay(Display):
  def start(self, atomic_nums, positions, command_queue):
    self.command_queue = command_queue
    self.atomic_nums, self.positions = atomic_nums, positions
    self.sphere_verts, self.sphere_norms, self.sphere_indxs = make_icosphere(1)
    self.vertices, self.normals, self.faces, self.colors = create_mesh(
      self.sphere_verts, self.sphere_norms, self.sphere_indxs,
      self.atomic_nums, self.positions)
    # ensure dtype/contiguity
    self.vertices = np.ascontiguousarray(self.vertices.astype(np.float32))
    self.normals  = np.ascontiguousarray(self.normals.astype(np.float32))
    self.faces    = np.ascontiguousarray(self.faces.astype(np.uint32))
    self.colors   = np.ascontiguousarray(self.colors.astype(np.float32))
    # setup buffers
    self._init_buffers()
  def _bind(self, buff, target=GL_ARRAY_BUFFER):
    assert buff.flags.c_contiguous, "array to be bound must be contiguous!"
    handle = glGenBuffers(1)
    assert handle > 0, "got a handle of 0, indicates context not set correctly, or other error"
    glBindBuffer(target, handle)
    glBufferData(target, buff.nbytes, buff, self.usage)
    glBindBuffer(target, 0) # clean up by making sure no buffer is bound
    return handle
  def _update(self, handle, new_data, target=GL_ARRAY_BUFFER):
    glBindBuffer(target, handle)
    glBufferData(target, new_data.nbytes, None, self.usage) # orphan old storage
    glBufferSubData(target, 0, new_data.nbytes, new_data)
    glBindBuffer(target, 0) # clean up by making sure no buffer is bound
  def _init_buffers(self):
    self.usage = GL_DYNAMIC_DRAW
    self.vbo = self._bind(self.vertices)
    self.nbo = self._bind(self.normals)
    self.cbo = self._bind(self.colors)
    self.ebo = self._bind(self.faces, target=GL_ELEMENT_ARRAY_BUFFER)
  def set_positions(self, new_positions):
    self.positions = new_positions
    update_mesh(self.sphere_verts, self.vertices, self.atomic_nums, self.positions)
    self._update(self.vbo, self.vertices)
  def continuous_update(self):
    while not self.command_queue.empty():
      command, *args = self.command_queue.get()
      if command == "update_pos":
        new_positions, = args
        self.set_positions(new_positions)
        self.dirty = True
      else:
        print(f"Warning: Ignoring unrecognized command {command}.")
  def draw(self):    
    # --- minimal lighting ---
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

    # white-ish headlight, a bit of ambient fill
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.6, 0.6, 0.6, 1.0))

    # put the light in view space; do this AFTER setting your camera/modelview
    glLightfv(GL_LIGHT0, GL_POSITION, (0.5, 1.0, 0.5, 0.0))  # directional from camera

    # set material
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glShadeModel(GL_SMOOTH)
    
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    
    glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
    glVertexPointer(3, GL_FLOAT, 0, ctypes.c_void_p(0))
    
    glBindBuffer(GL_ARRAY_BUFFER, self.nbo)
    glNormalPointer(GL_FLOAT, 0, ctypes.c_void_p(0))
    
    glBindBuffer(GL_ARRAY_BUFFER, self.cbo)
    glColorPointer(3, GL_FLOAT, 0, ctypes.c_void_p(0))
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
    glDrawElements(GL_TRIANGLES, self.faces.size, GL_UNSIGNED_INT, ctypes.c_void_p(0))
    
    glBindBuffer(GL_ARRAY_BUFFER, 0) # clean up by making sure no buffer is bound
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0) # clean up by making sure no buffer is bound
    glDisableClientState(GL_NORMAL_ARRAY) # cleanup
    glDisableClientState(GL_VERTEX_ARRAY) # cleanup
    glDisableClientState(GL_COLOR_ARRAY)  # cleanup



# ------------------ Interface for Interactions ------------------

class AtomsDisplayInterface:
  def __init__(self, atomic_nums, positions):
    self.commands = Queue()
    launch_display_thread(AtomsDisplay, 800, 600, "atoms display",
      start_args=[atomic_nums, positions, self.commands])
  def update_pos(self, new_pos):
    self.commands.put(("update_pos", new_pos.copy())) # sending data to another thread, so making a copy is the polite thing to do

def launch_atom_display(atomic_numbers, positions):
  disp = AtomsDisplayInterface(atomic_numbers, positions)
  return disp



# ------------------ Allow XYZ parsing to test the code ------------------

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



# ------------------ Main ------------------
def main():
  import time
  from sys import argv
  atomic_nums, positions = read_xyz(argv[1])
  atomic_nums[:] = 1
  positions *= 2.
  
  display = launch_atom_display(atomic_nums, positions)
  
  for i in range(0, 1000):
    time.sleep(0.03)
    positions = positions + 0.04*np.random.randn(*positions.shape)
    display.update_pos(positions)
    print(i)

if __name__ == "__main__":
  main()

