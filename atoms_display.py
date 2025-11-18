import ctypes
from queue import Queue

import numpy as np
from OpenGL.GL import *

from display import Display, launch_display_thread
from utils import must_be
from ribbon_mesh import ribbon_mesh
from constants import ATOM_COLORS, ATOM_VALENCE_RADII



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


def create_mesh(sphere_verts, sphere_norms, sphere_indxs, atomic_numbers, positions, atom_scales):
  nv,          must_be[3] = sphere_verts.shape
  must_be[nv], must_be[3] = sphere_norms.shape
  nidx, = sphere_indxs.shape
  natom, = atomic_numbers.shape
  must_be[natom], must_be[3] = positions.shape
  ans_verts = sphere_verts[None, :, :]*ATOM_VALENCE_RADII[atomic_numbers][:, None, None]*atom_scales[:, None, None] + positions[:, None, :]
  ans_norms = sphere_norms[None, :, :] + np.zeros((natom, nv, 3))
  ans_indxs = sphere_indxs[None, :] + (np.arange(natom)*nv)[:, None]
  ans_cols = (ATOM_COLORS[atomic_numbers])[:, None, :] + np.zeros((natom, nv, 3))
  return ans_verts.reshape(natom*nv, 3), ans_norms.reshape(natom*nv, 3), ans_indxs.reshape(natom*nidx), ans_cols.reshape(natom*nv, 3)

def update_mesh(sphere_verts, output_verts, atomic_numbers, positions, atom_scales):
  """ MUTATES output_verts """
  nv, must_be[3] = sphere_verts.shape
  natom, must_be[3] = positions.shape
  must_be[natom], = atomic_numbers.shape
  must_be[natom*nv], must_be[3] = output_verts.shape
  output_verts[:, :] = (
      sphere_verts[None, :, :]*ATOM_VALENCE_RADII[atomic_numbers][:, None, None]*atom_scales[:, None, None] + positions[:, None, :]
    ).reshape(natom*nv, 3)



# ------------------ Define Display subclass AtomsDisplay ------------------

class AtomsDisplay(Display):
  def start(self, atomic_nums, positions, command_queue):
    self.command_queue = command_queue
    self.atomic_nums, self.positions = atomic_nums, positions
    self.atom_scales = np.ones(self.positions.shape[0]) # can hide atoms by setting their scale to 0
    self.sphere_verts, self.sphere_norms, self.sphere_indxs = make_icosphere(1)
    self.vertices, self.normals, self.faces, self.colors = create_mesh(
      self.sphere_verts, self.sphere_norms, self.sphere_indxs,
      self.atomic_nums, self.positions, self.atom_scales)
    # ensure dtype/contiguity
    self.vertices = np.ascontiguousarray(self.vertices.astype(np.float32))
    self.normals  = np.ascontiguousarray(self.normals.astype(np.float32))
    self.faces    = np.ascontiguousarray(self.faces.astype(np.uint32))
    self.colors   = np.ascontiguousarray(self.colors.astype(np.float32))
    # setup list of ribbon meshes tuples of (vertices, normals, colors, faces)
    self.ribbons = []
    self.ribbon_meshes = []
    self.ribbon_buffers = []
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
    glBufferData(target, new_data.nbytes, None, GL_DYNAMIC_DRAW) # orphan old storage
    glBufferSubData(target, 0, new_data.nbytes, new_data)
    glBindBuffer(target, 0) # clean up by making sure no buffer is bound
  def _init_buffers(self):
    self.usage = GL_DYNAMIC_DRAW
    self.vbo = self._bind(self.vertices)
    self.nbo = self._bind(self.normals)
    self.cbo = self._bind(self.colors)
    self.ebo = self._bind(self.faces, target=GL_ELEMENT_ARRAY_BUFFER)
  def update_atom_vertices(self):
    update_mesh(self.sphere_verts, self.vertices, self.atomic_nums, self.positions, self.atom_scales)
    self._update(self.vbo, self.vertices)
  def set_positions(self, new_positions):
    self.positions = new_positions
    self.update_atom_vertices()
    self.update_ribbons()
  def add_ribbon(self, ribbon):
    self.ribbons.append(ribbon)
    mesh = ribbon_mesh(self.positions[ribbon])
    self.ribbon_meshes.append(mesh)
    vertices, normals, colors, faces = mesh
    buffers = self._bind(vertices), self._bind(normals), self._bind(colors), self._bind(faces, target=GL_ELEMENT_ARRAY_BUFFER)
    self.ribbon_buffers.append(buffers)
    # hide the atoms that were part of the ribbon
    self.atom_scales[ribbon[1:-4]] = 0.0 # drawn ribbon doesn't include all atoms that were provided to compute it
    self.update_atom_vertices()
  def update_ribbons(self):
    for i, ribbon in enumerate(self.ribbons):
      self.ribbon_meshes[i] = ribbon_mesh(self.positions[ribbon]) # recompute the whole ribbon mesh, sadly
    for (vbo, nbo, cbo, ebo), (vertices, normals, colors, faces) in zip(self.ribbon_buffers, self.ribbon_meshes):
      self._update(vbo, vertices)
      self._update(nbo, normals)
      self._update(cbo, colors)
      self._update(ebo, faces, target=GL_ELEMENT_ARRAY_BUFFER)
  def continuous_update(self):
    positions_changed = False
    while not self.command_queue.empty():
      command, *args = self.command_queue.get()
      if command == "update_pos":
        new_positions, = args
        positions_changed = True
        self.dirty = True
      elif command == "add_ribbon":
        ribbon, = args
        self.add_ribbon(ribbon)
        self.dirty = True
      elif command == "get_current_screen":
        ans_queue, = args
        width, height = self._get_wh()
        glReadBuffer(GL_BACK)
        pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        ans_queue.put((width, height, pixels))
        self.dirty = True # request a redraw
      else:
        print(f"Warning: Ignoring unrecognized command {command}.")
    if positions_changed: # group together all position change updates into one
      self.set_positions(new_positions)
  def _draw_mesh(self, vbo, nbo, cbo, ebo, nfaces:int):
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glVertexPointer(3, GL_FLOAT, 0, ctypes.c_void_p(0))
    
    glBindBuffer(GL_ARRAY_BUFFER, nbo)
    glNormalPointer(GL_FLOAT, 0, ctypes.c_void_p(0))
    
    glBindBuffer(GL_ARRAY_BUFFER, cbo)
    glColorPointer(3, GL_FLOAT, 0, ctypes.c_void_p(0))
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glDrawElements(GL_TRIANGLES, nfaces, GL_UNSIGNED_INT, ctypes.c_void_p(0))
    
    glBindBuffer(GL_ARRAY_BUFFER, 0) # clean up by making sure no buffer is bound
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0) # clean up by making sure no buffer is bound
  def draw(self):    
    # --- minimal lighting ---
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

    # white-ish headlight, a bit of ambient fill
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1.0))

    # put the light in view space; do this AFTER setting your camera/modelview
    glLightfv(GL_LIGHT0, GL_POSITION, (0.5, 1.0, 0.5, 0.0))  # directional from camera

    # set material
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glShadeModel(GL_SMOOTH)
    
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    
    # draw atoms:
    self._draw_mesh(self.vbo, self.nbo, self.cbo, self.ebo, self.faces.size)
    # draw ribbons:
    for mesh, buffers in zip(self.ribbon_meshes, self.ribbon_buffers):
      self._draw_mesh(*buffers, mesh[3].size)
    
    glDisableClientState(GL_VERTEX_ARRAY) # cleanup
    glDisableClientState(GL_NORMAL_ARRAY) # cleanup
    glDisableClientState(GL_COLOR_ARRAY)  # cleanup



# ------------------ Interface for Interactions ------------------

class AtomsDisplayInterface:
  def __init__(self, atomic_nums, positions, ribbons=None):
    """ atomic_nums: (num_atoms) int
        positions: (num_atoms, 3) float
        ribbons: (4*residues_in_ribbon) int -- indexes into the num_atoms dim,
        for each residue, put the indices for the following atom names in order: N, CA, C, O """
    self.commands = Queue()
    launch_display_thread(AtomsDisplay, 800, 600, "atoms display",
      start_args=[atomic_nums, positions, self.commands])
    if ribbons is None: ribbons = []
    for ribbon in ribbons:
      self.commands.put(("add_ribbon", ribbon))
  def update_pos(self, new_pos):
    self.commands.put(("update_pos", new_pos.copy())) # sending data to another thread, so making a copy is the polite thing to do
  def get_current_screen(self):
    ans_queue = Queue()
    self.commands.put(("get_current_screen", ans_queue))
    width, height, pixels = ans_queue.get()
    return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)

def launch_atom_display(atomic_numbers, positions, ribbons=None):
  disp = AtomsDisplayInterface(atomic_numbers, positions, ribbons)
  return disp



# ------------------ Main ------------------
def main():
  import time
  from sys import argv
  
  from read_xyz import read_xyz
  from read_pdb import read_pdb
  
  fnm = argv[1]
  if fnm[-4:] == ".xyz":
    atomic_nums, positions = read_xyz(argv[1])
    ribbons = None
  elif fnm[-4:] == ".pdb":
    with open(fnm, "r") as f:
      atomic_nums, positions, ribbons = read_pdb(f)
  else:
    raise RuntimeError("unrecognized file type")
  
  positions -= positions.mean(0) # center the initial positioning of everything
  display = launch_atom_display(atomic_nums, positions, ribbons)
  
  while True:
    time.sleep(0.5)
    #curr_screen = display.get_current_screen()
    #positions = positions + 0.05*np.random.randn(*positions.shape)
    #display.update_pos(positions) # test position updating
    #print(i)

if __name__ == "__main__":
  main()

