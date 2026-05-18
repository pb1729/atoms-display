from dataclasses import dataclass

import numpy as np

from quaternion import Quat
from constants import RIBBON_COLOR, SIDEVEC_RAD


ATOMS_PER_RESIDUE = 4
BACKBONE_ATOMS_PER_RESIDUE = 3
ATOM_N = 0
ATOM_CA = 1
ATOM_C = 2
ATOM_O = 3
FRAME_TARGET_SMOOTHING_PASSES = 1


@dataclass
class Mesh:
  vertices: np.ndarray
  normals: np.ndarray
  colors: np.ndarray
  faces: np.ndarray

  def __iter__(self):
    return iter((self.vertices, self.normals, self.colors, self.faces))

  def __getitem__(self, idx):
    return (self.vertices, self.normals, self.colors, self.faces)[idx]


def catmull_rom(points, res:int=10):
  """ points: (N, 3)
      ans, d_ans: ((N-1)*res, 3)
      create a catmull-rom spline with a chain of points as guides
      you can actually replace `3` with any desired dimension
      returns both spline points and tangents """
  # t0, t1, t2, t3 = -1, 0, 1, 2
  t = np.arange(res)[:, None]/float(res) # (res, 1)
  points = np.concatenate([points[:1], points, points[-1:]], axis=0) # pad with duplicate endpoints (N+2, 3)
  points = points[:, None, :] # (N+2, 1, 3)
  # shape for everything below: (N-1, res, 3)
  A1 = -t*points[:-3] + (t + 1.)*points[1:-2]
  A2 = (1. - t)*points[1:-2] + t*points[2:-1]
  A3 = (2. - t)*points[2:-1] + (t - 1.)*points[3:]
  B1 = 0.5*((1. - t)*A1 + (t + 1.)*A2)
  B2 = 0.5*((2. - t)*A2 + t*A3)
  C1 = (1. - t)*B1 + t*B2
  dA1 = -points[:-3] + points[1:-2]
  dA2 = -points[1:-2] + points[2:-1]
  dA3 = -points[2:-1] + points[3:]
  dB1 = 0.5*((1. - t)*dA1 + (t + 1.)*dA2 + A2 - A1)
  dB2 = 0.5*((2. - t)*dA2 + t*dA3 + A3 - A2)
  dC1 = (1. - t)*dB1 + t*dB2 + B2 - B1
  C1 = C1.reshape(-1, C1.shape[-1]) # ((N-1)*res, 3)
  dC1 = dC1.reshape(-1, dC1.shape[-1]) # ((N-1)*res, 3)
  return C1, dC1


def normalize(v):
  """ v: (..., 3) """
  return v/np.linalg.norm(v, axis=-1, keepdims=True)

def reject_from(v, axis):
  """Remove the component of v parallel to axis."""
  axis = normalize(axis)
  return v - axis*np.sum(v*axis, axis=-1, keepdims=True)

def slerp(q:Quat, t:float):
  """ Spherical linear quarternion interpolation between 1 and q (or -q). """
  if q.w < 0: q = -q # choose the shorter path
  theta = np.arccos(np.clip(q.w, -1.0, 1.0))
  sin_theta = np.sin(theta)
  if abs(sin_theta) < 1e-8:
    return Quat()
  q_1 = Quat()
  return q_1.scale(np.sin((1. - t)*theta)/sin_theta) + q.scale(np.sin(t*theta)/sin_theta)

def align_vector_signs(vectors):
  """Flip vectors as needed so adjacent directions stay on the same side."""
  aligned = vectors.copy()
  for i in range(1, aligned.shape[0]):
    if np.dot(aligned[i - 1], aligned[i]) < 0:
      aligned[i] *= -1
  return aligned

def smooth_directions(directions, passes:int=1):
  """Lightweight sign-aware smoothing for sparse frame target directions."""
  directions = normalize(align_vector_signs(directions))
  for _ in range(passes):
    padded = np.concatenate([directions[:1], directions, directions[-1:]], axis=0)
    directions = normalize(0.25*padded[:-2] + 0.5*padded[1:-1] + 0.25*padded[2:])
    directions = align_vector_signs(directions)
  return directions

def get_tangent_frames(target_indices, width_dir_targets, tangents):
  """ target_indices: (M) --> ints, should be sorted in increasing order!
      width_dir_targets: (M, 3)
      tangents: (N, 3)
      width_dirs, face_normals: (N, 3) """
  N = tangents.shape[0]
  M, = target_indices.shape
  assert target_indices[0] == 0 and target_indices[-1] == N - 1, "must have targets at the endpoints of the range"
  tangents = normalize(tangents)
  width_dir_targets = normalize(width_dir_targets)
  # initialize answer memory
  width_dirs = np.empty((N, 3))
  face_normals = np.empty((N, 3))
  # prepare for start of loop
  width_dirs[0] = width_dir_targets[target_indices[0]]
  face_normals[0] = np.cross(width_dirs[0], tangents[0])
  # loop through remaining M indices
  for i in range(M - 1):
    idx_start, idx_end = target_indices[i], target_indices[i + 1]
    # choose the closer vector to rotate to
    if np.dot(width_dirs[idx_start], width_dir_targets[i + 1]) > 0:
      next_width_dir = width_dir_targets[i + 1]
    else:
      next_width_dir = -width_dir_targets[i + 1]
    curr_tangent = tangents[idx_start]
    next_tangent = tangents[idx_end]
    q_rel = Quat.from_frames(curr_tangent, width_dirs[idx_start], next_tangent, next_width_dir)
    j = 1 + np.arange(idx_end - idx_start)
    t = j / (idx_end - idx_start)
    q = slerp(q_rel, t)
    width_dirs[idx_start + j] = q.rotate_vec3(width_dirs[idx_start])
    face_normals[idx_start + j] = np.cross(width_dirs[idx_start + j], tangents[idx_start + j])
  return width_dirs, face_normals

def frames_to_ribbon(width_dirs, face_normals):
  """ width_dirs, face_normals: (N, 3)
      pos, norm: (N, 4, 3)

      The four vertices per frame are front-left, front-right, back-left,
      back-right. Back vertices duplicate the positions with flipped normals
      so lighting works when the flat ribbon is viewed from either side.
  """
  offsets = np.stack([-SIDEVEC_RAD*width_dirs, SIDEVEC_RAD*width_dirs], axis=1)
  vertices = np.concatenate([offsets, offsets], axis=1)
  normals = np.concatenate([
    np.repeat(face_normals[:, None, :], 2, axis=1),
    np.repeat(-face_normals[:, None, :], 2, axis=1),
  ], axis=1)
  return vertices, normalize(normals)

def ribbon_faces(N:int):
  """Return triangle indices for an alternating flat ribbon strip."""
  frame_starts = 4*np.arange(N - 1)[:, None]
  even_segments = (np.arange(N - 1)[:, None] % 2) == 0

  front_even = np.array([0, 1, 4, 1, 5, 4])
  front_odd = np.array([0, 1, 5, 0, 5, 4])
  front = np.where(even_segments, front_even, front_odd) + frame_starts

  back_even = np.array([2, 6, 3, 3, 6, 7])
  back_odd = np.array([2, 7, 3, 2, 6, 7])
  back = np.where(even_segments, back_even, back_odd) + frame_starts

  return np.concatenate([front, back], axis=1).reshape(-1)

def interleave(a, b):
  """ a, b: (N, ...)
      ans: (2*N, ...) """
  return np.stack([a, b], axis=1).reshape(2*a.shape[0], *a.shape[1:])


def residue_layout(ribbon_positions):
  """Return residue atom positions shaped as (residues, N/CA/C/O, xyz)."""
  return ribbon_positions.reshape(-1, ATOMS_PER_RESIDUE, 3)

def backbone_from_residues(residues):
  """Return flattened N, CA, C backbone positions."""
  return residues[:, :BACKBONE_ATOMS_PER_RESIDUE, :].reshape(-1, 3)

def peptide_plane_normals(residues):
  """Return peptide-plane normal estimates near C and N anchors."""
  c_to_next_n = residues[:-1, ATOM_C] - residues[1:, ATOM_N]
  n_to_ca = residues[1:, ATOM_CA] - residues[1:, ATOM_N]
  c_to_o = residues[1:, ATOM_O] - residues[1:, ATOM_C]
  c_to_ca = residues[1:, ATOM_CA] - residues[1:, ATOM_C]
  normal_at_n = np.cross(c_to_next_n, n_to_ca)
  normal_at_c = np.cross(c_to_ca, c_to_o)
  return normal_at_c, normal_at_n

def backbone_spline_point_index(residue_indices, atom_offset, res):
  """Index into the sampled spline at a backbone atom anchor."""
  return res*(BACKBONE_ATOMS_PER_RESIDUE*residue_indices + atom_offset)

def frame_targets(residues, tangents, res):
  """Return sparse spline indices and ribbon-width direction targets."""
  normal_at_c, normal_at_n = peptide_plane_normals(residues)
  # Each peptide plane spans residue i to i+1, but the historical ribbon
  # anchors it at CA/C samples of residue i on the flattened N,CA,C backbone.
  peptide_indices = np.arange(normal_at_c.shape[0])
  target_indices_c = backbone_spline_point_index(peptide_indices, ATOM_CA, res)
  target_indices_n = backbone_spline_point_index(peptide_indices, ATOM_C, res)
  width_targets_c = np.cross(normal_at_c, tangents[target_indices_c])
  width_targets_n = np.cross(normal_at_n, tangents[target_indices_n])
  target_indices = interleave(target_indices_c, target_indices_n)
  width_targets = smooth_directions(
    interleave(width_targets_c, width_targets_n),
    passes=FRAME_TARGET_SMOOTHING_PASSES)
  width_targets = normalize(reject_from(width_targets, tangents[target_indices]))
  return target_indices, width_targets

def trim_to_targets(centers, tangents, target_indices):
  """Trim spline arrays so frame interpolation starts and ends at targets."""
  idx_start, idx_end = target_indices.min(), target_indices.max() + 1
  return centers[idx_start:idx_end], tangents[idx_start:idx_end], target_indices - idx_start

def build_ribbon_mesh(centers, normals, faces):
  vertices = centers.reshape(-1, 3)
  normals = normals.reshape(-1, 3)
  colors = np.ones_like(vertices)*RIBBON_COLOR
  return Mesh(
    np.ascontiguousarray(vertices.astype(np.float32)),
    np.ascontiguousarray(normals.astype(np.float32)),
    np.ascontiguousarray(colors.astype(np.float32)),
    np.ascontiguousarray(faces.astype(np.uint32)))

def ribbon_mesh(ribbon_positions, res:int=8, res_loop:int=12):
  """ ribbon_positions: (4*residues, 3)
      residue is unit of 4 atoms: N, CA, C, O """
  del res_loop # kept for API compatibility with the old tube mesh
  if ribbon_positions.shape[0] < 2*ATOMS_PER_RESIDUE:
    raise ValueError("ribbon mesh requires at least two residues")
  residues = residue_layout(ribbon_positions)
  centers, tangents = catmull_rom(backbone_from_residues(residues), res=res)
  target_indices, width_targets = frame_targets(residues, tangents, res)
  centers, tangents, target_indices = trim_to_targets(centers, tangents, target_indices)
  width_dirs, face_normals = get_tangent_frames(target_indices, width_targets, tangents)
  vertices, normals = frames_to_ribbon(width_dirs, face_normals)
  vertices += centers[:, None, :]
  return build_ribbon_mesh(vertices, normals, ribbon_faces(vertices.shape[0]))



if __name__ == "__main__":
  # test catmull rom
  import matplotlib.pyplot as plt
  points = np.stack([np.arange(-2, 2), np.abs(np.arange(-2, 2)*np.sqrt(np.abs(np.arange(-2, 2))))], axis=-1)
  spline, tangents = catmull_rom(points)
  tangents_compare = (spline[1:] - spline[:-1])/0.1
  plt.scatter(points[:, 0], points[:, 1])
  plt.plot(spline[:, 0], spline[:, 1], marker=".", alpha=0.5)
  plt.plot(tangents[:, 0], tangents[:, 1], marker=".", alpha=0.5)
  plt.plot(tangents_compare[:, 0], tangents_compare[:, 1], marker=".", alpha=0.5)
  plt.show()
