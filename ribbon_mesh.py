import numpy as np

from quaternion import Quat
from constants import RIBBON_COLOR, SIDEVEC_RAD, NORMVEC_RAD


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

def slerp(q:Quat, t:float):
  """ Spherical linear quarternion interpolation between 1 and q (or -q). """
  if q.w < 0: q = -q # choose the shorter path
  theta = np.arccos(q.w)
  q_1 = Quat()
  return q_1.scale(np.sin((1. - t)*theta)/np.sin(theta)) + q.scale(np.sin(t*theta)/np.sin(theta))

def get_tangent_frames(target_indices, sidevec_targets, tangents):
  """ target_indices: (M) --> ints, should be sorted in increasing order!
      sidevec_targets: (M, 3)
      tangents: (N, 3)
      sidevecs, normvecs: (N, 3) """
  N = tangents.shape[0]
  M, = target_indices.shape
  assert target_indices[0] == 0 and target_indices[-1] == N - 1, "must have targets at the endpoints of the range"
  tangents = normalize(tangents)
  sidevec_targets = normalize(sidevec_targets)
  # initialize answer memory
  sidevecs = np.empty((N, 3))
  normvecs = np.empty((N, 3))
  # prepare for start of loop
  sidevecs[0] = sidevec_targets[target_indices[0]]
  normvecs[0] = np.cross(sidevecs[0], tangents[0])
  # loop through remaining M indices
  for i in range(M - 1):
    idx_start, idx_end = target_indices[i], target_indices[i + 1]
    # choose the closer vector to rotate to
    if np.dot(sidevecs[idx_start], sidevec_targets[i + 1]) > 0:
      next_sidevec = sidevec_targets[i + 1]
    else:
      next_sidevec = -sidevec_targets[i + 1]
    curr_tangent = tangents[idx_start]
    next_tangent = tangents[idx_end]
    q_rel = Quat.from_frames(curr_tangent, sidevecs[idx_start], next_tangent, next_sidevec)
    j = 1 + np.arange(idx_end - idx_start)
    t = j / (idx_end - idx_start)
    q = slerp(q_rel, t)
    sidevecs[idx_start + j] = q.rotate_vec3(sidevecs[idx_start])
    normvecs[idx_start + j] = np.cross(sidevecs[idx_start + j], tangents[idx_start + j])
  return sidevecs, normvecs

def frames_to_loop(sidevecs, normvecs, res:int=20):
  """ sidevecs, normvecs: (N, 3)
      pos, norm: (N, res, 3) """
  theta = np.arange(res) * (2.*np.pi/res)
  z = np.stack([np.cos(theta), np.sin(theta)], axis=-1) # (res, 2)
  U = np.stack([SIDEVEC_RAD*sidevecs, NORMVEC_RAD*normvecs], axis=-1) # (N, 3, 2)
  U_pinv = U @ np.linalg.inv(U.mT @ U)
  pos = (U[:, None, :, :]*z[:, None, :]).sum(-1)
  norm = (U_pinv[:, None, :, :]*z[:, None, :]).sum(-1)
  return pos, normalize(norm)

def tube_faces(N:int, res_loop:int):
  # coding trick here is to describe mesh indices as polynomials in r=res_loop
  # 0+0r---0+1r
  #  |   /  |
  # 1+0r---1+1r
  base_mesh_1 = np.array([0, 1, 0, 0, 1, 1])
  base_mesh_r = np.array([0, 0, 1, 1, 0, 1])
  ring_mesh_1 = (base_mesh_1[None, :] + np.arange(res_loop)[:, None]) % res_loop
  ring_mesh_r = base_mesh_r[None, :]
  tube_mesh_1 = ring_mesh_1[None, :, :]
  tube_mesh_r = ring_mesh_r[None, :, :] + np.arange(N - 1)[:, None, None]
  ans = res_loop*tube_mesh_r + tube_mesh_1
  return ans.reshape(-1)

def interleave(a, b):
  """ a, b: (N, ...)
      ans: (2*N, ...) """
  return np.stack([a, b], axis=1).reshape(2*a.shape[0], *a.shape[1:])


def ribbon_mesh(ribbon_positions, res:int=8, res_loop:int=12):
  """ ribbon_positions: (4*residues, 3)
      residue is unit of 4 atoms: N, CA, C, O """
  # Layout is the following:
  # N CA C N CA C N CA C N CA C N CA C
  #      ---    ---    ---    ---
  # where `---` shows planes with normal vec defined
  residue_layout = ribbon_positions.reshape(-1, 4, 3)
  backbone_positions = residue_layout[:, :-1, :].reshape(-1, 3) # strip oxygens from backbone
  # compute some displacement vectors
  vecs_N_to_C = residue_layout[:-1, 2] - residue_layout[1:, 0]
  vecs_N_to_CA = residue_layout[1:, 1] - residue_layout[1:, 0]
  vecs_C_to_O  = residue_layout[1:, 3] - residue_layout[1:, 2]
  vecs_C_to_CA = residue_layout[1:, 1] - residue_layout[1:, 2]
  # compute backbone spline
  spline, tangents = catmull_rom(backbone_positions, res=res)
  # compute normal vectors
  normvec_N = np.cross(vecs_N_to_C, vecs_N_to_CA)
  normvec_C = np.cross(vecs_C_to_CA, vecs_C_to_O)
  # compute target indices and target side vectors
  target_indices_N = res*(2 + np.arange(normvec_N.shape[0])*3) # TODO: these offsets seem to work, but I don't really know why
  target_indices_C = res*(1 + np.arange(normvec_C.shape[0])*3)
  sidevec_targets_N = np.cross(normvec_N, tangents[target_indices_N])
  sidevec_targets_C = np.cross(normvec_C, tangents[target_indices_C])
  target_indices = interleave(target_indices_C, target_indices_N)
  sidevec_targets = interleave(sidevec_targets_C, sidevec_targets_N)
  # trim down spline and tangents to match target indices
  idx_start, idx_end = target_indices.min(), target_indices.max() + 1
  target_indices -= idx_start
  spline, tangents = spline[idx_start:idx_end], tangents[idx_start:idx_end]
  # get frames and loops
  sidevecs, normvecs = get_tangent_frames(target_indices, sidevec_targets, tangents)
  vertices, normals = frames_to_loop(sidevecs, normvecs, res=res_loop)
  vertices += spline[:, None, :]
  # okay, time to make the outputs!
  faces = tube_faces(vertices.shape[0], vertices.shape[1])
  vertices = vertices.reshape(-1, 3)
  normals  = normals.reshape(-1, 3)
  colors = np.ones_like(vertices)*RIBBON_COLOR
  return (
    np.ascontiguousarray(vertices.astype(np.float32)),
    np.ascontiguousarray(normals.astype(np.float32)),
    np.ascontiguousarray(colors.astype(np.float32)),
    np.ascontiguousarray(faces.astype(np.int32)))



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


