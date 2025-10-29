import numpy as np



try:
  import openmm
  import openmm.app
  import openmm.unit
except ImportError:
  print("Warning, openmm package not installed, pdb reading not supported.")
  def read_pdb(f):
    raise RuntimeError("Package openmm required to read a .pdb file!")
else:
  ANGSTROM = openmm.unit.Quantity(1., unit=openmm.unit.angstrom)
  
  def read_pdb(f, keep_hoh=False):
    """ returns atomic numbers, positions, and ribbon_indices """
    ribbons = []
    atomic_numbers = []
    indices = []
    pdb = openmm.app.PDBFile(f)
    positions = np.array(pdb.positions/ANGSTROM)
    for chain in pdb.topology.chains():
      ribbon = []
      for residue in chain.residues():
        if "HOH" != residue.name or keep_hoh:
          for atom in residue.atoms():
            if atom.name in ["C", "CA", "N", "O"]:
              ribbon.append(len(atomic_numbers))
            atomic_numbers.append(atom.element.atomic_number)
            indices.append(atom.index)
      if len(ribbon) > 0:
        ribbons.append(ribbon)
    
    indices = np.array(indices)
    atomic_numbers = np.array(atomic_numbers)
    positions = positions[indices]
    ribbons = [np.array(ribbon) for ribbon in ribbons]
    return atomic_numbers, positions, ribbons
    



if __name__ == "__main__":
  with open("/home/phillip/projects/eralpha/atoms-display/test_files/9PF2.pdb", "r") as f:
    read_pdb(f)


