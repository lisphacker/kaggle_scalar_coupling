from dataclasses import dataclass
import io

import numpy as np
from pprint import pprint

cc1_err = 0.09
err = 0.04

# bond_distances = {
#     ('C', 'C'): (1.2, 1.54),
#     ('C', 'F'): (1.34 - 0.04, 1.34 + 0.04),
#     ('C', 'H'): (1.06, 1.12),
#     ('C', 'N'): (1.15, 2.10),
#     ('C', 'O'): (1.43, 2.15),
#     ('N', 'H'): (0.99 - 0.04, 0.99 + 0.04),
#     ('O', 'H'): (0.98 - 0.04, 0.98 + 0.04),
# }
bond_distances = {
    ('C', 'C'): [(1.54 - cc1_err, 1.54 + cc1_err), (1.34 - err, 1.34 + err), (1.20 - err, 1.20 + err), (1.40 - err, 1.40 + err)],
    ('C', 'F'): [(1.34 - err, 1.34 + err)],
    ('C', 'H'): [(1.09 - err, 1.09 + err)],
    ('C', 'N'): [(1.47 - err, 1.47 + err), (1.25 - err, 1.25 + err), (1.16 - err, 1.16 + err), (1.34 - err, 1.34 + err)],
    ('C', 'O'): [(1.43 - err, 1.43 + err), (1.21 - err, 1.21 + err)],
    ('N', 'H'): [(0.99 - err, 0.99 + err)],
    ('O', 'H'): [(0.98 - err, 0.98 + err)],
}

class Atom:
    def __init__(self, symbol, position, index=-1):
        self.symbol = symbol
        self.position = np.array(position)
        self.index = index
        
    def __eq__(self, o):
        return self.symbol == o.symbol and self.position == o.position
    
@dataclass
class Bond:
    atom1: Atom
    atom2: Atom
        
    def __eq__(self, o):
        return self.atom1.index == o.atom1.index and self.atom2.index == o.atom2.index

class Molecule:
    def __init__(self, name, atoms):
        self.name = name
        self.atoms = atoms
        self.bonds = []
        
        for i, atom in enumerate(self.atoms):
            atom.index = i
            
        self.__compute_bonds()
        
    def __compute_bonds(self):
        for atom1 in self.atoms:
            for atom2 in self.atoms:
                if atom1.index >= atom2.index:
                    continue
                    
                bond = self.__compute_bond(atom1, atom2)
                if bond is None:
                    bond = self.__compute_bond(atom2, atom1)
                if bond is None:
                    continue
                    
                self.bonds.append(bond)
                
    def __compute_bond(self, atom1, atom2):
        dist = self.__get_distance(atom1, atom2)
        pair = (atom1.symbol, atom2.symbol)

        if pair in bond_distances:
            for bond_dist_min, bond_dist_max in bond_distances[pair]:
                #print(pair, atom1.index, atom2.index, dist, bond_dist_min, bond_dist_max)
                if dist >= bond_dist_min and dist <= bond_dist_max:
                    return Bond(atom1, atom2)
            
        return None

    def __get_distance(self, atom1, atom2):
        diff = atom1.position - atom2.position
        return np.sqrt(np.dot(diff, diff))
    
    def __repr__(self):
        sio = io.StringIO()
        sio.write('Name: {}\n'.format(self.name))
        
        sio.write('Atoms:\n')
        for atom in self.atoms:
            sio.write('  {} {}: {}\n'.format(atom.symbol, atom.index, atom.position))
            
        if len(self.bonds) > 0:
            sio.write('Bonds:\n')
            for bond in self.bonds:
                sio.write('  {}({}) - {}({})\n'.format(bond.atom1.symbol, bond.atom1.index, bond.atom2.symbol, bond.atom2.index))
        sio.write('\n')
        
        return sio.getvalue()

def compute_path(molecule, atomidx0, atomidx1):
    unvisited = set([atom.index for atom in molecule.atoms])
    edges = set([(bond.atom1.index, bond.atom2.index) for bond in molecule.bonds] + [(bond.atom2.index, bond.atom1.index) for bond in molecule.bonds])
    #pprint(edges)
    previous_node = {atom.index:None for atom in molecule.atoms}
    n_atoms = len(unvisited)

    dist = np.empty(n_atoms, dtype='float32')
    dist[:] = np.inf
    dist[atomidx0] = 0

    while len(unvisited) > 0:
        mx = np.inf
        mxi = -1
        for u in unvisited:
            if dist[u] < mx:
                mxi = u
                mx = dist[u]

        if mxi == -1:
            break

        u = mxi
        unvisited.remove(u)

        for i, j in edges:
            if i == u:                
                new_dist = dist[u] + 1
                if new_dist < dist[j]:
                    dist[j] = new_dist
                    previous_node[j] = i

    node = atomidx1
    path = [node]
    while previous_node[node] is not None:
        node = previous_node[node]
        path.append(node)

    return list(reversed(path))
