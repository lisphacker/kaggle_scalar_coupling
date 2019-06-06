bond_distances = {
    ('C', 'C'): (1.2, 1.54),
    ('C', 'H'): (1.06, 1.12),
    ('C', 'O'): (1.43, 2.15),
    ('C', 'N'): (1.47, 2.10),
    ('C', 'F'): (1.34 - 0.04, 1.34 + 0.04),
    ('O', 'H'): (0.98 - 0.04, 0.98 + 0.04),
    ('C', 'O'): (1.43 - 0.04, 1.43 + 0.04)
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
        
        if pair == ('O', 'C') or pair == ('C', 'O'):
            print(atom1.symbol, atom2.symbol, dist)

        
        if pair in bond_distances:
            bond_dist_min, bond_dist_max = bond_distances[pair]
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
        
        return sio.getvalue()