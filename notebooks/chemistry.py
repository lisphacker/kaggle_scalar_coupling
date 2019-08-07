from dataclasses import dataclass
import io

from collections import namedtuple

import numpy as np
from pprint import pprint

import math

from functools import reduce
import ase
import ase.visualize

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import plotly
import plotly.graph_objs as go

import ipyvolume.pylab as p3

#cc1_err = 0.206
cc1_err = 0.21
#cc1_err = 0.203

ch1_err = 0.16
cn1_err = 0.32
co1_err = 0.11
hn1_err = 0.14
ho1_err = 0.07
nn2_err = 0.08
no1_err = 0.17
oo1_err = 0.13
err = 0.05

BondDef = namedtuple('BondDef', ['min_dist', 'max_dist', 'valency', 'strength'])
Bond = namedtuple('Bond', ['dist', 'valency', 'strength'])

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
    ('C', 'C'): [
        BondDef(1.54 - cc1_err, 1.54 + cc1_err, 1, 346), 
        BondDef(1.34 - err, 1.34 + err, 2, 602), 
        BondDef(1.20 - err, 1.20 + err, 3, 835), 
        BondDef(1.40 - err, 1.40 + err, 1.5, 518)
    ],
    ('C', 'F'): [BondDef(1.34 - err, 1.34 + err, 1, 135)],
    ('C', 'H'): [BondDef(1.09 - ch1_err, 1.09 + ch1_err, 1, 411)],
    ('C', 'N'): [
        BondDef(1.47 - cn1_err, 1.47 + cn1_err, 1, 305), 
        BondDef(1.25 - err, 1.25 + err, 2, 615), 
        BondDef(1.16 - err, 1.16 + err, 3, 887), 
        BondDef(1.34 - err, 1.34 + err, 1.5, -1), 
        BondDef(1.32 - err, 1.32 + err, 1.5, -1) # Check this
    ],
    ('C', 'O'): [
        BondDef(1.43 - co1_err, 1.43 + co1_err, 1, 358), 
        BondDef(1.21 - err, 1.21 + err, 2, 799), 
        BondDef(1.29 - err, 1.29 + err, 1.5, -1) # Check this
    ],
    ('F', 'N'): [BondDef(1.36 - err, 1.36 + err, 1, 283)],
    ('H', 'N'): [BondDef(1.01 - hn1_err, 1.01 + hn1_err, 1, 386)],
    ('H', 'O'): [BondDef(0.98 - ho1_err, 0.98 + ho1_err, 1, 459)],
    ('N', 'N'): [BondDef(1.45 - err, 1.45 + err, 1, 167), 
        BondDef(1.25 - nn2_err, 1.25 + nn2_err, 2, 418), 
        BondDef(1.1 - err, 1.1 + err, 3, 942), 
        BondDef(1.35 - err, 1.35 + err, 1.5, -1)
    ],
    ('N', 'O'): [
        BondDef(1.4 - no1_err, 1.4 + no1_err, 1, 201), 
        BondDef(1.21 - err, 1.21 + err, 2, 607), 
        BondDef(1.24 - err, 1.24 + err, 1.5, -1) # Check this
    ],
    ('O', 'O'): [BondDef(1.48 - oo1_err, 1.48 + oo1_err, 1, 142)]
}

atom_symbols = ['C', 'H', 'F', 'O', 'N']

atom_color = {
    'C': 'black',
    'H': 'white',
    'F': 'green',
    'O': 'red',
    'N': 'blue'
}

atom_size = {
    'C': 12,
    'H': 1,
    'F': 19,
    'O': 16,
    'N': 14,
    'X': 0
}

atomic_number = {
    'C': 6,
    'H': 1,
    'F': 9,
    'O': 8,
    'N': 7,
    'X': 0
}
valency = {    
    'C': 4,
    'H': 1,
    'F': 1,
    'O': 2,
    'N': 3
}

class Molecule:
    __slots__ = [
        'name', 'n_atoms', 'positions', 'symbols', 'bonds', 
        'test_atom_index_set', 'simple_field_intensity', 'field_intensity',
        'force_per_pair', 'force']

    def __init__(self, df=None, test_atom_index_set=None):
        self.test_atom_index_set = test_atom_index_set

        if df is not None:
            self.__init(df, test_atom_index_set)    
            self.__compute_bonds()

    def __init(self, df, test_atom_index_set):
        self.name = df.iloc[0, 0]

        self.n_atoms = len(df)

        self.positions = np.empty((self.n_atoms, 3), dtype='float32')
        self.symbols = [None] * self.n_atoms

        self.positions[:, 0] = df.x.array
        self.positions[:, 1] = df.y.array
        self.positions[:, 2] = df.z.array
        self.symbols = [c for c in df.atom.array]

    @staticmethod
    def copy(m):
        mnew = Molecule()
        mnew.name = m.name
        mnew.n_atoms = m.n_atoms
        mnew.positions = m.positions
        mnew.symbols = m.symbols
        mnew.bonds = m.bonds
        return mnew

    def __compute_bonds(self):
        bonds = {}

        # bond_count = [0] * self.n_atoms
        # bond_valency = np.zeros((self.n_atoms, self.n_atoms))
        # bond_length = np.zeros((self.n_atoms, self.n_atoms))
        # bond_energy = np.zeros((self.n_atoms, self.n_atoms))

        for i1 in range(self.n_atoms):
            s1 = self.symbols[i1]
            p1 = self.positions[i1, :]
            for i2 in range(self.n_atoms):
                if i1 >= i2:
                    continue
                s2 = self.symbols[i2]
                p2 = self.positions[i2, :]
                    
                bond = self.__compute_bond(i1, i2, s1, s2, p1, p2)
                if bond is None:
                    bond = self.__compute_bond(i2, i1, s2, s1, p2, p1)
                if bond is None:
                    continue

                # bond_count[i1] += bond.valency
                # bond_count[i2] += bond.valency
                # bond_valency[i1, i2] = bond_valency[i2, i1] = bond.valency
                # bond_length[i1, i2] = bond_length[i2, i1] = bond.dist
                # bond_energy[i1, i2] = bond_energy[i2, i1] = bond.strength

                #bonds.append((i1, i2, bond))
                bonds[(i1, i2)] = bond

        # bonds_pruned = False

        # # Prune bonds
        # for i in range(self.n_atoms):
        #     v = valency[self.symbols[i]]
        #     if bond_count[i] > v:
        #         print('=====')
        #         print(i, bond_count[i])
        #         print(bond_valency[i, :])
        #         print(bond_energy[i, :])


        self.bonds = bonds#np.array(bonds, dtype='int8')

    def __compute_bond(self, i1, i2, s1, s2, p1, p2):
        pair = (s1, s2)

        if pair in bond_distances:
            dist = self.__get_distance(p1, p2)
            bond_defs_for_pair = bond_distances[pair]
            for bond_def in bond_defs_for_pair:
                if self.test_atom_index_set is not None and i1 in self.test_atom_index_set and i2 in self.test_atom_index_set:
                    print(pair, i1, i2, dist, bond_def.min_dist, bond_def.max_dist, dist >= bond_def.min_dist and dist <= bond_def.max_dist)

                if dist >= bond_def.min_dist and dist <= bond_def.max_dist:
                    return Bond(dist, bond_def.valency, bond_def.strength)
            
        return None

    def __get_distance(self, p1, p2):
        diff = p1 - p2
        return np.sqrt(np.dot(diff, diff))
    
    def __repr__(self):
        sio = io.StringIO()
        sio.write('Name: {}\n'.format(self.name))
        
        sio.write('Atoms:\n')
        for i in range(self.n_atoms):
            sio.write('  {} {}: {}\n'.format(self.symbols[i], i, self.positions[i, :]))
                        
        if len(self.bonds) > 0:
            sio.write('Bonds:\n')
            for i1, i2 in self.bonds:
                sio.write('  {}({}) - {}({})\n'.format(self.symbols[i1], i1, self.symbols[i2], i2))

        try:
            if self.field_intensity is not None and len(self.field_intensity) > 0:
                sio.write('Simple field intensity :\n')
                for i in range(self.n_atoms):
                    sio.write('  {}({}) - {}\n'.format(self.symbols[i], i, self.simple_field_intensity[i]))
        except:
            pass

        try:
            if self.field_intensity is not None and len(self.field_intensity) > 0:
                sio.write('Field intensity :\n')
                for i in range(self.n_atoms):
                    sio.write('  {}({}) - {}\n'.format(self.symbols[i], i, self.field_intensity[i]))
        except:
            pass

        sio.write('\n')

        return sio.getvalue()

    def compute_path(self, i1, i2):
        unvisited = set(range(self.n_atoms))
        unvisited_count = self.n_atoms
        previous_node = [None] * self.n_atoms

        edges = {}
        for i in range(self.n_atoms):
            edges[i] = []

        for i, j in self.bonds:
            edges[i].append(j)
            edges[j].append(i)

        dist = [math.inf] * self.n_atoms
        dist[i1] = 0

        while unvisited_count > 0:
            mn = math.inf
            mni = -1
            for u in unvisited:
                if dist[u] < mn:
                    mni = u
                    mn = dist[u]

            if mni == -1:
                break

            u = mni
            unvisited.remove(u)
            unvisited_count -= 1

            for v in edges[u]:
                new_dist = dist[u] + 1
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    previous_node[v] = u

        node = i2
        path = [node]
        while previous_node[node] is not None:
            node = previous_node[node]
            path.append(node)

        return list(reversed(path))

def make_ase_atoms(molecule):
    symbols = []
    positions = []
    data = []
    for atom in molecule.atoms:
        symbols.append(atom.symbol)
        positions.append(atom.position)
        data.append(atom.index)

    return ase.Atoms(symbols=symbols, positions=positions), data

def ase_plot_molecule(molecule):
    aseatoms, data = make_ase_atoms(molecule)
    return ase.visualize.view(aseatoms, viewer="x3d", data=data)

def plt_plot_molecule(molecule):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')

    x = molecule.positions[:, 0]
    y = molecule.positions[:, 1]
    z = molecule.positions[:, 2]
    c = [atom_color[symbol] for symbol in molecule.symbols]
    s = [20 * atom_size[symbol] for symbol in molecule.symbols]

    ax.scatter(x, y, zs=z, c=c, s=s, edgecolor='black')

    for i in range(molecule.n_atoms):
        ax.text(molecule.positions[i, 0], molecule.positions[i, 1], molecule.positions[i, 2], '{} {}'.format(molecule.symbols[i], i))

    for i1, i2, b in molecule.bonds:
        a1 = molecule.positions[i1, :]
        a2 = molecule.positions[i2, :]
        ax.plot([a1[0], a2[0]], [a1[1], a2[1]], [a1[2], a2[2]], 'black')

    plt.show()

def plotly_plot_molecule(molecule):
    x = [atom.position[0] for atom in molecule.atoms]
    y = [atom.position[1] for atom in molecule.atoms]
    z = [atom.position[2] for atom in molecule.atoms]
    c = [atom_color[atom.symbol] for atom in molecule.atoms]
    s = [20 * atom_size[atom.symbol] for atom in molecule.atoms]

    sg = go.Scatter3d(x=x, y=y, z=z,
                      mode='markers',
                      marker={
                          'size': 10,
                          'opacity': 1.0
                      })
    layout = go.Layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0})

    figure = go.FigureWidget(data=[sg], layout=layout)
    plotly.offline.iplot(figure)

def ipv_plot_molecule(molecule):
    for atom_symbol in atom_symbols:
        atoms = [atom for atom in molecule.atoms if atom.symbol == atom_symbol]
        if len(atoms) == 0:
            continue

        x = np.array([atom.position[0] for atom in atoms])
        y = np.array([atom.position[1] for atom in atoms])
        z = np.array([atom.position[2] for atom in atoms])

        p3.scatter(x=x, y=y, z=z, color=atom_color[atom_symbol], size=atom_size[atom_symbol], marker='sphere')

    for bond in molecule.bonds:
        p1 = bond.atom1.position
        p2 = bond.atom2.position
        p3.plot(x=np.array([p1[0], p2[0]]),
            y = np.array([p1[1], p2[1]]),
            z = np.array([p1[2], p2[2]]),
            color='black')
    p3.show()        
