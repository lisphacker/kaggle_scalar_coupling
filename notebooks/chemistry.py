from dataclasses import dataclass
import io

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

cc1_err = 0.20 # 0.21
ch1_err = 0.16
cn1_err = 0.32
co1_err = 0.11
hn1_err = 0.14
ho1_err = 0.07
nn2_err = 0.08
no1_err = 0.17
oo1_err = 0.13
err = 0.05

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
    ('C', 'H'): [(1.09 - ch1_err, 1.09 + ch1_err)],
    ('C', 'N'): [(1.47 - cn1_err, 1.47 + cn1_err), (1.25 - err, 1.25 + err), (1.16 - err, 1.16 + err), (1.34 - err, 1.34 + err), (1.32 - err, 1.32 + err)],
    ('C', 'O'): [(1.43 - co1_err, 1.43 + co1_err), (1.21 - err, 1.21 + err), (1.29 - err, 1.29 + err)],
    ('F', 'N'): [(1.36 - err, 1.36 + err)],
    ('H', 'N'): [(1.01 - hn1_err, 1.01 + hn1_err)],
    ('H', 'O'): [(0.98 - ho1_err, 0.98 + ho1_err)],
    ('N', 'N'): [(1.45 - err, 1.45 + err), (1.25 - nn2_err, 1.25 + nn2_err), (1.1 - err, 1.1 + err), (1.35 - err, 1.35 + err)],
    ('N', 'O'): [(1.4 - no1_err, 1.4 + no1_err), (1.21 - err, 1.21 + err), (1.24 - err, 1.24 + err)],
    ('O', 'O'): [(1.48 - oo1_err, 1.48 + oo1_err)]
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
    'H': 4,
    'F': 19,
    'O': 16,
    'N': 14
}

class Molecule:
    __slots__ = ['name', 'n_atoms', 'positions', 'symbols', 'bonds']

    def __init__(self, df):

        self.__init(df)    
        self.__compute_bonds()

    def __init(self, df):
        self.name = df.iloc[0, 0]

        self.n_atoms = len(df)

        self.positions = np.empty((self.n_atoms, 3), dtype='float32')
        self.symbols = [None] * self.n_atoms

        self.positions[:, 0] = df.x.array
        self.positions[:, 1] = df.y.array
        self.positions[:, 2] = df.z.array
        self.symbols = [c for c in df.atom.array]

    def __compute_bonds(self):
        bonds = []
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
                    
                bonds.append(bond)
        self.bonds = bonds#np.array(bonds, dtype='int8')

    def __compute_bond(self, i1, i2, s1, s2, p1, p2):
        pair = (s1, s2)

        #s = set([2, 7])
        #if atom1.index in s and atom2.index in s:
        #    print(pair, pair in bond_distances)
        if pair in bond_distances:
            dist = self.__get_distance(p1, p2)
            bd_pair = bond_distances[pair]
            for bond_dist_min, bond_dist_max in bd_pair:
                #print(pair, atom1.index, atom2.index, dist, bond_dist_min, bond_dist_max, dist >= bond_dist_min and dist <= bond_dist_max)
                #if atom1.index in s and atom2.index in s:
                #    print(pair, atom1.index, atom2.index, dist, bond_dist_min, bond_dist_max, dist >= bond_dist_min and dist <= bond_dist_max)
                #    pass

                if dist >= bond_dist_min and dist <= bond_dist_max:
                    return (i1, i2)
            
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

    for i1, i2 in molecule.bonds:
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
