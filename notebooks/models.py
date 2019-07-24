import pandas as pd
import numpy as np

from numba import jit

import xgboost as xgb
import lightgbm as lgb

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense

from util import score
import chemistry

from pprint import pprint

def partition_data(data_df, count=None, train_frac=0.7):
    n_labelled = count if count is not None else len(data_df)
    n_train = int(n_labelled * train_frac)
    indices = np.arange(0, n_labelled)
    np.random.shuffle(indices)
    
    train_indices = indices[0:n_train]
    test_indices = indices[n_train:]
    
    train = data_df.iloc[train_indices, :]
    test = data_df.iloc[test_indices, :]

    return train, test

class Model:
    def __init__(self, molecules, structures):
        if molecules is None:
            raise Exception('molecules cannot be None')

        if structures is None:
            raise Exception('structures cannot be None')

        self.molecules = molecules
        self.structures = structures
            
        self.atom_types = set(structures.atom.unique())
        self.atom_type_index = {a:i for i, a in enumerate(self.atom_types)}

    def setup_data(self, input_df, output_df=None):
        self.coupling_types = set(input_df.type.unique()) | set(output_df.type.unique())
        self.coupling_type_index = {t:i for i, t in enumerate(self.coupling_types)}

        self.coupling_types = set(input_df.type.unique()) | set(output_df.type.unique())

        self.input_df = self.make_input(input_df)
        
        if output_df is not None:   
            self.output_df = self.make_output(output_df)

        # print(self.input_df.columns)
        # print(self.input_df)
        # print()
        # print(self.output_df)

    def make_dist(self, data_df):
        m0 = data_df.merge(self.structures, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'], suffixes=('0', '0'))
        m1 = data_df.merge(self.structures, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'], suffixes=('1', '1'))

        l0 = m0[['x', 'y', 'z']]
        l1 = m1[['x', 'y', 'z']]
        d = l0 - l1
        d_sq = d * d
        dist_sq = d_sq.x + d_sq.y + d_sq.z
        dist = dist_sq.apply(np.sqrt)
        dist.name = 'coupling_distance'

        dist.index = data_df.index
        m0.atom.index = data_df.index
        m1.atom.index = data_df.index
        
        merged = data_df.join(dist)
        merged['atom_0'] = m0.atom
        merged['atom_1'] = m1.atom

        return merged

    def make_atom_columns(self, df, atom_sym_column, prefix):
        for a in self.atom_types:
            df[f'{prefix}_{a}'] = (atom_sym_column == a).astype('int8')
        df[f'{prefix}_weight'] = pd.Series([chemistry.atom_size[atom] for atom in atom_sym_column], index=df.index, dtype='int8')

    def make_input(self, input_df):
        df = self.make_dist(input_df)

        for t in self.coupling_types:
            df[f'coupling_{t}'] = (df.type == t).astype('int8')

        self.make_atom_columns(df, df.atom_0, 'atom0')
        self.make_atom_columns(df, df.atom_1, 'atomN')

        n = len(input_df)

        atom_syms = [None] * 3
        for i in range(3):
            atom_syms[i] = pd.Categorical(['X'] * n, categories=list(self.atom_types) + ['X'])

        bond_info = np.zeros((9, n), dtype='float32')

        for i, row in enumerate(input_df.itertuples()):
            m = self.molecules[row.molecule_name]
            bonds = m.bonds

            path = m.compute_path(row.atom_index_0, row.atom_index_1)
            syms = [m.symbols[idx] for idx in path]

            try:
                i0 = path[0]
                for ai, (i1, sym) in enumerate(zip(path[1:], syms[1:])):
                    atom_syms[ai][i] = sym

                    b = bonds.get((i0, i1), None)
                    if b is None:
                        b = bonds.get((i1, i0), None)
                    if b is None:
                        print(f'Unable to resolve bond - path = {path}, bond = {(i0, i1)})')
                        i0 = i1
                        continue
                    bi = ai * 3
                    bond_info[bi, i]     = b.dist
                    bond_info[bi + 1, i] = b.valency
                    bond_info[bi + 2, i] = b.strength

                    i0 = i1
            except:
                pass

        self.make_atom_columns(df, pd.Series(atom_syms[0], index=df.index), 'atom1')
        self.make_atom_columns(df, pd.Series(atom_syms[1], index=df.index), 'atom2')
        self.make_atom_columns(df, pd.Series(atom_syms[2], index=df.index), 'atom3')

        for i in range(3):
            bi = i * 3
            df[f'bond{i}{i + 1}_dist']     = pd.Series(bond_info[bi    , :], index=df.index)
            df[f'bond{i}{i + 1}_valency']  = pd.Series(bond_info[bi + 1, :], index=df.index)
            df[f'bond{i}{i + 1}_strength'] = pd.Series(bond_info[bi + 1, :], index=df.index)

        return df
        
    # def make_input_old(self, data):
    #     n = len(data)

    #     coupling_input = np.zeros((len(self.coupling_types), n), dtype='float32')
    #     atom_input = [None] * 5
    #     for i in range(len(atom_input)):
    #         atom_input[i] = np.zeros((len(self.atom_types) + 1, n), dtype='float32')

    #     bond_input = [None] * 3
    #     for i in range(len(bond_input)):
    #         bond_input[i] = np.zeros((3, n), dtype='float32')

    #     for t in self.coupling_type_index:
    #         coupling_input[self.coupling_type_index[t], data.type == t] = 1

    #     for i, row in enumerate(data.itertuples()):
    #         # coupling_input[type_index[row.type], i] = 1
            
    #         m = self.molecules[row.molecule_name]
    #         bonds = m.bonds
            
    #         path = m.compute_path(row.atom_index_0, row.atom_index_1)
    #         syms = [m.symbols[idx] for idx in path]

    #         atom_input[0][self.atom_type_index[syms[0]], i] = 1
    #         atom_input[0][len(self.atom_types), i] = chemistry.atom_size[syms[0]]
            
    #         try:
    #             i0 = path[0]
    #             for j, i1 in enumerate(path[1:]):
    #                 b = bonds.get((i0, i1), None)
    #                 if b is None:
    #                     b = bonds.get((i1, i0), None)
    #                 if b is None:
    #                     print(f'Unable to resolve bond - path = {path}, bond = {(i0, i1)})')
    #                     i0 = i1
    #                     continue

    #                 j2 = j + 1

    #                 bond_input[j][:, i] = [b.dist, b.valency, b.strength]            
    #                 atom_input[j2][self.atom_type_index[syms[j2]], i] = 1
    #                 atom_input[j2][len(self.atom_types), i] = chemistry.atom_size[syms[j2]]

    #                 i0 = i1

    #             atom_input[4][:, i] = atom_input[len(path) - 1][:, i]
    #         except Exception as e:
    #             pass
                
    #     # print('=====================')
    #     # print(self.atom_type_index, self.coupling_type_index)
    #     # print()
    #     # print(coupling_input)
    #     # print()
    #     # print(atom_input)
    #     # print()
    #     # print(bond_input)
    #     # print()

    #     return (coupling_input, atom_input, bond_input)

    def make_output(self, output_df):
        return output_df.loc[:, ['scalar_coupling_constant']]

    # def make_output_old(self, data):
    #     n = len(data)
    #     output = np.zeros(n, dtype='float32')
    #     #output[:] = data.norm_scc
    #     output[:] = data.scalar_coupling_constant
            
    #     return output

    # def combine_inputs(self, inputs_tuple):
    #     coupling_input, atom_input, bond_input = inputs_tuple
        
    #     inputs = [coupling_input]
    #     inputs.extend(atom_input)
    #     inputs.extend(bond_input)
        
    #     w, h = coupling_input.shape
    #     for a in atom_input:
    #         w += a.shape[0]
    #     for a in bond_input:
    #         w += a.shape[0]
        
    #     input = np.empty((w, h), dtype='float32')
    #     i = 0
    #     for a in inputs:
    #         w, _ = a.shape
    #         input[i:i + w, :] = a
    #         i += w
        
    #     return input

class SKModel(Model):
    def __init__(self, flatten_output=True, **model_args):
        Model.__init__(self, **model_args)

        self.flatten_output = flatten_output

    def fit(self, input_df, output_df):
        self.setup_data(input_df, output_df)

        print(self.input_df.values)

        ref_output = self.output_df.values.flatten() if self.flatten_output else self.output_df.values.reshape((len(self.output_df), 1))
        self.model.fit(self.input_df.values, ref_output)

    def corr(self, input_df, output_df):
        self.setup_data(input_df, output_df)
        return self.input_df.corr()

    def evaluate(self, input_df, output_df):
        self.setup_data(input_df, output_df)

        test_output = self.model.predict(self.combined_inputs.T)
        return test_output, score(output_df, self.output, test_output)

class XGBModel(SKModel):
    def __init__(self, model_args, xgb_args={}):
        SKModel.__init__(self, **model_args)

        self.model = xgb.XGBRegressor(**xgb_args)

class LGBModel(SKModel):
    def __init__(self, model_args, xgb_args={}):
        SKModel.__init__(self, **model_args)

        self.model = lgb.LGBMRegressor(**xgb_args)

class NNModel(Model):
    pass