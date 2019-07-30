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
        self.coupling_types = set(input_df.type.unique())
        if output_df is not None:
            self.coupling_types = self.coupling_types | set(output_df.type.unique())
        self.coupling_type_index = {t:i for i, t in enumerate(self.coupling_types)}

        self.input_df = self.make_input(input_df)

        numeric_col_names = []        
        columns_to_remove = set(['id', 'scalar_coupling_constant', 'atom_index_0', 'atom_index_1'])
        dtypes_to_keep = set([np.dtype(tn) for tn in ['int8', 'int16', 'float32']])
        for col_name, dtype in self.input_df.dtypes.items():
            if col_name in columns_to_remove:
                continue
            if dtype not in dtypes_to_keep:
                continue
            numeric_col_names.append(col_name)

        self.numeric_input_df = self.input_df[numeric_col_names]
        
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

    def make_atom_columns(self, df, atom_sym_column, atom_index_column, prefix):
        for a in self.atom_types:
            df[f'{prefix}_{a}'] = (atom_sym_column == a).astype('int8')
        df[f'{prefix}_weight'] = pd.Series([chemistry.atom_size[atom] for atom in atom_sym_column], index=df.index, dtype='int8')

        df['ai'] = atom_index_column

        m = df.merge(self.structures, left_on=['molecule_name', 'ai'], right_on=['molecule_name', 'atom_index'])
        m.index = df.index
        #print(df.columns)
        #print(self.structures.columns)
        #print(m.columns)

        df.drop(columns=['ai'])
        for c in ['dist_to_mean']:
            df[f'{prefix}_{c}'] = m[c]

        #print(df.columns)
        #print()


    def make_input(self, input_df):
        df = self.make_dist(input_df)

        for t in self.coupling_types:
            df[f'coupling_{t}'] = (df.type == t).astype('int8')

        self.make_atom_columns(df, df.atom_0, df.atom_index_0, 'atom0')
        self.make_atom_columns(df, df.atom_1, df.atom_index_1, 'atomN')

        n = len(input_df)

        atom_syms = [None] * 3
        atom_indices = [None] * 3
        for i in range(3):
            atom_syms[i] = pd.Categorical(['X'] * n, categories=list(self.atom_types) + ['X'])
            atom_indices[i] = np.zeros(n, dtype='int16')

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
                    atom_indices[ai][i] = i1

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

        self.make_atom_columns(df, pd.Series(atom_syms[0], index=df.index), pd.Series(atom_indices[0], index=df.index), 'atom1')
        self.make_atom_columns(df, pd.Series(atom_syms[1], index=df.index), pd.Series(atom_indices[1], index=df.index), 'atom2')
        self.make_atom_columns(df, pd.Series(atom_syms[2], index=df.index), pd.Series(atom_indices[2], index=df.index), 'atom3')

        for i in range(3):
            bi = i * 3
            df[f'bond{i}{i + 1}_dist']     = pd.Series(bond_info[bi    , :], index=df.index)
            df[f'bond{i}{i + 1}_valency']  = pd.Series(bond_info[bi + 1, :], index=df.index)
            df[f'bond{i}{i + 1}_strength'] = pd.Series(bond_info[bi + 2, :], index=df.index)

        # cols = ['molecule_name']
        # for c in self.structures.columns:
        #     if c[0] == 'n':
        #         cols.append(c)
        # df = df.merge(self.structures[cols].groupby('molecule_name').nth(0), how='left', left_on=['molecule_name'], right_on=['molecule_name'])

        return df
        
    def make_output(self, output_df):
        return output_df.loc[:, ['scalar_coupling_constant']]

class SKModel(Model):
    def __init__(self, flatten_output=True, **model_args):
        Model.__init__(self, **model_args)

        self.flatten_output = flatten_output

    def fit(self, input_df, output_df):
        self.setup_data(input_df, output_df)

        ref_output = self.output_df.values.flatten() if self.flatten_output else self.output_df.values.reshape((len(self.output_df), 1))
        self.model.fit(self.numeric_input_df.values, ref_output)

    def corr(self, input_df, output_df):
        self.setup_data(input_df, output_df)
        return self.input_df.corr()

    def evaluate(self, input_df, output_df):
        self.setup_data(input_df, output_df)

        ref_output = self.output_df.values.flatten() if self.flatten_output else self.output_df.values.reshape((len(self.output_df), 1))
        test_output = self.model.predict(self.numeric_input_df.values)
        return test_output, score(output_df, ref_output, test_output)

    def predict(self, input_df):
        self.setup_data(input_df)

        return self.model.predict(self.numeric_input_df.values)

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