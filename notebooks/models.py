import pandas as pd
import numpy as np

import xgboost as xgb

from util import score

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

        self.coupling_inputs, self.atom_inputs, self.bond_inputs = self.make_input(input_df)
        
        if output_df is not None:   
            self.output = self.make_output(output_df)

        self.combined_inputs = self.combine_inputs((self.coupling_inputs, self.atom_inputs, self.bond_inputs))

    def make_input(self, data):
        n = len(data)

        coupling_input = np.zeros((len(self.coupling_types), n), dtype='float32')
        atom_input = [None] * 4
        for i in range(len(atom_input)):
            atom_input[i] = np.zeros((len(self.atom_types), n), dtype='float32')

        bond_input = [None] * 3
        for i in range(len(bond_input)):
            bond_input[i] = np.zeros((3, n), dtype='float32')

        for t in self.coupling_type_index:
            coupling_input[self.coupling_type_index[t], data.type == t] = 1

        for i, row in enumerate(data.itertuples()):
            # coupling_input[type_index[row.type], i] = 1
            
            m = self.molecules[row.molecule_name]
            bonds = m.bonds
            
            path = m.compute_path(row.atom_index_0, row.atom_index_1)
            syms = [m.symbols[idx] for idx in path]
            
            atom_input[0][self.atom_type_index[syms[0]], i] = 1
            
            try:
                i0 = path[0]
                for j, i1 in enumerate(path[1:]):
                    b = bonds.get((i0, i1), None)
                    if b is None:
                        b = bonds.get((i1, i0), None)
                    if b is None:
                        print(f'Unable to resolve bond - path = {path}, bond = {(i0, i1)})')
                        i0 = i1
                        continue

                    j2 = j + 1

                    bond_input[j][:, i] = [b.dist, b.valency, b.strength]            
                    atom_input[j2][atom_index[syms[j2]], i] = 1

                    i0 = i1
            except:
                pass
                
        return (coupling_input, atom_input, bond_input)

    def make_output(self, data):
        n = len(data)
        output = np.zeros(n, dtype='float32')
        #output[:] = data.norm_scc
        output[:] = data.scalar_coupling_constant
            
        return output

    def combine_inputs(self, inputs_tuple):
        coupling_input, atom_input, bond_input = inputs_tuple
        
        inputs = [coupling_input]
        inputs.extend(atom_input)
        inputs.extend(bond_input)
        
        w, h = coupling_input.shape
        for a in atom_input:
            w += a.shape[0]
        for a in bond_input:
            w += a.shape[0]
        
        input = np.empty((w, h), dtype='float32')
        i = 0
        for a in inputs:
            w, _ = a.shape
            input[i:i + w, :] = a
            i += w
        
        return input


class XGBModel(Model):
    def __init__(self, **kwargs):
        Model.__init__(self, **kwargs)

        self.model = xgb.XGBRegressor()

    def fit(self, input_df, output_df):
        self.setup_data(input_df, output_df)

        self.model.fit(self.combined_inputs.T, self.output.reshape((len(self.output), 1)))

    def evaluate(self, input_df, output_df):
        self.setup_data(input_df, output_df)

        test_output = self.model.predict(self.combined_inputs.T)
        return score(output_df, self.output, test_output)

