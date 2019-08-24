import pandas as pd
import numpy as np

from numba import jit

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

#import xgboost as xgb
import lightgbm as lgb

import tensorflow as tf
from tensorflow.keras import Model as KerasModel, Input
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam

from util import score
import chemistry

from pprint import pprint

float32 = np.float32

class DummyScaler:
    def fit(self, X):
        return X

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X

def partition_data(data_df, count=None, train_frac=0.5, valid_frac=0.2):
    n_labelled = count if count is not None else len(data_df)
    n_train = int(n_labelled * train_frac)
    n_valid = int(n_labelled * valid_frac)
    indices = np.arange(0, n_labelled)
    np.random.shuffle(indices)

    max_valid = n_train + n_valid    
    train_indices = indices[0:n_train]
    valid_indices = indices[n_train:max_valid]
    test_indices = indices[max_valid:]
    
    train = data_df.iloc[train_indices, :].copy()
    valid = data_df.iloc[valid_indices, :].copy()
    test = data_df.iloc[test_indices, :].copy()

    return train, valid, test

class MM11Scaler(MinMaxScaler):
    def __init__(self):
        MinMaxScaler.__init__(self, feature_range=(-1, 1))

class Model:
    def __init__(self, molecules, structures, normalize_input=False, verbose=False):
        if molecules is None:
            raise Exception('molecules cannot be None')

        if structures is None:
            raise Exception('structures cannot be None')

        self.molecules = molecules
        self.structures = structures
            
        self.atom_types = set(structures.atom.unique())
        self.atom_type_index = {a:i for i, a in enumerate(self.atom_types)}

        self.normalize_input = normalize_input

        if self.normalize_input:
            # scaler = MM11Scaler
            self.scaler = RobustScaler
            self.input_scaler = self.scaler()
            self.output_scaler = self.scaler()

        self.verbose = verbose

    def setup_data(self, input_df, output_df=None):
        self.coupling_types = set(input_df.type.unique())
        if output_df is not None:
            self.coupling_types = self.coupling_types | set(output_df.type.unique())
        self.coupling_type_index = {t:i for i, t in enumerate(self.coupling_types)}

        input_df = self.make_input(input_df)
        self.make_complex_inputs(input_df)
        self.cleanup_columns(input_df)

        numeric_col_names = []        
        columns_to_remove = set(['id', 'scalar_coupling_constant', 'atom_index_0', 'atom_index_1'])
        dtypes_to_keep = set([np.dtype(tn) for tn in ['int8', 'int16', 'float32']])
        for col_name, dtype in input_df.dtypes.items():
            if col_name in columns_to_remove:
                continue
            if dtype not in dtypes_to_keep:
                continue
            numeric_col_names.append(col_name)

        numeric_input_df = input_df[numeric_col_names]
        
        if output_df is not None:   
            output_df = self.make_output(output_df)
        else:
            output_df = None

        return input_df, numeric_input_df, output_df

    def make_dist(self, data_df):
        m0 = data_df.merge(self.structures, how='left', left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'], suffixes=('0', '0'))
        m1 = data_df.merge(self.structures, how='left', left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'], suffixes=('1', '1'))

        m0.index = data_df.index
        m1.index = data_df.index

        l0 = m0[['x', 'y', 'z']]
        l1 = m1[['x', 'y', 'z']]
        d = l0 - l1
        d_sq = d * d
        dist_sq = d_sq.x + d_sq.y + d_sq.z
        dist = dist_sq.apply(np.sqrt)
        dist.name = 'coupling_distance'

        dist.index = data_df.index

        # dist.index = data_df.index
        # m0.atom.index = data_df.index
        # m1.atom.index = data_df.index

        merged = data_df.join(dist)
        merged['atom_0'] = m0.atom
        merged['atom_1'] = m1.atom

        return merged

    def make_atom_columns(self, df, atom_syms, atom_index_column, prefix):
        atom_sym_column = pd.Categorical(atom_syms, categories=list(self.atom_types) + ['X'])

        for a in self.atom_types:
            df[f'{prefix}_{a}'] = (atom_sym_column == a).astype('int8')
        df[f'{prefix}_weight'] = pd.Series([chemistry.atom_size[atom] for atom in atom_sym_column], index=df.index, dtype='int8')

        df['ai'] = atom_index_column

        m = df.merge(self.structures, how='left', left_on=['molecule_name', 'ai'], right_on=['molecule_name', 'atom_index'])
        m.index = df.index

        mask = atom_sym_column == 'X'

        df.drop(columns=['ai'])
        for c in ['x', 'y', 'z', 'x_mean', 'y_mean', 'z_mean', 'dist_to_mean']:
            df[f'{prefix}_{c}'] = m[c]
            df.loc[mask, [f'{prefix}_{c}']] = 0

        fi_list = [self.molecules[mn].field_intensity[ai] for (mn, ai) in zip(df.molecule_name, atom_index_column)]
        fi_array = np.array(fi_list, dtype='float32')
        df[f'{prefix}_fi'] = pd.Series(fi_array, index=df.index)
        df.loc[mask, [f'{prefix}_fi']] = 0

        fi_list = [self.molecules[mn].simple_field_intensity[ai] for (mn, ai) in zip(df.molecule_name, atom_index_column)]
        fi_array = np.array(fi_list, dtype='float32')
        df[f'{prefix}_simple_fi'] = pd.Series(fi_array, index=df.index)
        df.loc[mask, [f'{prefix}_simple_fi']] = 0

        # force_list = [self.molecules[mn].force[ai] for (mn, ai) in zip(df.molecule_name, atom_index_column)]
        # force_array = np.array(force_list, dtype='float32')
        # df[f'{prefix}_force'] = pd.Series(force_array, index=df.index)
        # df.loc[mask, [f'{prefix}_force']] = 0



    def make_input(self, input_df):
        df = self.make_dist(input_df)
        df['type'] = input_df['type']

        for t in self.coupling_types:
            df[f'coupling_{t}'] = (df.type == t).astype('int8')

        self.make_atom_columns(df, df.atom_0, df.atom_index_0, 'atom0')
        self.make_atom_columns(df, df.atom_1, df.atom_index_1, 'atomN')

        n = len(input_df)

        cc_type = input_df.type.iloc[0]
        n_bonds = int(cc_type[0])
        n_atoms = n_bonds + 1

        atom_syms = [None] * 3
        atom_indices = [None] * 3
        for i in range(3):
            #atom_syms[i] = pd.Categorical(['X'] * n, categories=list(self.atom_types) + ['X'])
            atom_syms[i] = ['X'] * n
            atom_indices[i] = np.zeros(n, dtype='int16')

        sergii_dist = np.zeros((4, n), dtype='float32')

        n_non_sergii_atoms = 8
        sergii_dist2 = np.zeros((4, n_non_sergii_atoms, n), dtype='float32')
        sergii_cos = np.zeros((4, n_non_sergii_atoms, n), dtype='float32')
        sergii_force = np.zeros((4, n_non_sergii_atoms, n), dtype='float32')

        #bond_info = np.zeros((9, n), dtype='float32')
        bond_info_dist = []
        bond_info_valency = []
        bond_info_strength = []
        bond_info_force = []
        bond_info_force2 = []
        bond_info_cos = []
        for i in range(3):
            bond_info_dist.append(np.zeros(n, dtype='float32'))
            bond_info_valency.append(np.zeros(n, dtype='float32'))
            bond_info_strength.append(np.zeros(n, dtype='float32'))
            bond_info_force.append(np.zeros(n, dtype='float32'))
            bond_info_force2.append(np.zeros(n, dtype='float32'))
            bond_info_cos.append(np.zeros(n, dtype='float32'))

        for i, row in enumerate(input_df.itertuples()):
            m = self.molecules[row.molecule_name]
            bonds = m.bonds

            d, d2, cos, force = self.compute_sergii_dist(m, row.atom_index_0, row.atom_index_1, n_non_sergii_atoms)
            sergii_dist[:, i] = d
            sergii_dist2[:, :, i] = d2
            sergii_cos[:, :, i] = cos
            sergii_force[:, :, i] = force

            path = m.compute_path(row.atom_index_0, row.atom_index_1)
            syms = [m.symbols[idx] for idx in path]

            if len(path) != n_bonds + 1:
                print(path, row.atom_index_0, row.atom_index_1, '\n', m.name)
                continue

            i0 = path[0]
            s0 = syms[0]
            for ai, (i1, s1) in enumerate(zip(path[1:], syms[1:])):
                atom_syms[ai][i] = s1
                atom_indices[ai][i] = i1

                # Bond properties
                b = bonds.get((i0, i1), None)
                if b is None:
                    b = bonds.get((i1, i0), None)
                if b is None:
                    print(f'Unable to resolve bond - path = {path}, bond = {(i0, i1)})')
                    i0 = i1
                    continue
                bi = ai
                # bond_info[bi, i]     = b.dist
                # bond_info[bi + 1, i] = b.valency
                # bond_info[bi + 2, i] = b.strength
                bond_info_dist[bi][i]     = b.dist
                bond_info_valency[bi][i] = b.valency
                bond_info_strength[bi][i] = b.strength

                # Compute bond force
                p0 = m.positions[i0]
                p1 = m.positions[i1]
                d = np.linalg.norm(p0 - p1)

                a0 = chemistry.atomic_number[s0]
                a1 = chemistry.atomic_number[s1]

                bond_info_force2[bi][i] = a0 * a1 / d
                bond_info_force[bi][i] = bond_info_force2[bi][i] / d

                if ai < len(path) - 2:
                    i2 = path[ai + 2]
                    p2 = m.positions[i2]
                    d0 = d
                    d1 = np.linalg.norm(p2 - p1)
                    
                    if d0 != 0 and d1 != 0:
                        dir0 = (p0 - p1) / d0
                        dir1 = (p2 - p1) / d1

                        bond_info_cos[bi][i] = np.dot(dir0, dir1)

                # Next pair
                i0 = i1
                s0 = s1

       
        for i in range(1, n_atoms):
            self.make_atom_columns(df, pd.Series(atom_syms[i - 1], index=df.index), pd.Series(atom_indices[i - 1], index=df.index), f'atom{i}')

        for i in range(n_bonds):
            df[f'bond{i}{i + 1}_dist']     = pd.Series(bond_info_dist[i], index=df.index)
            df[f'bond{i}{i + 1}_dist2']    = pd.Series(bond_info_dist[i] * bond_info_dist[i], index=df.index)

            df[f'bond{i}{i + 1}_valency']  = pd.Series(bond_info_valency[i], index=df.index)
            df[f'bond{i}{i + 1}_strength'] = pd.Series(bond_info_strength[i], index=df.index)
            df[f'bond{i}{i + 1}_force']    = pd.Series(bond_info_force[i], index=df.index)
            df[f'bond{i}{i + 1}_force2']    = pd.Series(bond_info_force2[i], index=df.index)

            df[f'bond{i}{i + 1}_cos']      = pd.Series(bond_info_cos[i], index=df.index)
            df[f'bond{i}{i + 1}_cos2']     = pd.Series(bond_info_cos[i] * bond_info_cos[i], index=df.index)
            df[f'bond{i}{i + 1}_sin2']     = 1 - df[f'bond{i}{i + 1}_cos2']

        sergii_cos2 = sergii_cos * sergii_cos
        sergii_sin2 = 1 - sergii_cos2
        sergii_sin2[sergii_sin2 < 0] = 0
        sergii_sin = np.sqrt(sergii_sin2)
        sergii_sin_cos = sergii_sin * sergii_cos

        for i in range(4):
            df[f'sergii_dist_{i}'] = pd.Series(sergii_dist[i, :], index=df.index)
            for j in range(n_non_sergii_atoms):
                df[f'sergii_dist2_{i}_{j}'] = pd.Series(sergii_dist2[i, j, :], index=df.index)
                df[f'sergii_dist2_sq_{i}_{j}'] = pd.Series(sergii_dist2[i, j, :] * sergii_dist2[i, j, :], index=df.index)
                df[f'sergii_dist2_sq_inv_{i}_{j}'] = 1 / df[f'sergii_dist2_sq_{i}_{j}']

                df[f'sergii_cos_{i}_{j}'] = pd.Series(sergii_cos[i, j, :], index=df.index)
                # df[f'sergii_sin_{i}_{j}'] = pd.Series(sergii_sin[i, j, :], index=df.index)
                # df[f'sergii_cos2_{i}_{j}'] = pd.Series(sergii_cos2[i, j, :], index=df.index)
                # df[f'sergii_sin2_{i}_{j}'] = pd.Series(sergii_sin2[i, j, :], index=df.index)
                # df[f'sergii_sin_cos_{i}_{j}'] = pd.Series(sergii_sin_cos[i, j, :], index=df.index)
                df[f'sergii_dist2_cos_{i}_{j}'] = pd.Series(sergii_dist2[i, j, :] * sergii_cos[i, j, :], index=df.index)
                df[f'sergii_dist2_sin_{i}_{j}'] = pd.Series(sergii_dist2[i, j, :] * sergii_sin[i, j, :], index=df.index)

                df[f'sergii_force_{i}_{j}'] = pd.Series(sergii_force[i, j, :], index=df.index)

        # cols = ['molecule_name']
        # for c in self.structures.columns:
        #     if c[0] == 'n':
        #         cols.append(c)
        # df = df.merge(self.structures[cols].groupby('molecule_name').nth(0), how='left', left_on=['molecule_name'], right_on=['molecule_name'])

        return df

    def compute_sergii_dist(self, m, i0, i1, n_non_sergii_atoms):        
        # Sergii's 4 point method
        p0, p1 = m.positions[i0], m.positions[i1]
        mid = (p0 + p1) * 0.5

        diff = m.positions - mid
        dist = np.linalg.norm(diff, axis=1)
        closest_indices = list(dist.argsort())


        closest_indices.remove(i0)
        closest_indices.remove(i1)

        i2 = closest_indices[0]
        p2 = m.positions[i2]
        if len(closest_indices) > 1:
            i3 = closest_indices[1]
            p3 = m.positions[i3]
            mid = (p0 + p1 + p2 + p3) / 4
            sergii_pos = [p0, p1, p2, p3]
        else:
            mid = (p0 + p1 + p2) / 3
            sergii_pos = [p0, p1, p2]

        d0 = np.linalg.norm(p0 - mid)
        d1 = np.linalg.norm(p1 - mid)
        d2 = np.linalg.norm(p2 - mid)

        if len(closest_indices) > 1:
            d3 = np.linalg.norm(p3 - mid)
            n_sergii_atoms = 4
        else:
            d3 = np.float32(0)
            n_sergii_atoms = 3

        diff = m.positions - mid
        dist = np.linalg.norm(diff, axis=1)
        closest_indices = list(dist.argsort())

        positions = m.positions[closest_indices]
        n_atoms = min(len(positions), n_non_sergii_atoms)

        charges = [chemistry.atomic_number[m.symbols[i]] for i in closest_indices]

        dist2 = np.zeros((4, n_non_sergii_atoms), dtype='float32')
        cos = np.zeros((4, n_non_sergii_atoms), dtype='float32')
        force = np.zeros((4, n_non_sergii_atoms), dtype='float32')
        
        for i in range(n_sergii_atoms):
            for j in range(n_atoms):
                dist2[i, j] = np.linalg.norm(sergii_pos[i] - positions[j])

                dir1 = sergii_pos[i] - mid
                dir2 = positions[j] - mid

                dir1 = dir1 / np.linalg.norm(dir1)
                dir2 = dir2 / np.linalg.norm(dir2)

                cos[i, j] = np.dot(dir1, dir2)

                if dist2[i, j] != 0:
                    force[i, j] = charges[i] * charges[j] / (dist2[i, j] * dist2[i, j])

        dist = sorted([d0, d1, d2, d3])
        return dist, dist2, cos, force


    def make_complex_inputs(self, df):
        for atomId in "0N":
            prefix = f'atom{atomId}'
            for axis in "xyz":
                df[f'{prefix}_dir_{axis}'] = (df[f'{prefix}_{axis}'] - df[f'{prefix}_{axis}_mean']) / df[f'{prefix}_dist_to_mean']

        df['cos0N'] = df['atom0_dir_x'] * df['atomN_dir_x'] + df['atom0_dir_y'] * df['atomN_dir_y'] + df['atom0_dir_z'] * df['atomN_dir_z']

        df['atoms0N_dx'] = (df['atom0_x'] - df['atomN_x']).abs()
        df['atoms0N_dy'] = (df['atom0_y'] - df['atomN_y']).abs()
        df['atoms0N_dz'] = (df['atom0_z'] - df['atomN_z']).abs()

        df['atoms0N_dist2'] = df['atoms0N_dx'] * df['atoms0N_dx'] + df['atoms0N_dy'] * df['atoms0N_dy'] + df['atoms0N_dz'] * df['atoms0N_dz']
        df['atoms0N_dist'] = df['atoms0N_dist2'].apply(np.sqrt)

        cc_type = df.type.iloc[0]
        n_atoms = int(cc_type[0]) + 1

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                prefix = f'atoms{i}{j}'                
                df[f'{prefix}_dx'] = (df[f'atom{i}_x'] - df[f'atom{j}_x']).abs()
                df[f'{prefix}_dy'] = (df[f'atom{i}_y'] - df[f'atom{j}_y']).abs()
                df[f'{prefix}_dz'] = (df[f'atom{i}_z'] - df[f'atom{j}_z']).abs()

                df[f'{prefix}_dist2'] = df[f'{prefix}_dx'] * df[f'{prefix}_dx'] + df[f'{prefix}_dy'] * df[f'{prefix}_dy'] + df[f'{prefix}_dz'] * df[f'{prefix}_dz']
                df[f'{prefix}_dist'] = df[f'{prefix}_dist2'].apply(np.sqrt)
        
    def cleanup_columns(self, df):
        columns = list(df.columns)
        to_drop = []
        for column in columns:
            for axis in 'xyz':
                if column.startswith('atom') and column.endswith(f'_{axis}'):
                    to_drop.append(column)

                if column.startswith('atom') and column.endswith(f'_d{axis}'):
                    to_drop.append(column)

                if column.startswith('atom') and column.endswith(f'_{axis}_mean'):
                    to_drop.append(column)

            if column.find('_dir_') >= 0:
                to_drop.append(column)

            # if len(df[column].unique()) == 1 and column != 'type':
            #     to_drop.append(column)

        df.drop(columns=to_drop, inplace=True)


    def make_output(self, output_df):
        return output_df.loc[:, ['scalar_coupling_constant']]

    def fit_scalers(self, numeric_input_df, output_df):
        self.input_scaler.fit(numeric_input_df.values)
        if output_df is not None:
            self.output_scaler.fit(output_df.values)

class SKModel(Model):
    def __init__(self, model_args, fit_args, flatten_output=True):
        Model.__init__(self, **model_args)

        self.fit_args = fit_args

        self.flatten_output = flatten_output

    def fit(self, input_df, output_df, val_input_df=None, val_output_df=None):
        if self.verbose:
            print('  Setting up data')
        input_df, numeric_input_df, output_df = self.setup_data(input_df, output_df)

        if val_input_df is not None and val_output_df is not None:
            val_input_df, val_numeric_input_df, val_output_df = self.setup_data(val_input_df, val_output_df)
            eval_set = (val_numeric_input_df, val_output_df)
        else:
            eval_set = None

        if self.verbose:
            print('  Fitting model')
        ref_output = output_df.values.flatten() if self.flatten_output else output_df.values.reshape((len(output_df), 1))
        #self.model.fit(self.numeric_input_df.values, ref_output)
        self.model.fit(numeric_input_df, output_df, eval_set=eval_set, **self.fit_args)

        self.last_input_df = input_df
        self.last_numeric_input_df = numeric_input_df
        self.last_output_df = output_df

    def corr(self, input_df, output_df):
        input_df, numeric_input_df, output_df = self.setup_data(input_df, output_df)
        return input_df.corr()

    def evaluate(self, input_df, output_df):
        if self.verbose:
            print('  Setting up data')
        input_df, numeric_input_df, output_df = self.setup_data(input_df, output_df)

        if self.verbose:
            print('  Fitting model')
        ref_output = output_df.values.flatten() if self.flatten_output else output_df.values.reshape((len(output_df), 1))
        test_output = self.model.predict(numeric_input_df.values)

        if self.verbose:
            print('  Evaluating model')
        return test_output, score(input_df, ref_output, test_output)

    def predict(self, input_df):
        if self.verbose:
            print('  Setting up data')
        input_df, numeric_input_df, output_df = self.setup_data(input_df)

        if self.verbose:
            print('  Predicting')
        return self.model.predict(numeric_input_df.values)

# class XGBModel(SKModel):
#     def __init__(self, model_args, xgb_args={}):
#         SKModel.__init__(self, **model_args)

#         self.model = xgb.XGBRegressor(**xgb_args)

class LGBModel(SKModel):
    def __init__(self, model_args, lightgbm_model_args={}, lightgbm_fit_args={}):
        SKModel.__init__(self, model_args, lightgbm_fit_args)

        self.model = lgb.LGBMRegressor(**lightgbm_model_args)

    def plot_importance(self, ax=None, height=1):
        lgb.plot_importance(self.model, ax=ax, height=height)

    def plot_split_value_histogram(self, ax=None, height=1):
        lgb.plot_split_value_histogram(self.model, ax=ax)

    def plot_metric(self, ax=None, height=1):
        lgb.plot_metric(self.model, ax=ax)

class RFModel(SKModel):
    def __init__(self, model_args, rf_args={}):
        SKModel.__init__(self, **model_args)

        self.model = RandomForestRegressor(**rf_args)
    

class NNModel(Model):
    def __init__(
        self, model_args, 
        dipole_moments, magnetic_shielding_tensors, mulliken_charges,
        potential_energy, scalar_coupling_contributions, 
        epochs=300, batch_size=512, learning_rate=0.001, validation_split=0.2):

        Model.__init__(self, normalize_input=True, **model_args)

        self.dipole_moments = dipole_moments
        self.magnetic_shielding_tensors = magnetic_shielding_tensors
        self.mulliken_charges = mulliken_charges
        self.potential_energy = potential_energy
        self.scalar_coupling_contributions = scalar_coupling_contributions

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split

        if self.normalize_input:
            self.dipole_moments_scaler = self.scaler()
            self.magnetic_shielding_tensors_scaler = self.scaler()
            self.mulliken_charges_scaler = self.scaler()
            self.potential_energy_scaler = self.scaler()
            self.scalar_coupling_contributions_scaler = self.scaler()

    def setup_additional_output_data(self, input_df):
        self.dipole_moments_output_df = input_df[['molecule_name']].merge(self.dipole_moments, left_on='molecule_name', right_on='molecule_name')
        self.dipole_moments_output_df.drop(columns=['molecule_name'], inplace=True)

        self.magnetic_shielding_tensors_output_df = input_df[['molecule_name', 'atom_index_0', 'atom_index_1']]\
            .merge(self.magnetic_shielding_tensors, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'])\
            .merge(self.magnetic_shielding_tensors, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'], suffixes=('_0', '_1'))
        self.magnetic_shielding_tensors_output_df.drop(columns=['molecule_name', 'atom_index_0', 'atom_index_1'], inplace=True)

        self.mulliken_charges_output_df = input_df[['molecule_name', 'atom_index_0', 'atom_index_1']]\
            .merge(self.mulliken_charges, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'])\
            .merge(self.mulliken_charges, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'], suffixes=('_0', '_1'))
        self.mulliken_charges_output_df.drop(columns=['molecule_name', 'atom_index_0', 'atom_index_1'], inplace=True)
        
        self.potential_energy_output_df = input_df[['molecule_name']].merge(self.potential_energy, left_on='molecule_name', right_on='molecule_name')
        self.potential_energy_output_df.drop(columns=['molecule_name'], inplace=True)
        
        self.scalar_coupling_contributions_output_df = input_df[['molecule_name', 'atom_index_0', 'atom_index_1']]\
            .merge(self.scalar_coupling_contributions, left_on=['molecule_name', 'atom_index_0', 'atom_index_1'], right_on=['molecule_name', 'atom_index_0', 'atom_index_1'])
        self.scalar_coupling_contributions_output_df.drop(columns=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'], inplace=True)


    def fit_scalers(self, numeric_input_df, output_df):
        Model.fit_scalers(self, numeric_input_df, output_df)

        self.dipole_moments_scaler.fit(self.dipole_moments_output_df.values)
        self.magnetic_shielding_tensors_scaler.fit(self.magnetic_shielding_tensors_output_df.values)
        self.mulliken_charges_scaler.fit(self.mulliken_charges_output_df.values)
        self.potential_energy_scaler.fit(self.potential_energy_output_df.values)
        self.scalar_coupling_contributions_scaler.fit(self.scalar_coupling_contributions_output_df.values)

    def fit(self, input_df, output_df):
        print('  Setting up data')
        input_df, numeric_input_df, output_df = self.setup_data(input_df, output_df)
        self.setup_additional_output_data(input_df)

        print('  Creating model')
        self.model = self.create_model(numeric_input_df.values.shape[1])

        print('  Fitting model')
        self.fit_scalers(numeric_input_df, output_df)

        i = self.input_scaler.transform(numeric_input_df.values)
        o = self.output_scaler.transform(output_df.values)


        o_dipole_moments = self.dipole_moments_scaler.transform(self.dipole_moments_output_df.values)
        o_magnetic_shielding_tensors = self.magnetic_shielding_tensors_scaler.transform(self.magnetic_shielding_tensors_output_df.values)
        o_mulliken_charges = self.mulliken_charges_scaler.transform(self.mulliken_charges_output_df.values)
        o_potential_energy = self.potential_energy_scaler.transform(self.potential_energy_output_df.values)
        o_scalar_coupling_contributions = self.scalar_coupling_contributions_scaler.transform(self.scalar_coupling_contributions_output_df.values)


        #es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=8,verbose=1, mode='auto', restore_best_weights=True)
        #rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=7, min_lr=1e-6, mode='auto', verbose=1)
        #callbacks=[es, rlr]
        callbacks=[]

        history = self.model.fit(
            i, 
            #[o, o_dipole_moments, o_magnetic_shielding_tensors, o_mulliken_charges, o_potential_energy, o_scalar_coupling_contributions], 
            [o],
            epochs=self.epochs, batch_size=self.batch_size, callbacks=callbacks, 
            validation_split=self.validation_split,
            shuffle=False, verbose=1)

        return history

    def corr(self, input_df, output_df):
        self.setup_data(input_df, output_df)
        self.setup_additional_output_data()

        return self.input_df.corr()

    def evaluate(self, input_df, output_df):
        print('  Setting up data')
        input_df, numeric_input_df, output_df = self.setup_data(input_df, output_df)
        self.setup_additional_output_data(input_df)

        print('  Fitting model')
        i = self.input_scaler.transform(numeric_input_df.values)
        ref_output = output_df.values
        o = self.model.predict(i)
        test_output = self.output_scaler.inverse_transform(o)

        print('  Evaluating model')
        return test_output, score(input_df, ref_output, test_output)

    def predict(self, input_df):
        print('  Setting up data')
        input_df, numeric_input_df, _ = self.setup_data(input_df)
        self.setup_additional_output_data(input_df)

        print('  Predicting')
        i = self.input_scaler.transform(numeric_input_df.values)
        o = self.model.predict(i)
        return self.output_scaler.inverse_transform(o)

    def create_model_x(self, num_inputs):
        i = Input(shape=(num_inputs,))

        l = i
        for j, n in enumerate([1024] * 5):
            l = self.create_complex_layer(l, n, name=f'common_{j}')

        l_dipole_moments = l
        l_magnetic_shielding_tensors = l
        l_mulliken_charges = l
        l_potential_energy = l
       
        for j, n in enumerate([1024] * 5):
            l_dipole_moments = self.create_complex_layer(l_dipole_moments, n, name=f'dipole_moments_{j}')
            l_magnetic_shielding_tensors = self.create_complex_layer(l_magnetic_shielding_tensors, n, name=f'magnetic_shielding_tensors_{j}')
            l_mulliken_charges = self.create_complex_layer(l_mulliken_charges, n, name=f'mulliken_charges_{j}')
            l_potential_energy = self.create_complex_layer(l_potential_energy, n, name=f'potential_energy_{j}')

        for j, n in enumerate([1024] * 5):
            l = self.create_complex_layer(l, n, name=f'scc_1_{j}')

        l_scalar_coupling_contributions = l

        for j, n in enumerate([1024] * 1):
            l = self.create_complex_layer(l, n, name=f'scc_2_{j}')
            l_scalar_coupling_contributions = self.create_complex_layer(l_scalar_coupling_contributions, n, name=f'scalar_coupling_contributions_{j}')

        o_dipole_moments = Dense(len(self.dipole_moments_output_df.columns), activation='linear', name='output_dipole_moments')(l_dipole_moments)
        o_magnetic_shielding_tensors = Dense(len(self.magnetic_shielding_tensors_output_df.columns), activation='linear', name='output_magnetic_shielding_tensors')(l_magnetic_shielding_tensors)
        o_mulliken_charges = Dense(len(self.mulliken_charges_output_df.columns), activation='linear', name='output_mulliken_charges')(l_mulliken_charges)
        o_potential_energy = Dense(len(self.potential_energy_output_df.columns), activation='linear', name='output_potential_energy')(l_potential_energy)
        o_scalar_coupling_contributions = Dense(len(self.scalar_coupling_contributions_output_df.columns), activation='linear', name='output_scalar_coupling_contributions')(l_scalar_coupling_contributions)

        o = Dense(1, activation='linear', name='scc')(l)

        model = KerasModel(inputs=[i], outputs=[o, o_dipole_moments, o_magnetic_shielding_tensors, o_mulliken_charges, o_potential_energy, o_scalar_coupling_contributions])

        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mae')

        #model.summary()

        return model

    def create_model(self, num_inputs):
        # 0.56 for 1JHC
        i = l = Input(shape=(num_inputs,))

        for j, n in enumerate([1024] * 11):
            l = self.create_complex_layer(l, n, name=f'1024x11_{j}')
            n >>= 1

        o = Dense(1, activation='linear', name='scc')(l)

        model = KerasModel(inputs=[i], outputs=[o])
        model.compile(optimizer=Adam(), loss='mae')

        #model.summary()

        return model

    def create_complex_layer(self, l, n, name=''):
        # l = Dense(n, kernel_initializer='normal', name=f'{name}_dense')(l)
        # l = BatchNormalization(name=f'{name}_batch_norm')(l)
        # l = LeakyReLU(alpha=0.05, name=f'{name}_leaky_relu')(l)
        # l = Dropout(0.2, name=f'{name}_dropout')(l)

        l = Dense(n, kernel_initializer='normal', name=f'{name}_dense')(l)
        l = LeakyReLU(alpha=0, name=f'{name}_leaky_relu')(l)
        l = BatchNormalization(name=f'{name}_batch_norm')(l)
        l = Dropout(0.25, name=f'{name}_dropout')(l)
        return l

