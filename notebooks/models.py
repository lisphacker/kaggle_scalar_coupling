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

class DummyScaler:
    def fit(self, X):
        return X

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X

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

class MM11Scaler(MinMaxScaler):
    def __init__(self):
        MinMaxScaler.__init__(self, feature_range=(-1, 1))

class Model:
    def __init__(self, molecules, structures, normalize_input=False):
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

    def setup_data(self, input_df, output_df=None):
        self.coupling_types = set(input_df.type.unique())
        if output_df is not None:
            self.coupling_types = self.coupling_types | set(output_df.type.unique())
        self.coupling_type_index = {t:i for i, t in enumerate(self.coupling_types)}

        self.input_df = self.make_input(input_df)
        self.make_complex_inputs(self.input_df)
        self.cleanup_columns(self.input_df)

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

    def make_complex_inputs(self, df):
        return
        for atomId in "0N":
            prefix = f'atom{atomId}'
            for axis in "xyz":
                df[f'{prefix}_dir_{axis}'] = (df[f'{prefix}_{axis}'] - df[f'{prefix}_{axis}_mean']) / df[f'{prefix}_dist_to_mean']

        df['cos0N'] = df['atom0_dir_x'] * df['atomN_dir_x'] + df['atom0_dir_y'] * df['atomN_dir_y'] + df['atom0_dir_z'] * df['atomN_dir_z']
        
        pprint(list(df.columns))

    def cleanup_columns(self, df):
        pass

    def make_output(self, output_df):
        return output_df.loc[:, ['scalar_coupling_constant']]

    def fit_scalers(self):
        self.input_scaler.fit(self.numeric_input_df.values)
        if self.output_df is not None:
            self.output_scaler.fit(self.output_df.values)

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

# class XGBModel(SKModel):
#     def __init__(self, model_args, xgb_args={}):
#         SKModel.__init__(self, **model_args)

#         self.model = xgb.XGBRegressor(**xgb_args)

class LGBModel(SKModel):
    def __init__(self, model_args, lightgbm_args={}):
        SKModel.__init__(self, **model_args)

        self.model = lgb.LGBMRegressor(**lightgbm_args)

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

    def setup_additional_output_data(self):
        self.dipole_moments_output_df = self.input_df[['molecule_name']].merge(self.dipole_moments, left_on='molecule_name', right_on='molecule_name')
        self.dipole_moments_output_df.drop(columns=['molecule_name'], inplace=True)

        self.magnetic_shielding_tensors_output_df = self.input_df[['molecule_name', 'atom_index_0', 'atom_index_1']]\
            .merge(self.magnetic_shielding_tensors, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'])\
            .merge(self.magnetic_shielding_tensors, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'], suffixes=('_0', '_1'))
        self.magnetic_shielding_tensors_output_df.drop(columns=['molecule_name', 'atom_index_0', 'atom_index_1'], inplace=True)

        self.mulliken_charges_output_df = self.input_df[['molecule_name', 'atom_index_0', 'atom_index_1']]\
            .merge(self.mulliken_charges, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'])\
            .merge(self.mulliken_charges, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'], suffixes=('_0', '_1'))
        self.mulliken_charges_output_df.drop(columns=['molecule_name', 'atom_index_0', 'atom_index_1'], inplace=True)
        
        self.potential_energy_output_df = self.input_df[['molecule_name']].merge(self.potential_energy, left_on='molecule_name', right_on='molecule_name')
        self.potential_energy_output_df.drop(columns=['molecule_name'], inplace=True)
        
        self.scalar_coupling_contributions_output_df = self.input_df[['molecule_name', 'atom_index_0', 'atom_index_1']]\
            .merge(self.scalar_coupling_contributions, left_on=['molecule_name', 'atom_index_0', 'atom_index_1'], right_on=['molecule_name', 'atom_index_0', 'atom_index_1'])
        self.scalar_coupling_contributions_output_df.drop(columns=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'], inplace=True)


    def fit_scalers(self):
        Model.fit_scalers(self)

        self.dipole_moments_scaler.fit(self.dipole_moments_output_df.values)
        self.magnetic_shielding_tensors_scaler.fit(self.magnetic_shielding_tensors_output_df.values)
        self.mulliken_charges_scaler.fit(self.mulliken_charges_output_df.values)
        self.potential_energy_scaler.fit(self.potential_energy_output_df.values)
        self.scalar_coupling_contributions_scaler.fit(self.scalar_coupling_contributions_output_df.values)

    def fit(self, input_df, output_df):
        self.setup_data(input_df, output_df)
        self.setup_additional_output_data()

        self.model = self.create_model(self.numeric_input_df.values.shape[1])

        self.fit_scalers()

        i = self.input_scaler.transform(self.numeric_input_df.values)
        o = self.output_scaler.transform(self.output_df.values)


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
            [o, o_dipole_moments, o_magnetic_shielding_tensors, o_mulliken_charges, o_potential_energy, o_scalar_coupling_contributions], 
            epochs=self.epochs, batch_size=self.batch_size, callbacks=callbacks, validation_split=self.validation_split, verbose=1)

        return history

    def corr(self, input_df, output_df):
        self.setup_data(input_df, output_df)
        self.setup_additional_output_data()

        return self.input_df.corr()

    def evaluate(self, input_df, output_df):
        self.setup_data(input_df, output_df)
        self.setup_additional_output_data()

        i = self.input_scaler.transform(self.numeric_input_df.values)
        ref_output = self.output_df.values
        o = self.model.predict(i)
        test_output = self.output_scaler.inverse_transform(o[0])

        return test_output, score(output_df, ref_output, test_output)

    def predict(self, input_df):
        self.setup_data(input_df)

        i = self.input_scaler.transform(self.numeric_input_df.values)
        o = self.model.predict(i)
        return self.output_scaler.inverse_transform(o[0])

    def create_model(self, num_inputs):
        # 0.56 for 1JHC
        i = Input(shape=(num_inputs,))

        l = i
        for j, n in enumerate([4096] * 6):
            l = self.create_complex_layer(l, n, name=f'1024x15_{j}')

        l_dipole_moments = l
        l_magnetic_shielding_tensors = l
        l_mulliken_charges = l
        l_potential_energy = l
        l_scalar_coupling_contributions = l
       
        for j, n in enumerate([1024] * 1):
            l_dipole_moments = self.create_complex_layer(l_dipole_moments, n, name=f'dipole_moments_{j}')
            l_magnetic_shielding_tensors = self.create_complex_layer(l_magnetic_shielding_tensors, n, name=f'magnetic_shielding_tensors_{j}')
            l_mulliken_charges = self.create_complex_layer(l_mulliken_charges, n, name=f'mulliken_charges_{j}')
            l_potential_energy = self.create_complex_layer(l_potential_energy, n, name=f'potential_energy_{j}')
            l_scalar_coupling_contributions = self.create_complex_layer(l_scalar_coupling_contributions, n, name=f'scalar_coupling_contributions_{j}')

            l = self.create_complex_layer(l, n, name=f'output_{j}')

        o_dipole_moments = Dense(len(self.dipole_moments_output_df.columns), activation='linear', name='output_dipole_moments')(l_dipole_moments)
        o_magnetic_shielding_tensors = Dense(len(self.magnetic_shielding_tensors_output_df.columns), activation='linear', name='output_magnetic_shielding_tensors')(l_magnetic_shielding_tensors)
        o_mulliken_charges = Dense(len(self.mulliken_charges_output_df.columns), activation='linear', name='output_mulliken_charges')(l_mulliken_charges)
        o_potential_energy = Dense(len(self.potential_energy_output_df.columns), activation='linear', name='output_potential_energy')(l_potential_energy)
        o_scalar_coupling_contributions = Dense(len(self.scalar_coupling_contributions_output_df.columns), activation='linear', name='output_scalar_coupling_contributions')(l_scalar_coupling_contributions)

        o = Dense(1, activation='linear', name='merged')(l)

        model = KerasModel(inputs=[i], outputs=[o, o_dipole_moments, o_magnetic_shielding_tensors, o_mulliken_charges, o_potential_energy, o_scalar_coupling_contributions])

        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mae')

        #model.summary()

        return model

    def create_model_best(self, num_inputs):
        # 0.56 for 1JHC
        i = l = Input(shape=(num_inputs,))

        for j, n in enumerate([1024] * 11):
            l = self.create_complex_layer(l, n, name=f'1024x11_{j}')
            n >>= 1

        o = Dense(1, activation='linear')(l)

        model = KerasModel(inputs=[i], outputs=[o])
        model.compile(optimizer=Adam(), loss='mae')

        #model.summary()

        return model

    def create_complex_layer(self, l, n, name=''):
        l = Dense(n, kernel_initializer='normal', name=f'{name}_dense')(l)
        l = BatchNormalization(name=f'{name}_batch_norm')(l)
        # l = LeakyReLU(alpha=0.2)(l)
        # l = Dropout(0.1)(l)
        l = LeakyReLU(alpha=0.05, name=f'{name}_leaky_relu')(l)
        l = Dropout(0.2, name=f'{name}_dropout')(l)
        return l

