import pandas as pd
import requests as req
import numpy as np
from scipy.spatial.distance import cdist
import biopandas.pdb as PandasPDB
import os

def get_pdb(pdb_id):
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    r = req.get(url)
    with open(f'{pdb_id}.pdb', 'wb') as f:
        f.write(r.content)

def get_elec_int(pdb_path):
    
    # for all possible electrostatic interactions
    # hetatm are ignored by defining the atoms that can form electrostatic interactions
    '''
    This block of code makes a list of all possible electrostatic interactions between the two chains
    returns a list of lists with the following format:
    [atom1, chain1, atom1_coord, atom1_chg, atom2, chain2, atom2_coord, atom2_chg, distance]
    '''
    pandas_pdb = PandasPDB.PandasPdb()

    # get only the first model in the pdb file

    pandas_pdb = pandas_pdb.read_pdb(pdb_path)
    try:
        pandas_pdb = pandas_pdb.get_model(1)
    except:
        pass

    elec_int_all = []
    dist_cutoff = 12

    charged_res = ['ARG', 'LYS', 'HIS', 'ASP', 'GLU']
    charges = {
        'NH1':0.5, 
        'NH2':0.5,
        'NZ':1,
        'ND1':0.5,
        'NE2':0.5,
        'OD1':-0.5,
        'OD2':-0.5,
        'OE1':-0.5,
        'OE2':-0.5,
    }

    charged_atoms = pandas_pdb.df['ATOM'][pandas_pdb.df['ATOM']['residue_name'].isin(charged_res)]
    charged_atoms = charged_atoms[charged_atoms['atom_name'].isin(charges.keys())]
    charged_atoms = charged_atoms.reset_index(drop=True)

    charged_atoms['charge'] = charged_atoms['atom_name'].apply(lambda x: charges[x])

    chainA = charged_atoms[charged_atoms['chain_id'] == 'A']
    chainB = charged_atoms[charged_atoms['chain_id'] == 'B']

    dists = cdist(chainA[['x_coord', 'y_coord', 'z_coord']], chainB[['x_coord', 'y_coord', 'z_coord']])
    close_pairs = np.where(dists < dist_cutoff)
    shift = len(chainA)

    for i, j in zip(*close_pairs):
        atom1 = chainA[chainA.index == i].squeeze()
        atom2 = chainB[chainB.index == j+shift].squeeze()

        distance = dists[i, j]
        elec_int_all.append([
                            atom1['atom_number'], atom1['residue_name'], atom1['chain_id'], atom1['x_coord'], atom1['y_coord'], atom1['z_coord'], atom1['charge'], atom1['residue_number'],
                            atom2['atom_number'], atom2['residue_name'], atom2['chain_id'], atom2['x_coord'], atom2['y_coord'], atom2['z_coord'], atom2['charge'], atom2['residue_number'],
                            distance])
            
    return elec_int_all

def get_vdw_int(pdb_path):
    # vdw interactions between the two chains
    '''
    This block of code makes a list of all possible Van der Waals interactions between the two chains
    returns a list of lists with the following format:
    [atom1, chain1, atom1_coord, atom1_resnum, atom2, chain2, atom2_coord, atom2_resnum, distance]
    '''

    pandas_pdb = PandasPDB.PandasPdb()
    pandas_pdb = pandas_pdb.read_pdb(pdb_path)

    try:
        pandas_pdb = pandas_pdb.get_model(1)
    except:
        pass
    
    vdw_int = []
    dist_cutoff = 4

    heavy_atoms = ['C', 'N', 'O', 'S']

    heavy_atoms = pandas_pdb.df['ATOM'][pandas_pdb.df['ATOM']['element_symbol'].isin(heavy_atoms)]
    heavy_atoms = heavy_atoms.reset_index(drop=True)

    chainA = heavy_atoms[heavy_atoms['chain_id'] == 'A']
    chainB = heavy_atoms[heavy_atoms['chain_id'] == 'B']

    dists = cdist(chainA[['x_coord', 'y_coord', 'z_coord']], chainB[['x_coord', 'y_coord', 'z_coord']])
    close_pairs = np.where(dists < dist_cutoff)
    shift = len(chainA)

    for i, j in zip(*close_pairs):
        atom1 = chainA[chainA.index == i].squeeze()
        atom2 = chainB[chainB.index == j+shift].squeeze()

        distance = dists[i, j]
        vdw_int.append([
                        atom1['atom_number'], atom1['residue_name'], atom1['chain_id'], atom1['x_coord'], atom1['y_coord'], atom1['z_coord'], atom1['residue_number'],
                        atom2['atom_number'], atom2['residue_name'], atom2['chain_id'], atom2['x_coord'], atom2['y_coord'], atom2['z_coord'], atom2['residue_number'],
                        distance])

    return vdw_int

if __name__ == '__main__':
    import os

pdb_files_dir = './pdb_files'
print(os.getcwd())
print(os.listdir(pdb_files_dir))

elec_files_dir = './elec_intr_files'
vdw_files_dir = './vdw_intr_files'

os.makedirs(elec_files_dir, exist_ok=True)
os.makedirs(vdw_files_dir, exist_ok=True)


for file in os.listdir(pdb_files_dir):
    if file.endswith('.pdb'):
        pdb_path = os.path.join(pdb_files_dir, file)
        id = file.split('.')[0]
        print(f'Working on pdb: {id}')

        # electrostatic interactions
        elec_int_all = get_elec_int(pdb_path)
        elec_int_df = pd.DataFrame(elec_int_all, columns=['atom1', 'res1', 'chain1', 'atom1_xcoord', 'atom1_ycoord', 'atom1_zcoord', 'atom1_chg', 'atom1_resnum',
                                                    'atom2', 'res2', 'chain2', 'atom2_xcoord', 'atom2_ycoord', 'atom2_zcoord', 'atom2_chg', 'atom2_resnum', 
                                                    'distance'])
        elec_int_df['intene'] = 332*4.184*elec_int_df['atom1_chg']*elec_int_df['atom2_chg']/(29*elec_int_df['distance'])

        elec_int_path = os.path.join(elec_files_dir, f'elec_intr_{id}.csv')
        elec_int_df.to_csv(elec_int_path, index=False)

        print('Electrostatic interactions done.')

        # Van Der Waals interactions
        vdw = get_vdw_int(pdb_path)
        vdw_intr = pd.DataFrame(vdw, columns=['atom1', 'res1', 'chain1', 'atom1_xcoord', 'atom1_ycoord', 'atom1_zcoord', 'atom1_resnum',
                                            'atom2', 'res2', 'chain2', 'atom2_xcoord', 'atom2_ycoord', 'atom2_zcoord', 'atom2_resnum',
                                            'distance'])
        
        vdw_intr_path = os.path.join(vdw_files_dir, f'vdw_intr_{id}.csv')
        vdw_intr.to_csv(vdw_intr_path, index=False)
        print('Van der Waals interactions done.')
        print(f'Done with pdb: {id}')
        print('-------------------------------------------')
print('All done!')

