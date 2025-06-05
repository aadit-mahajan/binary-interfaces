# 
from tqdm import tqdm
from biopandas.pdb import PandasPdb as pandaspdb
import numpy as np
import pandas as pd 
import os 
import matplotlib.pyplot as plt
import json

def generate_z_planes(interchain_intr, N_PLANES) -> list[float]:
    z_values = []
    for i in range(len(interchain_intr)):
        z_values.append(interchain_intr.atom1_zcoord.iloc[i])
        z_values.append(interchain_intr.atom2_zcoord.iloc[i])

    z_values = np.array(z_values)
    buffer = 1e-3
    z_planes = list(np.linspace(z_values.min() - buffer, z_values.max() + buffer, N_PLANES))
    z_planes.sort()
    return z_planes

def get_min_g_val_se(master_matrix, g_vecs):

    struct_group = master_matrix.groupby('struct_elem')
    min_g_vals = g_vecs.min(axis='index')

    # for each temperature, get the no of structured elements with minimum free energy values
    min_g_vals_struct_elems = []
    for temp in g_vecs.columns:
        min_g = min_g_vals[temp]
        min_g_vals_struct_elems += (g_vecs[temp].index[g_vecs[temp] == min_g].tolist())

    min_g_vals_struct_elems = set(min_g_vals_struct_elems)
    min_g_vals_groups = struct_group.get_group(min_g_vals_struct_elems.pop())
    for elem in min_g_vals_struct_elems:
        min_g_vals_groups = pd.concat([min_g_vals_groups, struct_group.get_group(elem)])

    return min_g_vals_struct_elems, min_g_vals_groups

def get_struct_elem_list(master_matrix) -> list:
    struct_elem_list = []
    for i in range(len(master_matrix)):
        start1 = int(master_matrix.iloc[i].start1)
        end1 = int(master_matrix.iloc[i].end1)
        start2 = int(master_matrix.iloc[i].start2)
        end2 = int(master_matrix.iloc[i].end2)
        comb_type = master_matrix.iloc[i].type
        if comb_type == 1:
            elems_in_comb = list(range(start1, end1))
        elif comb_type == 2:
            elems_in_comb = list(range(start1, end1)) + list(range(start2, end2))
        struct_elem_list.append(elems_in_comb)

    return struct_elem_list

def load_pdb_list(pdb_dir='./pdb_files'):
    pdb_list = os.listdir(pdb_dir)
    pdb_list = [i.split('.')[0] for i in pdb_list]
    return pdb_list

def plot_min_g_vals_ms_hist(min_g_vals_groups, pdb_id, output_dir = './plots/min_gvals_ms'):

    os.makedirs(output_dir, exist_ok=True)

    total_entries = min_g_vals_groups.reset_index()
    mean_g_part_total = total_entries.g_part.mean()
    std_g_part_total = total_entries.g_part.std()
    plt.hist(total_entries.g_part, bins=50)
    plt.axvline(mean_g_part_total, color='r', linestyle='dashed', linewidth=1)
    plt.fill_betweenx([-2, 150], mean_g_part_total - std_g_part_total, mean_g_part_total + std_g_part_total, color='r', alpha=0.2)
    plt.grid()
    plt.ylim(-2, 150)
    plt.xlabel('free energy (J/mol)')
    plt.ylabel('frequency')
    plt.title(f'g_part for all microstates in min_g_vals_groups for {pdb_id}')
    plt.savefig(os.path.join(output_dir, f'{pdb_id}_ms_hist.png'))
    plt.close()

def get_se_freq_stable(min_g_vals_groups):
    '''
    se_freq_stable_df -> ms with highest frequency 
    se_freq_stable_cond_df -> ms with energy < (mean-std)
    '''

    total_entries = min_g_vals_groups.reset_index()
    mean_g_part_total = total_entries.g_part.mean()
    std_g_part_total = total_entries.g_part.std()
    hist = np.histogram(total_entries.g_part, bins = 50)
    max_freq = max(hist[0])
    max_freq_idx = np.where(hist[0] == max_freq)[0][0]

    # boundaries of the bins with the highest frequency
    bin_start = hist[1][max_freq_idx]
    bin_end = hist[1][max_freq_idx+1]

    # microstates which fall within the boundaries of the bins with the highest frequency
    stable_struct_elems_freq = total_entries[(total_entries.g_part >= bin_start) & (total_entries.g_part <= bin_end)]
    stable_struct_elems_cond = total_entries[(total_entries.g_part < (mean_g_part_total - std_g_part_total))]

    stable_struct_elems_list = get_struct_elem_list(stable_struct_elems_freq)
    stable_struct_elems_list_cond = get_struct_elem_list(stable_struct_elems_cond)

    # print('stable structured elements:', stable_struct_elems_list)
    # print(len(stable_struct_elems_list))
    se_freq_stable = {}
    se_freq_stable_cond = {}

    for i in range(len(stable_struct_elems_list)):
        temp = stable_struct_elems_list[i]
        se_list = np.zeros(N_PLANES+1)
        for elem in temp:
            se_list[elem] += 1
        se_freq_stable[i] = se_list

    for i in range(len(stable_struct_elems_list_cond)):
        temp = stable_struct_elems_list_cond[i]
        se_list = np.zeros(N_PLANES+1)
        for elem in temp:
            se_list[elem] += 1
        se_freq_stable_cond[i] = se_list

    se_freq_stable_df = pd.DataFrame(se_freq_stable)
    se_freq_stable_cond_df = pd.DataFrame(se_freq_stable_cond)

    return se_freq_stable_df, se_freq_stable_cond_df

def mod_bfac_to_ms_freq(se_freq_stable_df, se_freq_stable_cond_df, pdb_file_dir, id, z_planes, output_dir='./plots/se_freq_stable'):
    os.makedirs(output_dir, exist_ok=True)
    counts = se_freq_stable_df.sum(axis=1)
    counts_cond = se_freq_stable_cond_df.sum(axis=1)

    plt.plot(counts)
    plt.plot(counts_cond)
    plt.legend(['stable structured elements', 'stable structured elements with energy < (mean-std)'])
    plt.title(f'Frequency of stable structured elements for {id}')
    plt.xlabel('Z-axis')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, f'{id}_se_freq_stable.png'))
    plt.close()

    elec_intr = pd.read_csv(f'./elec_intr_files/elec_intr_{id}.csv')
    vdw_intr = pd.read_csv(f'./vdw_intr_files/vdw_intr_{id}.csv')
    residue_list = list(set(elec_intr.atom1_resnum.tolist() + elec_intr.atom2_resnum.tolist() + 
                           vdw_intr.atom1_resnum.tolist() + vdw_intr.atom2_resnum.tolist()))
    print('residue list:', residue_list)
    
    pdb = pandaspdb()
    pdb.read_pdb(f'{pdb_file_dir}/{id}.pdb')

    pdb_df = pdb.df['ATOM']
    
    # Create a dictionary to map residue numbers to their b-factor values
    residue_to_bfactor = {}
    
    # Process only residues in the residue list
    for i in range(len(pdb_df)):
        if pdb_df.iloc[i].residue_number in residue_list:
            z = pdb_df.z_coord.iloc[i]
            section = np.digitize(z, z_planes)
            
            # Get the count for this section
            count_value = counts[section]
            
            # Store the section's count value for this residue
            if pdb_df.iloc[i].residue_number not in residue_to_bfactor:
                residue_to_bfactor[pdb_df.iloc[i].residue_number] = count_value
    
    # Scale the b-factor values to the range of 0-99
    if residue_to_bfactor:  # Check if dictionary is not empty
        min_val = min(residue_to_bfactor.values())
        max_val = max(residue_to_bfactor.values())
        
        # Avoid division by zero if all values are the same
        if max_val != min_val:
            for res_num in residue_to_bfactor:
                residue_to_bfactor[res_num] = 100 - int(((residue_to_bfactor[res_num] - min_val) / 
                                                 (max_val - min_val)) * 99)
        else:
            # If all values are the same, set them to a middle value like 50
            for res_num in residue_to_bfactor:
                residue_to_bfactor[res_num] = 50
    
    # Assign b-factor values only to residues in the residue list
    for i in range(len(pdb_df)):
        res_num = pdb_df.iloc[i].residue_number
        if res_num in residue_to_bfactor:
            pdb_df.at[i, 'b_factor'] = residue_to_bfactor[res_num]
        else:
            # color the two chains differently
            if pdb_df.iloc[i].chain_id == 'A':
                pdb_df.at[i, 'b_factor'] = 40
            elif pdb_df.iloc[i].chain_id == 'B':
                pdb_df.at[i, 'b_factor'] = 60
    
    pdb.to_pdb(path=f'../mod_pdb_files/{id}_se_freq_stable.pdb', records=['ATOM'])

def mod_bfac_to_blockwise_FE(id, pdb_file_dir, blockwise_g_vecs_dir):
    output_dir = '../mod_pdb_files'
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(blockwise_g_vecs_dir, f'blockwise_g_vals_{id}.csv')
    blockwise_g_vecs = pd.read_csv(file_path)
    block_g_vals = blockwise_g_vecs.iloc[:, 1].values
    block_g_vals = block_g_vals.astype(float)

    elec_intr = pd.read_csv(f'./elec_intr_files/elec_intr_{id}.csv')
    vdw_intr = pd.read_csv(f'./vdw_intr_files/vdw_intr_{id}.csv')
    residue_list = list(set(elec_intr.atom1_resnum.tolist() + elec_intr.atom2_resnum.tolist() + 
                           vdw_intr.atom1_resnum.tolist() + vdw_intr.atom2_resnum.tolist()))
    print('residue list:', residue_list)
    
    pdb = pandaspdb()
    pdb.read_pdb(f'{pdb_file_dir}/{id}.pdb')

    pdb_df = pdb.df['ATOM']
    
    # Create a dictionary to map residue numbers to their b-factor values
    residue_to_bfactor = {}
    
    # Process only residues in the residue list
    for i in range(len(pdb_df)):
        if pdb_df.iloc[i].residue_number in residue_list:
            z = pdb_df.z_coord.iloc[i]
            section = np.digitize(z, z_planes)
            
            # Get the count for this section
            count_value = block_g_vals[section]
            
            # Store the section's count value for this residue
            if pdb_df.iloc[i].residue_number not in residue_to_bfactor:
                residue_to_bfactor[pdb_df.iloc[i].residue_number] = count_value
    
    # Scale the b-factor values to the range of 50-99
    if residue_to_bfactor:  # Check if dictionary is not empty
        min_val = min(residue_to_bfactor.values())
        max_val = max(residue_to_bfactor.values())
        
        # Avoid division by zero if all values are the same
        if max_val != min_val:
            for res_num in residue_to_bfactor:
                residue_to_bfactor[res_num] = int(((residue_to_bfactor[res_num] - min_val) / 
                                                 (max_val - min_val)) * 99)
        else:
            # If all values are the same, set them to a middle value like 50
            for res_num in residue_to_bfactor:
                residue_to_bfactor[res_num] = 50
    
    # Assign b-factor values only to residues in the residue list
    for i in range(len(pdb_df)):
        res_num = pdb_df.iloc[i].residue_number
        if res_num in residue_to_bfactor:
            pdb_df.at[i, 'b_factor'] = residue_to_bfactor[res_num]
        else:
            # color the two chains differently
            if pdb_df.iloc[i].chain_id == 'A':
                pdb_df.at[i, 'b_factor'] = 40
            elif pdb_df.iloc[i].chain_id == 'B':
                pdb_df.at[i, 'b_factor'] = 60
    
    pdb.to_pdb(path=f'../mod_pdb_files/{id}_block_fe.pdb', records=['ATOM'])


if __name__ == '__main__':

    with open('params.json') as f:
        config = json.loads(f.read())

    N_PLANES = config["N_PLANES"]
    R = config["R"]
    pdb_list = load_pdb_list()

    master_matrix_dir = './master_matrices'
    g_vecs_dir = './g_vecs'
    pdb_file_dir = './pdb_files'
    interchain_intr_dir = './elec_intr_files'
    blockwise_g_vecs_dir = './analysis_output/blockwise_g_vals'
    vdw_intr_dir = './vdw_intr_files'

    for pdb_file in tqdm(os.listdir(pdb_file_dir), desc='Processing PDB files'):
        pdb = pdb_file.split('.')[0]

        try:
            interchain_intr = pd.read_csv(os.path.join(interchain_intr_dir, f'elec_intr_{pdb}.csv'))
            vdw_intr = pd.read_csv(os.path.join(vdw_intr_dir, f'vdw_intr_{pdb}.csv'))

            master_matrix_path = os.path.join(master_matrix_dir, f'{pdb}_master_matrix.csv')
            g_vecs_path = os.path.join(g_vecs_dir, f'{pdb}_g_vecs.csv')

            z_planes = generate_z_planes(interchain_intr, N_PLANES)
            master_matrix = pd.read_csv(master_matrix_path)
            g_vecs = pd.read_csv(g_vecs_path)
            min_g_val_struct_elems, min_g_vals_groups = get_min_g_val_se(master_matrix, g_vecs)
            min_g_val_group_z_total = min_g_vals_groups.sw_part.sum()

            min_g_vals_groups = min_g_vals_groups.copy()
            min_g_vals_groups.loc[:, 'partial_z'] = min_g_vals_groups['sw_part'] / min_g_val_group_z_total
            min_g_vals_groups.loc[:, 'g_part'] = min_g_vals_groups['partial_z'].apply(lambda x: -R * 298 * np.log(x) if x > 0 else np.nan)

            struct_elem_list = get_struct_elem_list(min_g_vals_groups)

            se_freq_stable_df, se_freq_stable_cond_df = get_se_freq_stable(min_g_vals_groups)
            mod_bfac_to_ms_freq(se_freq_stable_df=se_freq_stable_df, 
                                se_freq_stable_cond_df=se_freq_stable_cond_df, 
                                pdb_file_dir=pdb_file_dir, 
                                id=pdb, 
                                z_planes=z_planes)
            
            mod_bfac_to_blockwise_FE(id=pdb, 
                                   pdb_file_dir=pdb_file_dir, 
                                   blockwise_g_vecs_dir=blockwise_g_vecs_dir)
            
            plot_min_g_vals_ms_hist(min_g_vals_groups, pdb)

            # print(f'{pdb} done')
        except Exception as e:
            print(f'Error in {pdb}: {e}')
            continue

    print('All done')






        
        





        

