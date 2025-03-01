import numpy as np
import pandas as pd
import os
from free_energy import generate_z_planes
import json
import matplotlib.pyplot as plt
from collections import defaultdict

def load_params(param_file_path):
    with open(param_file_path, 'r') as f:
        params = json.load(f)
    return params

def get_intr_sections(interactions, section_boundaries) -> pd.DataFrame:
    
    # generate partitions
    part_1 = pd.cut(interactions['atom1_zcoord'], bins=section_boundaries, labels=False)
    part_2 = pd.cut(interactions['atom2_zcoord'], bins=section_boundaries, labels=False)

    # handle nan values 
    part_1 = part_1.fillna(-1)
    part_2 = part_2.fillna(-1)

    # assign section number to the atoms if both atoms are in the same section
    for i in range(len(interactions)):
        p1 = int(part_1.iloc[i])
        p2 = int(part_2.iloc[i])

        # if p1 == p2, assign the section number to both atoms, else push to section of atom2
        if p1 == p2:
            interactions.at[i, 'section'] = p1
        else:
            interactions.at[i, 'section'] = p2
    
    return interactions

def calc_min_g_struct_elem(g_vecs):

    # get the structured elements with minimum free energy values
    min_g_vals = g_vecs.min(axis='index')
    # print(min_g_vals)

    # for each temperature, get the no of structured elements with minimum free energy values
    min_g_vals_struct_elems = []
    for temp in g_vecs.columns:
        min_g = min_g_vals[temp]
        min_g_vals_struct_elems += (g_vecs[temp].index[g_vecs[temp] == min_g].tolist())

    min_g_vals_struct_elems = list(set(min_g_vals_struct_elems))
    return min_g_vals_struct_elems

def get_min_g_val_groups(master_matrix, min_g_vals_struct_elem):
    struct_group = master_matrix.groupby('struct_elem')
    return struct_group.get_group(min_g_vals_struct_elem)

def frequency_dict(min_g_vals_group, z_planes):
    '''
    Finding the occurrence of each interacting block in the groups with minimum free energy values
    This will give a good idea of the favourably interacting blocks. 
    '''
    
    int_blocks = []
    for row in min_g_vals_group.iterrows():
        row = row[1].astype(int)
        seq1 = list(range(row.start1, row.end1+1))
        seq2 = list(range(row.start2, row.end2+1))
        if len(seq1) != (row.struct_elem +1) and len(seq2) != (row.struct_elem + 1):
            int_blocks += [seq1, seq2]
        else:
            if len(seq1) == row.struct_elem + 1:
                int_blocks += [seq1]
            else:
                int_blocks += [seq2]
            # print(int_blocks)

    freq = np.zeros(len(z_planes)+1).astype(int)

    for block in int_blocks:
        for elem in block:
            freq[elem] += 1

    freq_dict = {}
    for i in range(len(freq)):
        freq_dict[i] = freq[i]

    freq_dict = dict(sorted(freq_dict.items(), key=lambda x: x[1], reverse=True))
    return freq_dict

def max_freq_residues(freq_dict, elec_intr):
    '''
    extracts the residues with the highest frequency of interaction from each chain in the binary complex. 
    '''
    max_freq_residues_ch1 = []
    max_freq_residues_ch2 = []

    for elem, _ in freq_dict:
        block1 = (elec_intr.loc[elec_intr.section == elem]).atom1_resnum
        block2 = (elec_intr.loc[elec_intr.section == elem]).atom2_resnum
        if block1.empty or block2.empty:
            continue
        max_freq_residues_ch1 += list(block1)
        max_freq_residues_ch2 += list(block2)

    return max_freq_residues_ch1, max_freq_residues_ch2

def count_res_freq(chain1_max_freq, chain2_max_freq):
    res_freq1 = {}
    for res in chain1_max_freq:
        if res in res_freq1:
            res_freq1[res] += 1
        else:
            res_freq1[res] = 1

    res_freq2 = {}
    for res in chain2_max_freq:
        if res in res_freq2:
            res_freq2[res] += 1
        else:
            res_freq2[res] = 1

    return res_freq1, res_freq2

def calculate_partial_deltaG(master_matrix, min_energy_struct_elem, temp) -> pd.DataFrame:
    '''
    Use this only for generating deltaG values for the minimum energy group 
    '''
    struct_group = master_matrix.groupby('struct_elem')
    min_g_vals_groups = struct_group.get_group(min_energy_struct_elem)

    z_total = min_g_vals_groups.sw_part.sum()
    min_g_vals_groups = min_g_vals_groups.copy()
    min_g_vals_groups['partial_z'] = min_g_vals_groups.sw_part / z_total
    min_g_vals_groups['g_part'] = -R * temp * np.log(min_g_vals_groups.partial_z)

    return min_g_vals_groups

def get_block_g_vals(id, min_g_vals_struct_elem, output_dir):
    master_matrix = pd.read_csv(f'{master_matrices_dir}/{id}_master_matrix.csv')
    min_FE_macrostate = min_g_vals_struct_elem
    min_FE_group = master_matrix[master_matrix['struct_elem'] == min_FE_macrostate]

    TEMP = 300
    R = 8.314/1000
    
    blockwise_sw = defaultdict(float)
    total_sw = 0

    for _, row in min_FE_group.iterrows():
        sw_part = row['sw_part']
        total_sw += sw_part

        ranges = [(row['start1'], row['end1'])]
        if row['type'] == 2:
            ranges.append((row['start2'], row['end2']))

        for start, end in ranges:
            for i in range(int(start), int(end) + 1):
                blockwise_sw[i] += sw_part

    log_total_sw = np.log(total_sw)
    blockwise_g_vals = {
        i: -R * TEMP * (np.log(sw) - log_total_sw)
        for i, sw in blockwise_sw.items()
    }

    blockwise_g_vals = dict(sorted(blockwise_g_vals.items()))
    blockwise_g_vals = pd.DataFrame.from_dict(blockwise_g_vals, orient='index', columns=['g_val'])
    blockwise_g_vals.to_csv(f'{output_dir}/blockwise_g_vals_{id}.csv', index_label='block')
    return blockwise_g_vals

if __name__ == '__main__':
    params = load_params('params.json')

    VDW_INT_ENE = params['VDW_INT_ENE']             
    DCP = params['DCP']                     
    T_REF = params['T_REF']                 
    min_temp = params['MIN_TEMP']
    max_temp = params['MAX_TEMP']
    temp_step = params['TEMP_STEP']
    R = params['R']
    DIELEC = params['DIELEC_CYTO']          # change this to DIELEC_TM if working with transmembrane proteins. 
    N_PLANES = params['N_PLANES']

    master_matrices_dir = './master_matrices'
    g_vecs_dir = './g_vecs'
    elec_intr_dir = './elec_intr_files'
    vdw_intr_dir = './vdw_intr_files'
    pdb_files_dir = './pdb_files'

    TOP_K_HI_FREQ = 10

    output_dir = './analysis_output'
    analysis_output_dir = f'{output_dir}/output_data'
    blockwise_g_vals_dir = f'{output_dir}/blockwise_g_vals'
    min_g_val_output_dir = f'{output_dir}/min_g_vals_groups'
    part_deltaG_output_dir = f'{output_dir}/partial_deltaG'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(analysis_output_dir, exist_ok=True)
    os.makedirs(blockwise_g_vals_dir, exist_ok=True)
    os.makedirs(min_g_val_output_dir, exist_ok=True)
    os.makedirs(part_deltaG_output_dir, exist_ok=True)


    id_list = [file.split('_')[0] for file in os.listdir(master_matrices_dir)]

    for id in id_list:
        id_data = {}

        print(f'Processing {id}...')
        elec_intr = pd.read_csv(f'{elec_intr_dir}/elec_intr_{id}.csv')
        vdw_intr = pd.read_csv(f'{vdw_intr_dir}/vdw_intr_{id}.csv')

        z_planes = generate_z_planes(elec_intr, params['N_PLANES'])
        elec_intr = get_intr_sections(elec_intr, z_planes)
        vdw_intr = get_intr_sections(vdw_intr, z_planes)

        master_matrix = pd.read_csv(f'{master_matrices_dir}/{id}_master_matrix.csv')
        g_vecs = pd.read_csv(f'{g_vecs_dir}/{id}_g_vecs.csv')

        min_g_struct_elems = calc_min_g_struct_elem(g_vecs)                                     # this is a list
        min_g_vals_groups = get_min_g_val_groups(master_matrix, min_g_struct_elems[0])          # this is a dataframe 
        # print(f'Minimum free energy structured elements for {id}: {min_g_struct_elems}')
        # print('-------------------------------------------')
        
        min_gval_gr_df = master_matrix[master_matrix['struct_elem'].isin(min_g_struct_elems)]

        # get the blockwise free energy values
        blockwise_g_vals = get_block_g_vals(id, min_g_struct_elems[0], blockwise_g_vals_dir)

        # fig = plot_row_g_vals(min_gval_gr_df)
        # fig.savefig(f'{output_dir}/row_g_vals_{id}.jpg')

        id_data['min_g_struct_elems'] = [int(x) for x in min_g_struct_elems]
        id_data['min_g_vals_groups'] = min_g_struct_elems
      
        min_g_vals_groups.to_csv(f'{min_g_val_output_dir}/{id}_min_g_vals_groups.csv', index=False)
        # print(type(min_g_vals_groups))

        # calculate partial deltaG
        part_deltaG_data = pd.DataFrame(columns=list(range(min_temp, max_temp+temp_step, temp_step)))

        part_deltaG_data_df = pd.DataFrame()
        for temp in range(min_temp, max_temp+temp_step, temp_step):
            min_g_vals_groups = calculate_partial_deltaG(master_matrix, min_g_struct_elems[0], temp)
            # print(min_g_vals_groups)
            part_deltaG_data[temp] = min_g_vals_groups.g_part
            part_deltaG_data_df = pd.concat([part_deltaG_data_df, min_g_vals_groups])

        # print(part_deltaG_data_df)
        # save the partial deltaG data
        part_deltaG_data.to_csv(f'{part_deltaG_output_dir}/partial_deltaG_{id}.csv', index=False)

        freq_dict = frequency_dict(part_deltaG_data_df, z_planes)
        id_data['freq_dict'] = {int(k): int(v) for k, v in freq_dict.items()}
        id_data['highest_freq'] = [(int(k), int(v)) for k, v in freq_dict.items()][:TOP_K_HI_FREQ]

        max_freq_residues_elec1, max_freq_residues_elec2 = max_freq_residues(id_data['highest_freq'], elec_intr)
        max_freq_residues_vdw1, max_freq_residues_vdw2 = max_freq_residues(id_data['highest_freq'], elec_intr)

        res_freq_elec1, res_freq_elec2 = count_res_freq(max_freq_residues_elec1, max_freq_residues_elec2)
        res_freq_vdw1, res_freq_vdw2 = count_res_freq(max_freq_residues_vdw1, max_freq_residues_vdw2)

        id_data['res_freq_elec1'] = dict(res_freq_elec1)
        id_data['res_freq_elec2'] = dict(res_freq_elec2)
        id_data['res_freq_vdw1'] = dict(res_freq_vdw1)
        id_data['res_freq_vdw2'] = dict(res_freq_vdw2)

        # Before writing to JSON

        with open(f'{analysis_output_dir}/analysis_output_{id}.json', 'w') as f:
            json.dump(id_data, f, indent=4)
        print(f'Analysis for {id} complete.')
        # print('-------------------------------------------')

    print('Analysis complete.')


