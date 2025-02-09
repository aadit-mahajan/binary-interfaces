import numpy as np
import pandas as pd
import os
from free_energy import generate_z_planes
import json
import datetime

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

def get_min_g_val_groups(master_matrix, min_g_vals_struct_elems):
    min_g_vals_groups = []
    struct_group = master_matrix.groupby('struct_elem')
    for struct_elem in min_g_vals_struct_elems:
        min_g_vals_groups.append(struct_group.get_group(struct_elem))
    return min_g_vals_groups

def frequency_dict(min_g_vals_groups, z_planes):
    '''
    Finding the occurrence of each interacting block in the groups with minimum free energy values
    This will give a good idea of the favourably interacting blocks. 
    '''
    for group in min_g_vals_groups:
        int_blocks = []
        for row in group.iterrows():
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

if __name__ == '__main__':
    master_matrices_dir = './master_matrices'
    g_vecs_dir = './g_vecs'
    elec_intr_dir = './elec_intr_files'
    vdw_intr_dir = './vdw_intr_files'
    pdb_files_dir = './pdb_files'

    params = load_params('params.json')

    TOP_K_HI_FREQ = 10

    output_dir = './analysis_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    id_list = [file.split('_')[0] for file in os.listdir(master_matrices_dir)]

    data = {}
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

        min_g_struct_elems = calc_min_g_struct_elem(g_vecs)
        min_g_vals_groups = get_min_g_val_groups(master_matrix, min_g_struct_elems)
        print(f'Minimum free energy structured elements for {id}: {min_g_struct_elems}')
        print('-------------------------------------------')
        
        # print(type(min_g_struct_elems), type(min_g_vals_groups))
        id_data['min_g_struct_elems'] = [int(x) for x in min_g_struct_elems]
        # id_data['min_g_vals_groups'] = list(min_g_vals_groups)

        freq_dict = frequency_dict(min_g_vals_groups, z_planes)
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
        
        data[id] = id_data
        print(f'Analysis for {id} complete.')
        print('-------------------------------------------')

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    with open(f'{output_dir}/analysis_output_{timestamp}.json', 'w') as f:
        json.dump(data, f, indent=4)
    
    print('Analysis complete.')


