import numpy as np
import pandas as pd 
import os
import json
import math

def load_params(param_file_path):
    with open(param_file_path, 'r') as f:
        params = json.load(f)
    return params

def load_interactions(elec_intr_file_path, vdw_intr_file_path):

    elec_intr = pd.read_csv(elec_intr_file_path)
    vdw_intr = pd.read_csv(vdw_intr_file_path)

    elec_intr['intr_type'] = 'elec'
    vdw_intr['intr_type'] = 'vdw'

    # combine the two dataframes
    total_interactions = pd.concat([elec_intr, vdw_intr], ignore_index=True)

    return total_interactions

def generate_master_matrix(total_interactions):
    n = len(total_interactions)
    master_mat_combs = []
    for i in range(n):
        for j in range(i, n):
            if j-i >= 0:
                master_mat_combs.append((j-i, 0, i, j, 0, 0, 1))
            for k in range(j, n):
                for l in range(k, n):
                    if j-i >= 0 and k-j > 1 and l-k >= 0:
                        master_mat_combs.append([(j-i) + (l-k), 0, i, j, k, l, 2])

    master_mat_combs_df = pd.DataFrame(master_mat_combs, columns=['struct_elem', 'type', 'start1', 'end1', 'start2', 'end2', 'arr_type'])

    # generate a binary matrix from the combinations
    master_mat = np.zeros((len(master_mat_combs), n), dtype=int)
    for idx, comb in enumerate(master_mat_combs):
        if isinstance(comb, tuple):
            master_mat[idx, comb[2]:comb[3]+1] = 1
        else:
            master_mat[idx, comb[2]:comb[3]+1] = 1
            master_mat[idx, comb[4]:comb[5]+1] = 1

    return master_mat, master_mat_combs_df

def calculate_sw_part(master_matrix, total_interactions, temp, params):
     # Pre-calculate constants
    ISfac = 5.66 * np.sqrt(params['IS']/temp) * np.sqrt(80/29)
    exp_ISfac = np.exp(-ISfac)
    solv_energy_const = params['DCP'] * (temp - params['T_REF']) - temp * params['DCP'] * np.log(temp/params['T_REF'])

    int_energy = []
    for index, row in total_interactions.iterrows():
        if row['intr_type'] == 'elec':
            energy = (332 * 4.184 * row['atom1_chg'] * row['atom2_chg']) / (params['DIELEC_CYTO'] * row['distance'])
        if row['intr_type'] == 'vdw':
            # Assuming VDW interaction energy is calculated in a similar way
            energy = params['VDW_INT_ENE'] + solv_energy_const

        int_energy.append(energy)
    total_interactions['int_energy'] = int_energy

    # Calculate the sw_part for each interaction
    total_E = master_matrix @ total_interactions['int_energy'].values * exp_ISfac
    sw_part = np.exp(-total_E / (params['R'] * temp)) 
    struct_elem = master_matrix.sum(axis=1)
    sw_part = pd.DataFrame({'sw_part': sw_part, 'struct_elem': struct_elem})
    return sw_part

def main():
    params = load_params('params.json')        
    min_temp = params['MIN_TEMP']
    max_temp = params['MAX_TEMP']
    temp_step = params['TEMP_STEP']
    R = params['R']

    pdb_files_dir = './pdb_files'
    elec_intr_files_dir = './elec_intr_files'
    vdw_intr_files_dir = './vdw_intr_files'
    master_matrix_dir = './master_matrices'

    pdb_id_list = [file.split('.')[0] for file in os.listdir(pdb_files_dir)]

    for id in pdb_id_list:
        print(f"Processing {id}...")
        elec_intr_file_path = os.path.join(elec_intr_files_dir, f'elec_intr_{id}.csv')
        vdw_intr_file_path = os.path.join(vdw_intr_files_dir, f'vdw_intr_{id}.csv')

        if not os.path.exists(elec_intr_file_path) or not os.path.exists(vdw_intr_file_path):
            print(f"Interaction files for {id} not found. Skipping...")
            continue

        total_interactions = load_interactions(elec_intr_file_path, vdw_intr_file_path)
        master_mat, master_mat_combs = generate_master_matrix(total_interactions)

        print('combinations generated')
        g_vecs = {}

        for temp in range(min_temp, max_temp + temp_step, temp_step):

            sw_part = calculate_sw_part(master_mat,total_interactions, temp, params)
            master_mat_combs['sw_part'] = sw_part['sw_part']

            z_part_total = list(sw_part.groupby('struct_elem').sw_part.sum())
            z_total = sum(z_part_total)
            p_vec = [i/z_total for i in z_part_total]
            g_vec = [-R*temp*math.log(i) for i in p_vec]
            g_vec = [g-g_vec[0] for g in g_vec]  # normalize g_vec
            g_vecs[temp] = g_vec
        
        # save the master_mat_combs 
        master_mat_combs.to_csv(f'{master_matrix_dir}/{id}_master_matrix.csv', index=False)

        print("g_vecs for each temperature calculated.")
        # Save the g_vecs for each temperature
        g_vecs_df = pd.DataFrame(g_vecs)
        g_vecs_file_path = f'./g_vecs/{id}_g_vecs_total.csv'
        os.makedirs(os.path.dirname(g_vecs_file_path), exist_ok=True)
        g_vecs_df.to_csv(g_vecs_file_path, index=False)
        print("-"*20) 

if __name__ == "__main__":
    main()
