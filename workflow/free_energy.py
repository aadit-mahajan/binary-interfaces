import pandas as pd 
import numpy as np
import math
import os
import json
import logging
import concurrent.futures
from sklearn.cluster import KMeans
from sklearn.manifold import Isomap
import networkx as nx

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='debug.log')

pdb_file_dir = './pdb_files'
elec_intr_dir = './elec_intr_files'
vdw_intr_dir = './vdw_intr_files'
g_vecs_dir = './g_vecs'
master_matrices_dir = './master_matrices'

def load_params(param_file):
    with open(param_file) as f:
        params = json.load(f)
    return params

def get_pdb_filepaths(pdb_file_dir):
    pdb_files = []
    for file in os.listdir(pdb_file_dir):
        if file.endswith('.pdb'):
            pdb_files.append(file)
    return pdb_files

def get_elec_intr_filepaths(elec_intr_dir):
    elec_intr_files = []
    for file in os.listdir(elec_intr_dir):
        if file.endswith('.csv'):
            elec_intr_files.append(file)
    return elec_intr_files

def get_vdw_intr_filepaths(vdw_intr_dir):
    vdw_intr_files = []
    for file in os.listdir(vdw_intr_dir):
        if file.endswith('.csv'):
            vdw_intr_files.append(file)
    return vdw_intr_files

def generate_z_planes(interchain_intr, N_PLANES) -> list[float]:
    z_values = []
    for i in range(len(interchain_intr)):
        z_values.append(interchain_intr.atom1_zcoord.iloc[i])
        z_values.append(interchain_intr.atom2_zcoord.iloc[i])

    z_values = np.array(z_values)
    if z_values.size == 0:
        return []
    z_planes = list(np.linspace(z_values.min(), z_values.max(), N_PLANES))
    z_planes.sort()
    return z_planes

def get_blockdet(z_planes, interchain_intr, vdw_intr) -> np.array:
    blockdet = []
    for i in range(len(interchain_intr)):
        z1 = interchain_intr.atom1_zcoord.iloc[i]
        z2 = interchain_intr.atom2_zcoord.iloc[i]
        z1_idx = np.digitize(z1, z_planes)
        z2_idx = np.digitize(z2, z_planes)
        res1 = int(interchain_intr.atom1.iloc[i])
        res2 = int(interchain_intr.atom2.iloc[i])
        if z1_idx == z2_idx:
            blockdet.append((res1, res2, z1_idx))
        else:
            blockdet.append((res1, res2, z2_idx))
    
    for i in range(len(vdw_intr)):
        z1 = vdw_intr.atom1_zcoord.iloc[i]
        z2 = vdw_intr.atom2_zcoord.iloc[i]
        z1_idx = np.digitize(z1, z_planes)
        z2_idx = np.digitize(z2, z_planes)
        res1 = int(vdw_intr.atom1.iloc[i])
        res2 = int(vdw_intr.atom2.iloc[i])
        if z1_idx == z2_idx:
            blockdet.append((res1, res2, z1_idx))
        else:
            blockdet.append((res1, res2, z2_idx))
    
    blockdet = np.array(list(set(blockdet)))
    blockdet = blockdet[blockdet[:, 2].argsort()]
    return blockdet

def get_vdw_intr_sections(vdw_intr, section_boundaries) -> pd.DataFrame:
    
    # generate partitions
    part_1 = pd.cut(vdw_intr['atom1_zcoord'], bins=section_boundaries, labels=False)
    part_2 = pd.cut(vdw_intr['atom2_zcoord'], bins=section_boundaries, labels=False)

    # handle nan values 
    part_1 = part_1.fillna(-1)
    part_2 = part_2.fillna(-1)

    # assign section number to the atoms if both atoms are in the same section
    for i in range(len(vdw_intr)):
        p1 = int(part_1.iloc[i])
        p2 = int(part_2.iloc[i])

        # if p1 == p2, assign the section number to both atoms, else push to section of atom2
        if p1 == p2:
            vdw_intr.at[i, 'section'] = p1
        else:
            vdw_intr.at[i, 'section'] = p2
    
    return vdw_intr

def get_interchain_intr_sections(interchain_intr, section_boundaries) -> pd.DataFrame:
    p1 = pd.cut(interchain_intr['atom1_zcoord'], bins=section_boundaries, labels=False)
    p2 = pd.cut(interchain_intr['atom2_zcoord'], bins=section_boundaries, labels=False)

    # handle nan values 
    p1 = p1.fillna(-1)
    p2 = p2.fillna(-1)
    
    for i in range(len(interchain_intr)):
        p1_val = int(p1.iloc[i])
        p2_val = int(p2.iloc[i])

        # same logic as above. If both atoms are in the same section, assign the section number to both atoms
        if p1_val == p2_val:
            interchain_intr.at[i, 'section'] = p1_val
        else:
            interchain_intr.at[i, 'section'] = p2_val

    return interchain_intr

def calc_path(int_df):
    coords = int_df[['mean_coord_x', 'mean_coord_y', 'mean_coord_z']].values
    isomap = Isomap(n_neighbors=8, n_components=1)
    projected_coords = isomap.fit_transform(coords)

    def do_clustering_kmeans_1d(projected_coords, k=20):
        kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++')
        labels = kmeans.fit_predict(projected_coords)
        cluster_centers = kmeans.cluster_centers_

        return labels, cluster_centers
    
    labels, cluster_centers = do_clustering_kmeans_1d(projected_coords)

    # argsort the cluster centers according to the projected coordinates
    sorted_indices = np.argsort(cluster_centers.flatten())
    cluster_centers = int_df[['mean_coord_x', 'mean_coord_y', 'mean_coord_z']].values[sorted_indices]

    perm = sorted_indices.tolist()
    # print("Optimal path:", perm)
    return perm, labels, cluster_centers

def split_int_df(int_df):
    vdw_int = int_df[int_df['type'] == 'vdw']
    elec_int = int_df[int_df['type'] == 'elec']

    vdw_int = vdw_int.drop(columns=['type'])
    elec_int = elec_int.drop(columns=['type'])  

    return vdw_int, elec_int

def calc_elec_intr_energy(interchain_intr, DIELEC) -> pd.DataFrame:
    interchain_intr['intene'] = (332*4.184*interchain_intr['atom1_chg']*interchain_intr['atom2_chg'])/(DIELEC*interchain_intr['distance'])
    return interchain_intr

def generate_master_mat(n_sections) -> pd.DataFrame:
    n = n_sections
    master_matrix = pd.DataFrame(columns=['struct_elem', 'sw_part', 'start1', 'end1', 'start2', 'end2', 'start3', 'end3', 'type'])
    for i in range(n):
        for j in range(i, n):
            if j-i >= 0:
                master_matrix.loc[len(master_matrix)] = [j-i, 0, i, j, 0, 0, 0, 0, 1]
            for k in range(j, n):
                for l in range(k, n):
                    if j-i >= 0 and k-j > 1 and l-k >= 0:
                        master_matrix.loc[len(master_matrix)] = [(j-i) + (l-k), 0, i, j, k, l, 0, 0, 2]
                    for m in range(l, n):
                        if j-i >= 0 and k-j > 1 and l-k > 1 and m-l >= 0:
                            master_matrix.loc[len(master_matrix)] = [(j-i) + (l-k) + (m-l), 0, i, j, k, l, m, n-1, 3]
    
    return master_matrix

def get_sw_values(master_matrix, elec_intr, vdw_intr, temp) -> pd.DataFrame:
    # VDW_INT_ENE = -77.75/1000  
    # DCP = -0.3112/1000  # arbitrary value for now
    # T_REF = 385
    # R = 8.314/1000
    # IS = 0.043
    
    # Pre-calculate constants
    ISfac = 5.66 * np.sqrt(IS/temp) * np.sqrt(80/29)
    exp_ISfac = np.exp(-ISfac)
    solv_energy_const = DCP * (temp - T_REF) - temp * DCP * np.log(temp/T_REF)
    
    # Group data
    vdw_intr_sec_grp = vdw_intr.groupby('section')
    elec_intr_sec_grp = elec_intr.groupby('section')
    
    # Pre-calculate sums and counts
    elec_sums = elec_intr_sec_grp['intene'].sum()
    vdw_counts = vdw_intr_sec_grp.size()
    
    sections = sorted(list(set(vdw_intr['section']).union(set(elec_intr['section']))))
    section_idx_map = {section: idx for idx, section in enumerate(sections)}
    n_sections = N_PLANES + 1

    elec_vec = np.zeros(n_sections)
    vdw_vec = np.zeros(n_sections)

    for sec, val in elec_sums.items():
        elec_vec[section_idx_map[sec]] = val
    for sec, count in vdw_counts.items():
        vdw_vec[section_idx_map[sec]] = count

    # create a combination matrix (binary matrix)
    comb_mat = np.zeros((len(master_matrix), n_sections), dtype=int)
    for idx, row in master_matrix.iterrows():
        if row['type'] == 1:
            elems = list(range(int(row['start1']), int(row['end1']) + 1))
        if row['type'] == 2:
            elems = list(range(int(row['start1']), int(row['end1']) + 1)) + list(range(int(row['start2']), int(row['end2']) + 1))
        if row['type'] == 3:
            elems = list(range(int(row['start1']), int(row['end1']) + 1)) + list(range(int(row['start2']), int(row['end2']) + 1)) + list(range(int(row['start3']), int(row['end3']) + 1))
        for elem in elems:
            if elem in section_idx_map:
                comb_mat[idx, section_idx_map[elem]] = 1

    # Calculate the total energy for each combination
    elec_intene_tot = comb_mat @ elec_vec * exp_ISfac
    vdw_intene_tot = comb_mat @ vdw_vec * VDW_INT_ENE
    
    solv_energy = comb_mat @ vdw_vec * solv_energy_const
    deltaE = vdw_intene_tot + elec_intene_tot + solv_energy
    
    # def _calculate_sw_partial(row):
    #     if row['type'] == 1:
    #         elems_in_comb = range(int(row['start1']), int(row['end1'])+1)
    #     else:
    #         elems_in_comb = list(range(int(row['start1']), int(row['end1'])+1)) + list(range(int(row['start2']), int(row['end2'])+1))
        
    #     elec_intene_tot_part = sum(elec_sums.get(elem, 0) for elem in elems_in_comb)
    #     vdw_counts_part = sum(vdw_counts.get(elem, 0) for elem in elems_in_comb)
        
    #     elec_intene_tot_part *= exp_ISfac
    #     vdw_intene_tot_part = vdw_counts_part * VDW_INT_ENE
    #     solv_energy = vdw_counts_part * solv_energy_const
    #     deltaE_part = vdw_intene_tot_part + elec_intene_tot_part + solv_energy
        
    #     return np.exp(-(deltaE_part / (R * temp)))
    
    # master_matrix['sw_part'] = master_matrix.apply(_calculate_sw_partial, axis=1)
    master_matrix['sw_part'] = np.exp(-(deltaE / (R * temp)))
    
    return master_matrix

def save_master_matrix(master_matrix, id, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    master_matrix.to_csv(f'{output_dir}/{id}_master_matrix.csv', index=False)

def save_g_vecs(g_vecs, id, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    g_vecs.to_csv(f'{output_dir}/{id}_g_vecs.csv', index=False)

if __name__ == '__main__':
    params = load_params('params.json')

    VDW_INT_ENE = params['VDW_INT_ENE']             
    DCP = params['DCP']                     
    T_REF = params['T_REF']      
    IS = params['IS']           
    min_temp = params['MIN_TEMP']
    max_temp = params['MAX_TEMP']
    temp_step = params['TEMP_STEP']
    R = params['R']
    DIELEC_CYTO = params['DIELEC_CYTO']          
    DIELEC_TM = params['DIELEC_TM']
    N_PLANES = params['N_PLANES']

    elec_intr_files = get_elec_intr_filepaths(elec_intr_dir) 
    vdw_intr_files = get_vdw_intr_filepaths(vdw_intr_dir)
    # pdb_files = get_pdb_filepaths(pdb_file_dir)
    
    # id_list = []
    # for file in pdb_files:
    #     id = os.path.basename(file).split('.')[0]
    #     id_list.append(id)
    
    id_list = [file.split('.')[0] for file in os.listdir(pdb_file_dir)]

    print(
        f'Found pdb {len(id_list)} files...\n', '\n'.join(id_list), '\n-------------\n'
    )

    def process_pdb_files(id):
        try:
            
            if id.find('_') != -1:
                choice = id.split('_')[1]
                if choice == "m":
                    DIELEC = DIELEC_TM
                if choice == "c":
                    DIELEC = DIELEC_CYTO
            else:
                DIELEC = DIELEC_CYTO

            logging.info(f'Processing {id}...')
            
            elec_intr = pd.read_csv(f'{elec_intr_dir}/elec_intr_{id}.csv')
            vdw_intr = pd.read_csv(f'{vdw_intr_dir}/vdw_intr_{id}.csv')

            g_vecs = pd.DataFrame()

            # # 1. generate z_planes
            # z_planes = generate_z_planes(elec_intr, N_PLANES)

            # 2. generate master_matrix
            n_sections = N_PLANES + 1
            master_matrix = generate_master_mat(n_sections)

            # # 3. get sections for vdw_intr
            # vdw_intr = get_vdw_intr_sections(vdw_intr, z_planes)

            # # 4. get sections for elec_intr
            # elec_intr = get_interchain_intr_sections(elec_intr, z_planes)

            # calculate mean of each pair of interactions
            vdw_intr['type'] = 'vdw'
            elec_intr['type'] = 'elec'

            vdw_intr['mean_coord_x'] = (vdw_intr['atom1_xcoord'] + vdw_intr['atom2_xcoord']) / 2
            vdw_intr['mean_coord_y'] = (vdw_intr['atom1_ycoord'] + vdw_intr['atom2_ycoord']) / 2
            vdw_intr['mean_coord_z'] = (vdw_intr['atom1_zcoord'] + vdw_intr['atom2_zcoord']) / 2

            elec_intr['mean_coord_x'] = (elec_intr['atom1_xcoord'] + elec_intr['atom2_xcoord']) / 2
            elec_intr['mean_coord_y'] = (elec_intr['atom1_ycoord'] + elec_intr['atom2_ycoord']) / 2
            elec_intr['mean_coord_z'] = (elec_intr['atom1_zcoord'] + elec_intr['atom2_zcoord']) / 2

            # cluster centers
            total_int = pd.concat([vdw_intr, elec_intr], ignore_index=True)

            perm, labels, cluster_centers = calc_path(total_int)
            # Create a mapping from old labels to new labels based on the optimal path
            perm_inverse = {old_label: new_label for new_label, old_label in enumerate(perm)}

            # Assign new cluster labels based on sorted path
            sorted_labels = np.vectorize(perm_inverse.get)(labels)
            total_int['section'] = sorted_labels

            # split the interactions into vdw and elec
            vdw_intr, elec_intr = split_int_df(total_int)

            # save these interactions
            vdw_intr.to_csv(f'{vdw_intr_dir}/vdw_intr_{id}.csv', index=False)
            elec_intr.to_csv(f'{elec_intr_dir}/elec_intr_{id}.csv', index=False)
            
            # 5. calculate elec_intr_energy
            elec_intr = calc_elec_intr_energy(elec_intr, DIELEC)

            # 6. get sw_values
            for temp in range(min_temp, max_temp + temp_step, temp_step):
                fin_master_matrix = get_sw_values(master_matrix, elec_intr, vdw_intr, temp)
                z_part_total = list(fin_master_matrix.groupby('struct_elem').sw_part.sum())
                z_total = sum(z_part_total)
                p_vec = [i/z_total for i in z_part_total]
                g_vec = [-R*temp*math.log(i) for i in p_vec]
                g_vec = [g-g_vec[0] for g in g_vec]  # normalize g_vec
                g_vecs[temp] = g_vec

            # 7. save master_matrix
            save_master_matrix(fin_master_matrix, id, master_matrices_dir)

            # 8. save g_vecs
            save_g_vecs(g_vecs, id, g_vecs_dir)

            logging.info(f'Finished processing {id}...')
        except Exception as e:
            logging.error(f'Error processing {id}: {e}', exc_info=True)
            print(f'Error processing {id}: {e}')

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_pdb_files, id_list)

    sample_id = id_list[0] if id_list else None
    process_pdb_files(sample_id)

    print('Done!')