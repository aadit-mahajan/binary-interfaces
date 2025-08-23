# plot_utils.py
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_g_vecs(g_vecs, temps, id, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure()
    for temp in temps:
        plt.plot(g_vecs[temp])
    plt.legend(temps)
    plt.title(f'FE curve for PDB ID: {id}')
    plt.xlabel('Number of interacting blocks')
    plt.ylabel('Free energy (kJ/mol)')
    # plt.ylim(-200, 100)
    plt.xticks(range(0, len(g_vecs[temp]), 2))
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/FE_curve_{id}.jpg')
    plt.close()

def get_struct_elem_list(master_matrix) -> list:
    struct_elem_list = []
    for i in range(len(master_matrix)):
        start1 = int(master_matrix.iloc[i].start1)
        end1 = int(master_matrix.iloc[i].end1)
        start2 = int(master_matrix.iloc[i].start2)
        end2 = int(master_matrix.iloc[i].end2)
        comb_type = master_matrix.iloc[i].type
        if comb_type == 1:
            elems_in_comb = list(range(start1, end1+1))
        elif comb_type == 2:
            elems_in_comb = list(range(start1, end1+1)) + list(range(start2, end2+1))
        struct_elem_list.append(elems_in_comb)

    return struct_elem_list

def plot_row_g_vals(min_g_vals_groups):
    ssa_entries = min_g_vals_groups[min_g_vals_groups.type == 1].reset_index()
    dsa_entries = min_g_vals_groups[min_g_vals_groups.type == 2].reset_index()
    tsa_entries = min_g_vals_groups[min_g_vals_groups.type == 3].reset_index()
    total_entries = min_g_vals_groups.reset_index()

    fig = plt.figure(figsize=(15, 5))
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(ssa_entries.g_part)
    plt.ylabel('free energy (kJ/mol)')
    plt.title('Free energy values for ssa structured elements')
    plt.xlabel('Combination index')
    plt.xticks(range(0, len(dsa_entries.g_part)))

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(dsa_entries.g_part)
    plt.ylabel('free energy (kJ/mol)')
    plt.title('Free energy values for dsa structured elements')
    plt.xlabel('Combination index')
    plt.xticks(range(0, len(dsa_entries.g_part)))

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(tsa_entries.g_part)
    plt.ylabel('free energy (kJ/mol)')
    plt.title("Free energy values for tsa structured elements")
    plt.xlabel('Combination index')
    plt.xticks(range(0, len(tsa_entries.g_part)))

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(total_entries.g_part)
    plt.ylabel('free energy (kJ/mol)')
    plt.title('Free energy values for all structured elements')
    plt.xlabel('Combination index')
    plt.xticks(range(0, len(dsa_entries.g_part)))

    plt.tight_layout()
    plt.suptitle('Free energy values for structured elements', fontsize=16)

    plt.savefig(f'{plot_output_dir}/row_g_vals.jpg')
    plt.close()

def plot_blockwise_g_vals(id, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    blockwise_g_vals = pd.read_csv(f'{blockwise_g_vals_data_dir}/blockwise_g_vals_{id}.csv')
    plt.plot(blockwise_g_vals['g_val'])
    plt.xlabel('Structured element index')
    plt.xticks(range(0, len(blockwise_g_vals)))
    plt.ylabel('Free energy (kJ/mol)')
    plt.title(f'Free energy values for PDB ID: {id}')
    plt.grid()

    plt.savefig(f'{output_dir}/{id}_blockwise_g_vals.jpg')
    plt.close()

def plot_part_deltaG_temp(id, output_dir, temp):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    partial_delta_G = pd.read_csv(f'{part_deltaG_dir}/partial_deltaG_{id}.csv')
    xdat = range(len(partial_delta_G))
    plt.plot(xdat, partial_delta_G[f'{temp}'])
    plt.xlabel('Combination index')
    plt.xticks(range(0, len(partial_delta_G), 5))
    plt.ylabel('Free energy (kJ/mol)')
    plt.title(f'Free energy values for PDB ID: {id} at temperature: {temp}')
    plt.grid()

    plt.savefig(f'{output_dir}/{id}_partial_delta_G_{temp}.jpg')
    plt.close()

def plot_heatmap_part_deltaG(id, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    delta_G = pd.read_csv(f'{part_deltaG_dir}/partial_deltaG_{id}.csv')
    fig = plt.figure(figsize=(15, 10))
    sns.heatmap(delta_G)
    plt.xlabel('Temperature')
    plt.ylabel('Combination index')
    plt.title(f'Heatmap of free energy values for PDB ID: {id}')
    plt.grid()

    plt.savefig(f'{output_dir}/{id}_delta_G_heatmap.jpg')
    plt.close()

if __name__ == '__main__':
    
    plot_output_dir = './plots'
    pdb_files_dir = './pdb_files'

    id_list = [file.split('.')[0] for file in os.listdir(pdb_files_dir) if file.endswith('.pdb')]
    os.makedirs(plot_output_dir, exist_ok=True)

    total_gvecs_output_dir = f'./{plot_output_dir}/total_gvecs'
    os.makedirs(total_gvecs_output_dir, exist_ok=True)

    deltaG_part_output_dir = f'./{plot_output_dir}/deltaG_part'
    os.makedirs(deltaG_part_output_dir, exist_ok=True)

    deltaG_heatmap_output_dir = f'./{plot_output_dir}/deltaG_heatmap'
    os.makedirs(deltaG_heatmap_output_dir, exist_ok=True)

    blockwise_g_vals_dir = f'./{plot_output_dir}/blockwise_g_vals'
    os.makedirs(blockwise_g_vals_dir, exist_ok=True)

    g_vecs_dir = './g_vecs'
    part_deltaG_dir = './analysis_output/partial_deltaG'
    blockwise_g_vals_data_dir = './analysis_output/blockwise_g_vals'

    # load g_vecs
    for file in os.listdir(g_vecs_dir):
        g_vecs = pd.read_csv(f'{g_vecs_dir}/{file}')
        id = file.split('_')[0]
        temps = g_vecs.columns
        plot_g_vecs(g_vecs, temps, id, total_gvecs_output_dir)

    # load blockwise_g_vals
    for file in id_list:
        id = file.split('.')[0]
        plot_blockwise_g_vals(file, blockwise_g_vals_dir)

    # load partial_deltaG
    for file in id_list:
        id = file.split('.')[0]
        plot_part_deltaG_temp(id, deltaG_part_output_dir, '300')
        plot_part_deltaG_temp(id, deltaG_part_output_dir, '310')
        plot_heatmap_part_deltaG(id, deltaG_heatmap_output_dir)

    print('Plots saved successfully!')


