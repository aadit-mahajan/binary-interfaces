# plot_utils.py
import os
import matplotlib.pyplot as plt
import seaborn as sns

plot_output_dir = './plots'

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
    
    plt.savefig(f'../plots/FE_curve_{id}.jpg')


