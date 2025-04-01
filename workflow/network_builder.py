import networkx as nx
import numpy as np
import pandas as pd
from Levenshtein import distance
import matplotlib.pyplot as plt
import math

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

def create_comb_mat(struct_elem_list, N_PLANES) -> np.array:
    n = len(struct_elem_list)
    comb_mat = np.zeros((n, N_PLANES+1))
    for i in range(n):
        for elem in struct_elem_list[i]:
            comb_mat[i][elem] = 1
    return comb_mat

def get_struct_elem_string(struct_elem_list, N_PLANES) -> list:
    struct_elem_str = []
    for entry in struct_elem_list:
        temp = np.zeros(N_PLANES+1)
        for elem in entry:
            temp[elem] = 1

        struct_elem_str.append(''.join([str(int(i)) for i in temp]))
    return struct_elem_str

def calc_edge_weight(g1, g2, t=10) -> float:
    '''
    This maps the difference in free energy values to an edge weight. 
    The molecule should have higher chance of switching between conformations that have similar free energy values.
    hence the edge weights have been defined as the reciprocal of the square root of the difference in free energy values.
    this makes two conformations that have close free energy values to have a higher edge weight and vice versa.
    '''
    
    diff = abs(g1 - g2)
    if diff == 0:
        return 1
    return t/math.sqrt(diff)

def get_graph(struct_elem_str, min_g_vals_groups) -> nx.Graph:
    G = nx.Graph()
    for i in range(len(struct_elem_str)):
        for j in range(i+1, len(struct_elem_str)):
            dist = distance(struct_elem_str[i], struct_elem_str[j])
            if dist == 1:
                edge_weight = calc_edge_weight(min_g_vals_groups.iloc[i].g_part, min_g_vals_groups.iloc[j].g_part, t=20)
                # print(edge_weight)
                G.add_edge(i, j, weight=edge_weight)
    return G

struct_elem_str = get_struct_elem_string(struct_elem_list, N_PLANES)
# print(struct_elem_str)
G = get_graph(struct_elem_str, min_g_vals_groups)

def draw_weighted_network(G):
    pos = nx.spring_layout(G, k=0.3, seed=42, iterations=75)
    weights = [G[u][v]['weight'] for u,v in G.edges()]
    plt.figure()
    plt.title("Graph of structured elements")
    nx.draw_networkx(G, pos, 
            with_labels=True, 
            width=weights, 
            node_size=200,
            node_color='gray',
            font_size=7,
            font_color='black')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    
    
    


