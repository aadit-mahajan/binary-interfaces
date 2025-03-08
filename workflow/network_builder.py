import networkx as nx
import numpy as np
import pandas as pd
from Levenshtein import distance


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

