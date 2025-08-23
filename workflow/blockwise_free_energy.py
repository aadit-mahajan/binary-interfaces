import numpy as np
import pandas as pd
import math
import os
import json

master_matrix_dir = './master_matrices'
blockwise_g_vals_dir = './blockwise_g_vals'
def load_params(param_file_path):
      with open(param_file_path, 'r') as f:
         params = json.load(f)
      return params

def get_block_g_vals(id, min_g_vals_struct_elems, master_matrix_dir):
   master_matrix = pd.read_csv(f'{master_matrix_dir}/master_matrix_{id}.csv')
   min_FE_macrostate = min_g_vals_struct_elems[0]
   min_FE_group = master_matrix.groupby('struct_elem').get_group(min_FE_macrostate)

   params = load_params('params.json')
   blockwise_sw = {}
   total_sw = 0
   TEMP = params['MIN_TEMP']  # Assuming we are using the minimum temperature for blockwise calculations
   R = params['R']
   total_sw = min_FE_group['sw_part'].sum()
   print(f'Total SW: {total_sw}')

   for i in range(len(min_FE_group)):
      row = min_FE_group.iloc[i]
      
      s1 = row.start1
      s2 = row.start2
      e1 = row.end1
      e2 = row.end2
      arr_type = row.type
      sw_part = row.sw_part
      total_sw += sw_part

      if arr_type == 1:
         i = s1
         while i <= e1:
            if i in blockwise_sw.keys():
               blockwise_sw[i] += sw_part
            else:
               blockwise_sw[i] = sw_part
            i+=1
      
      if arr_type == 2:
         i = s1
         while i <= e1:
            if i in blockwise_sw.keys():
               blockwise_sw[i] += sw_part
            else:
               blockwise_sw[i] = sw_part
            i+=1
         i = s2
         while i <= e2:
            if i in blockwise_sw.keys():
               blockwise_sw[i] += sw_part
            else:
               blockwise_sw[i] = sw_part
            i+=1

   blockwise_g_vals = {}
   for i in blockwise_sw.keys():
      probability = blockwise_sw[i] / total_sw
      print(f'Probability: {probability}')
      blockwise_g_vals[i] = -R*TEMP*math.log(blockwise_sw[i] / total_sw)
   
   blockwise_g_vals = dict(sorted(blockwise_g_vals.items(), key = lambda x: x[0]))
   return blockwise_g_vals

def save_blockwise_g_vals(blockwise_g_vals, id, output_dir):
   if not os.path.exists(output_dir):
      os.makedirs(output_dir)

   blockwise_g_vals_df = pd.DataFrame(blockwise_g_vals.items(), columns=['struct_elem', 'g_val'])
   blockwise_g_vals_df.to_csv(f'{output_dir}/blockwise_g_vals_{id}.csv', index=False)
