import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict

def load_params(param_file_path: str) -> dict:
    with open(param_file_path, "r") as f:
        return json.load(f)

def ensure_dirs(dirs: list[str]) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def get_intr_sections(interactions: pd.DataFrame, section_boundaries: list[float]) -> pd.DataFrame:
    """
    Assign z-section numbers to interactions.
    """
    p1 = pd.cut(interactions["atom1_zcoord"], bins=section_boundaries, labels=False).fillna(-1).astype(int)
    p2 = pd.cut(interactions["atom2_zcoord"], bins=section_boundaries, labels=False).fillna(-1).astype(int)
    interactions["section"] = np.where(p1 == p2, p1, p2)
    return interactions

# Free energy calculations
def calc_min_g_struct_elem(g_vecs: pd.DataFrame) -> list[int]:
    """
    Return structured elements with minimum free energy values.
    """
    min_g_vals = g_vecs.min(axis="index")
    elems = []
    for temp in g_vecs.columns:
        min_val = min_g_vals[temp]
        elems.extend(g_vecs[temp].index[g_vecs[temp] == min_val].tolist())
    return list(set(elems))

def get_min_g_val_groups(master_matrix: pd.DataFrame, struct_elem: int) -> pd.DataFrame:
    """
    Return the group of rows for a minimum energy structured element.
    """
    return master_matrix.groupby("struct_elem").get_group(struct_elem)

def frequency_dict(min_g_vals_group: pd.DataFrame, n_planes: int) -> dict[int, int]:
    """
    Count frequency of interacting blocks in min-energy groups.
    """
    int_blocks = []
    for _, row in min_g_vals_group.iterrows():
        row = row.astype(int)
        seq1, seq2 = list(range(row.start1, row.end1 + 1)), list(range(row.start2, row.end2 + 1))
        if len(seq1) != (row.struct_elem + 1) and len(seq2) != (row.struct_elem + 1):
            int_blocks.extend([seq1, seq2])
        else:
            int_blocks.append(seq1 if len(seq1) == row.struct_elem + 1 else seq2)

    freq = np.zeros(n_planes + 1, dtype=int)
    for block in int_blocks:
        for elem in block:
            freq[elem] += 1
    return dict(sorted(enumerate(freq), key=lambda x: x[1], reverse=True))

def max_freq_residues(freq_dict: dict[int, int], elec_intr: pd.DataFrame) -> tuple[list[int], list[int]]:
    """
    Extract residues with highest frequency of interaction.
    """
    max_res_ch1, max_res_ch2 = [], []
    for elem, _ in freq_dict.items():
        block1, block2 = elec_intr.loc[elec_intr.section == elem, "atom1_resnum"], elec_intr.loc[elec_intr.section == elem, "atom2_resnum"]
        if block1.empty or block2.empty:
            continue
        max_res_ch1.extend(block1)
        max_res_ch2.extend(block2)
    return max_res_ch1, max_res_ch2

def count_res_freq(residues: list[int]) -> dict[int, int]:
    """
    Count frequency of residues.
    """
    freq = defaultdict(int)
    for res in residues:
        freq[res] += 1
    return dict(freq)

def calculate_partial_deltaG(master_matrix: pd.DataFrame, struct_elem: int, temp: int, R: float) -> pd.DataFrame:
    """
    Generate deltaG values for minimum energy group.
    """
    group = master_matrix.groupby("struct_elem").get_group(struct_elem).copy()
    z_total = group.sw_part.sum()
    group["partial_z"] = group.sw_part / z_total
    group["g_part"] = -R * temp * np.log(group.partial_z)
    return group

def get_block_g_vals(id: str, struct_elem: int, master_matrices_dir: str, output_dir: str, R: float) -> pd.DataFrame:
    """
    Compute blockwise free energy values for min-energy group.
    """
    master_matrix = pd.read_csv(f"{master_matrices_dir}/{id}_master_matrix.csv")
    group = master_matrix[master_matrix["struct_elem"] == struct_elem]

    TEMP = 300
    blockwise_total = defaultdict(float)

    for _, row in group.iterrows():
        ranges = [(row.start1, row.end1)]
        if row.type == 2:
            ranges.append((row.start2, row.end2))
        for start, end in ranges:
            for i in range(int(start), int(end) + 1):
                blockwise_total[i] += row.sw_part

    total = master_matrix["sw_part"].sum()
    blockwise_g_vals = {
        i: -R * TEMP * np.log(w / total) for i, w in blockwise_total.items()
    }

    df = pd.DataFrame.from_dict(dict(sorted(blockwise_g_vals.items())), orient="index", columns=["g_val"])
    df.to_csv(f"{output_dir}/blockwise_g_vals_{id}.csv", index_label="block")
    return df

# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------
if __name__ == "__main__":
    params = load_params("params.json")

    # Parameters
    R = params["R"]
    N_PLANES = params["N_PLANES"]
    min_temp, max_temp, temp_step = params["MIN_TEMP"], params["MAX_TEMP"], params["TEMP_STEP"]

    # Directories
    master_matrices_dir = "./master_matrices"
    g_vecs_dir = "./g_vecs"
    elec_intr_dir = "./elec_intr_files"
    vdw_intr_dir = "./vdw_intr_files"
    pdb_files_dir = "./pdb_files"
    output_dir = "./analysis_output"

    dirs = {
        "analysis": f"{output_dir}/output_data",
        "blockwise": f"{output_dir}/blockwise_g_vals",
        "min_g": f"{output_dir}/min_g_vals_groups",
        "partial": f"{output_dir}/partial_deltaG",
    }
    ensure_dirs([output_dir, *dirs.values()])

    id_list = [file.split(".")[0] for file in os.listdir(pdb_files_dir)]

    for id in id_list:
        print(f"Processing {id}...")
        elec_intr = pd.read_csv(f"{elec_intr_dir}/elec_intr_{id}.csv")

        master_matrix = pd.read_csv(f"{master_matrices_dir}/{id}_master_matrix.csv")
        g_vecs = pd.read_csv(f"{g_vecs_dir}/{id}_g_vecs.csv")

        # Minimum free energy structured elements
        min_struct_elems = calc_min_g_struct_elem(g_vecs)
        min_groups = master_matrix[master_matrix["struct_elem"].isin(min_struct_elems)]
        min_groups.to_csv(f"{dirs['min_g']}/{id}_min_g_vals_groups.csv", index=False)

        # Blockwise free energies
        blockwise_g_vals = get_block_g_vals(id, min_struct_elems[0], master_matrices_dir, dirs["blockwise"], R)

        # Partial ΔG over temperatures
        part_deltaG = pd.DataFrame()
        part_deltaG_all = pd.DataFrame()
        for temp in range(min_temp, max_temp + temp_step, temp_step):
            deltaG_df = calculate_partial_deltaG(master_matrix, min_struct_elems[0], temp, R)
            part_deltaG[temp] = deltaG_df.g_part
            part_deltaG_all = pd.concat([part_deltaG_all, deltaG_df])
        part_deltaG.to_csv(f"{dirs['partial']}/partial_deltaG_{id}.csv", index=False)

        # Frequencies
        freq = frequency_dict(part_deltaG_all, N_PLANES)
        top_freq = list(freq.items())[:10]
        max_res1, max_res2 = max_freq_residues(dict(top_freq), elec_intr)
        res_freq1, res_freq2 = count_res_freq(max_res1), count_res_freq(max_res2)

        # Save JSON summary
        id_data = {
            "min_g_struct_elems": [int(x) for x in min_struct_elems],
            "freq_dict": {int(k): int(v) for k, v in freq.items()},
            "highest_freq": [(int(k), int(v)) for k, v in top_freq],
            "res_freq_elec1": res_freq1,
            "res_freq_elec2": res_freq2,
        }
        with open(f"{dirs['analysis']}/analysis_output_{id}.json", "w") as f:
            json.dump(id_data, f, indent=4)

        print(f"Analysis for {id} complete.")

    print("All analyses complete.")
