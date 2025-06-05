import os 
import subprocess

# 1. contact_calc
# 2. free_energy
# 3. interaction analysis
# 4. plot utils
# 5. mod pdb

if __name__ == "__main__":
    # Define the directory containing the scripts
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # clean the pdb_files directory
    pdb_files_dir = os.path.join(script_dir, 'pdb_files')
    if os.path.exists(pdb_files_dir):
        for file in os.listdir(pdb_files_dir):
            file_path = os.path.join(pdb_files_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Cleaned the directory: {pdb_files_dir}")

    else:
        print(f"Directory does not exist: {pdb_files_dir}")
        
    # List of scripts to run
    scripts = [
        "pdb_aligner.py",
        "contact_calc.py",
        "free_energy.py",
        "intr_analysis.py",
        "plot_utils.py",
        "mod_pdb.py"
    ]

    # Run each script
    for script in scripts:
        script_path = os.path.join(script_dir, script)
        print(f"--------------------\nRunning {script}...\n--------------------\n")
        subprocess.run(["python3", script_path], check=True)
