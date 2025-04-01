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

    # List of scripts to run
    scripts = [
        "contact_calc.py",
        "free_energy.py",
        "intr_analysis.py",
        "plot_utils.py",
        "mod_pdb.py"
    ]

    # Run each script
    for script in scripts:
        script_path = os.path.join(script_dir, script)
        subprocess.run(["python3", script_path], check=True)
