# binary-interfaces
## Analysing protein-protein interfaces and their folding patterns using a block-wise WSME model. 

The workflow folder contains the python scripts used for the project. The program takes PDB format files as input. To run the program on any protein of your choice, add the PDB file of the protein in to the 'input_pdb_files' folder and run the orchestrator.py script from the workflow folder. The generated data from the analysis of the outputs of the WSME model is stored in the analysis_output folder

The plots are in the plots folder. Four types of plots have been generated for each structure. 
1. blockwise free energy plot
2. Free energy for the whole interface 
3. Free energies of all the microstates in the minimum energy macrostate
4. Free energy for the minimum energy state macrostate (for all temperatures across all microstates, as a heatmap)

The normalized free energy values are imprinted onto the PDB structure ass b-factor values in a modded PDB file. Find these files in the mod_pdb_files folder, once having complete running the program successfully. 

The jupyter notebook free_energy.ipynb is a sandbox notebook (can be ignored, or used for testing). 
