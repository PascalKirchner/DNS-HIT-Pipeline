# DNS-Hit
**DNS-Hit** is a software pipeline for the identification of easy-to-acquire ligands of a given protein. To do so, it combines the use of a machine learning model, [MoleGuLAR](https://github.com/devalab/MoleGuLAR) ([*J. Chem. Inf. Model.* 2021, 61, 12, 5815–5826](https://pubs.acs.org/doi/10.1021/acs.jcim.1c01341)), for the structure-based *de novo* generation of promising compounds and an analogue search with openbabel (https://github.com/openbabel/openbabel) and a library of commercially available compounds, such as a subset of the [Enamine REAL database](https://enamine.net/compound-collections/real-compounds), to identify similarly promising compounds that are easy to obtain.

## 1) Prerequisites

- **GPU:** A Nvidia GPU is required for the *de novo* generative model. The underlying ML model, [ReLeaSE](https://github.com/isayev/ReLeaSE) requires a [compute capability](https://developer.nvidia.com/cuda-gpus) of 3.5 or better. GPU is also necessary for running AutoDock GPU or AutoDock Vina GPU. We tested running multiple experiments in parallel which required about 4 GB of VRAM per experiment.
A Cuda version compatible with both the software and the available GPU architecture must be used. The package was tested with Cuda 11.7 on Ampere architecture.
- **RAM:** A higher amount of RAM is required for the analogue search step. We used library files with about 51 million entries each. This requires about 8 GB of RAM per experiment running in parallel.
- **Memory:** Molecular libraries for analogue search require significant disk space. With the library used in our experiment, containing about 956 million molecules, the required disk space was about 213 GB. Using an SSD is highly recommended, performing analogue search with the library files on an HDD took about ten times longer. 
- **Software:** Most software dependencies are installed with the provided docker package but to use it, [docker](https://docs.docker.com/get-started/get-docker/) needs to be installed on your system first. Additionally, [Nvidia Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) is required to use Cuda inside the Docker environment.

## 2) Installation

With the prerequisites met you can install the tool by following these steps:
1. Download the tool package from this git into a directory you want to use for your experiments (*e.g.* using git clone).
2. Open a terminal in that directory and build a docker image using the following command:<br>
`docker build -t name -f Dockerfile_MolePipeline .`  
Replace “name” with the name you want to use for your docker image.
3. Create a data directory for pipeline input and output. 
4. You can now run the docker image using a docker run command such as:  
`docker run -it --rm --runtime=nvidia --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v ./data_dir:/workspace/data_and_results name`  
Replace `name` with the image name from above and `./data_dir` with the path to the data directory from above.
5. To run an experiment, a molecular library for analogue search is required. Due to their size, libraries are not included in the download package. Building libraries is explained in section 3.

## 3) Adding molecular libraries

Due to their large size, molecular libraries are not included in the download package. Generating a library consists of the following steps:

1. Acquiring raw data with SMILES, *e.g.*, download it
2. Clean the data and dstribute it to generate files of easy-to-use size
3. Build fast search indices with openbabel  

The detailed explanations use a dataset from enamine as example which we used in our experiments. The process can, however, be easily adapted to generate other libraries.  

1. First, create a subdirectory in your data directory and download the relevant data into it. In our case, a subset of the Enamine REAL data base called `REAL Samples, 956M cpds, CXSMILES` was used and downloaded from the [Enamine REAL subset site](https://enamine.net/compound-collections/real-compounds/real-database-subsets) into a subdirectory called `enamine956`.
2. Second, prepare the data into multiple *.smi files of appropriate size containing SMILES and compound ids. In our case, the data was provided as cxsmiles and contained various additional pieces of information. We transformed the SMILES format and removed information other than SMILES and supplier ids. We also removed the header which can otherwise cause problems when generating fastsearch indices. We then distributed the total of about 956 million molecules into 19 files of about 51 million molecules. The maximum size of a library file depends primarily on RAM, with the order of magnitude used by us, each concurrent experiment required about 8 GB of RAM. A similar amount is also required while building the fastsearch indices.
3. Third, prepare fast search indices using the command `obabel name -ofs` for each library file, replacing `name` with the respecive *smi filename. The DNS-HIT dockerfile installs openbabel, so the conversion can be performed by running the docker and navigating to /workspace/data_and_results, then to the subdirectory for the library, then running the command there. 
<br>

Generating a fastsearch index this way is time-consuming, in our specific case it took about two to three hours per file. Depending on the available processing power and RAM, multiple conversions, can however be run in parallel. With the file size of 51 million entries we used, each conversion in parallel used less than 10 GB of RAM, so we could run seven conversions in parallel. The generation of fastsearch indices needs to be performed only once for any number of experiments using the same set of molecules. The total disk space required for the library used by us (19 *.smi files and 19 *.fs files) was about 213 GB which will change depending on the number of molecules included in the library.

## 4) Running experiments

Before running an experiment, set up the tool and generate an appropriate library for analogue search (sections 1-3). To run an experiment you need a *.pdb file for the protein target and a config file specifying various parameters. 
- **PDB file:** The pdb file needs to be prepared by removing any ligands, water molecules, and other non-protein contents. Conserved water molecules can be retained to consider them in the docking procedures. Once prepared, the PDB file needs to be placed in the data directory (see section 2).
- **Config file:** The config file is a simple txt file specifying the protein target to use, the docking box, the analogue search library, and various other parameters. The config file needs to be placed in the data directory (see section 2). 

The following is a config file used by us for an experiment with human protein tyrosine phosphatase 1B, [PDB 1bzc](https://www.rcsb.org/structure/1BZC):  

```
protein = 1bzc
analogue_mode = mixed
dock_de_novo = gpu4
dock_screen = gpu4
molecule_lib = enamine956
grid_center =(-19/56/17)
grid_size = (12/16/12)
bias =  (-21.334,59.323,11.198,acc)
bias = (-20.441,56.979,11.46,acc)
bias =  (-19.069,59.041,12.246,acc)
bias = (-15.621,50.987,19.617,acc)
bias =  (-18.539,51.832,19.352,don)
```

The parameters included are explained in the following:
- **protein** indicates the name of the PDB file to use, in this case it will look for a file named `1bzc.pdb`. 
- **analogue_mode** indicates how analogues are identified for *de novo* generated molecules. Recommended modes are `sub`, using openbabel substructure search and `mixed` with an additional filter to remove molecules with Tanimoto similarity < 0.5 between *de novo* generated molecule and analogue.
- **dock_de_novo** and **dock_screen** indicate which docking tools are to be used for docking *de novo* generated molecules and for screening the analogues. In this case, `gpu4` will cause AutoDock GPU to be used. Alternatively, `vina` will cause AutoDock Vina to be used. 
- **molecule_lib** indicates the directory name containing the library files with compounds for analogue search (see section 3). We named our library directory `enamine956` after the dataset used.
- **grid_center** and **grid_size** indicate the parameters for the docking grid box. Both are written as (*x*/*y*/*z*) and measured in A. 
- **bias** indicates a bias to use for biased docking. A bias is written as (*x*,*y*,*z*,*t*) with *t* either `acc` (hydrogen bond acceptor) or `don` (hydrogen bond donor). A bias will give an extra bonus on the docking score for an atom or group of the appropriate type at or close to the indicated position. Multiple biases can be used by adding one line for each as shown in the example file. Using biases required AutoDock GPU (`gpu4`) to be used as docking tool instead of AutoDock Vina (`vina`).  

**Starting an experiment:** To start an experiment you need to prepare the relevant files as described above, start the docker, navigate to `/workspace/Evaluation` (this happens automatically on docker startup so it's only necessary if you navigate other directories before), and run `python Main.py name` where `name` is the filename of your config file in the data directory, *e.g.* `config_1bzc.txt`. MoleGuLAR supports tracking molecular properties with [Weights & Biases](https://wandb.ai/site). Toward the beginning of the experiment, you need to indicate whether to use this tool. If you want to use it, you need an account and need to enter an API key. The software pipeline otherwise runs automatically. 

## 5) Result files
Results for each experiments are saved in a new directory in the data directory. The name of the subdirectory has the format `id_protein` where protein is the protein name of the pdb file used. id is a randomly chosen identifier for the experiment consisting of four upper case letters and / or numbers. This directory contains all the information related to the experiment. Most files created will contain the experiment identifier in their name.  
- **Config:** The config file used to start the experiment is copied here.
- **Generated:** This file contains SMILES and docking scores for the *de novo* generated compounds.
- **Dataset:** This file contains various pieces of information for all generated molecules, including predicted binding affinity (docking score), logP, Mw, number of heavy atoms, ligand efficiency, synthesizability score, molecular formula. Each molecule also gets assigned an id. These correspond to their order of generation.
- **Candidates:** This file contains a list of candidate molecules used for analogue search, including SMILES, ids, docking scores and Mw.
- **Analogues:** This file contains a list of all identified analogues, including candidate SMILES, analogue SMILES, analogue id, Tanimoto similarity between candidate and analogue, and the candidate id.
- **Screening:** This file contains the docking results for the analogue compounds, including docking scores, ids, and SMILES.
- **Log files:** The software will create log files listing parameters used. One log file is created when starting an experiment and one when finishing. If the software crashes, a log file will also be created normally. Log file names consist of `log + protein name + experiment id + timestamp`.
- **top_molecules.png** / **directory best:** An image of the best-scored molecules identified and a directory containing their docking poses.
- **diverse_molecules_80.png** / **directory diverse_80:** As above but excluding similar molecules. Molecules are again collected starting with top-scores but molecules are excluded if another molecule with Tanimoto similarity > 0.8 is already listed.
- **Directory de_novo_docking:** This directory contains files necessary for docking such as the protein pdbqt file (and grids if using AutoDock GPU), as well as the docking results from docking the *de novo* generated molecules.
- **Directory analogue_docking:** This directory contains files necessary for docking such as the protein pdbqt file (and grids if using AutoDock GPU), as well as the docking results from docking the identified analogue molecules.
- **Directory figures:** This directory contains some automatically generated figures to visualize properties of the *de novo* generated molecules such as regression plots for binding affinity, number of heavy atoms, and ligand efficiency, as well as a histogram of molecular weights, and a contour plot showing distributions of logP and QED. This was proposed by the MoleGuLAR paper as a means to verify that drug-like properties are maintained during optimization for binding affinity, by comparing molecules generated at the beginning and the end of optimization.
- **Directory statistics:** The software statistically evaluates results from identified analogues by comparing them to randomly chosen molecules and this directory contains the data related to that process. random_selection is a list of the randomly selected molecules, screening contains the docking results for those molecules. statistics_results contains information on both compound sets (mean, median, variance, Shapiro-Wilk test for normality) and the statistical results for comparison (Mann-Whitney U test for significance; effect size).

## 6) Modifying parameters
Beside the parameters shown in the example file in section 4, various other parameters can be modified when running experiments. Each parameter can be included as `name = value` where “name” is the parameter name indicated below and “value” is an appropriate value for that parameter.
- **logp_cutoff** (default: 5): Maximum logP used for filtering *de novo* molecules before candidate selection.
- **synthes_cutoff** (default: 0): Minimum synthesizability score used for filtering *de novo* molecules before candidate selection. Scores go from 0 to 1 with 1 being the best.
- **lower_weight** (default: 0) and upper_weight (default: 500): Molecular weight limits used for filtering *de novo* molecules before candidate selection.
- **n_of_candidates** (default: 100): Number of *de novo* molecules to use as candidates for analogue search. Candidates are the top n molecules remaining after filtering by logP, Mw, and synthesizability. 
- **run_id** and **start_at**: Used to restart an experiment at a later stage. If certain parameters should be changed or caused an error, they can be changed, the experiment can be started after the last step that was successfully completed or at an earlier step. To this end, data that is to be regenerated should be backed up and removed from the respective experiment directory. Both parameters need to be used in conjunction. `run_id` takes the four-character id of the experiment to restart (this is needed to find the relevant directory) and start_at indicates the step to start from. Possible option are: `denovo` (start from the very beginning), `dataset` (start at evaluating the *de novo* generated molecules), `visualization` (start at generating the figures for the properties of the *de novo* generated molecules), `selection` (start at candidate selection), `analogues` (start at analogue search), `screening` (start at docking of the analogues), `screen_fast` (start at docking of the analogues but skip preparation of the pdbqt files, *i.e.*, if they are already prepared), `best_analogues` (start at analyzing the analogue docking results / collecting best compounds), `validate_random` (start at statistical evaluation / comparison against randomly chosen molecules).
- **target_logP** (default: 2.5): Target value for the intermittent logP rewards in the MoleGuLAR generator model. 
- **gen_iterations** (default: 175): Number of iterations to run the generator model MoleGuLAR.
- **wandb_logging** (default: yes): The generator model can log molecular properties using Weights & Biases. Using this requires and account. This can also be managed when starting the experiment. To skip this entirely, set this option to “no” or “false”
- **grid_spacing** (default: 0.375): Resolution of the docking grid. 
- **de_novo_multi_mol** (default: 1): Generate multiple molecules (*e.g.*, 4) in each *de novo* step and select one with a higher number of heteroatoms.
- Weigths for vina scoring: Customize various weights for the scoring function when docking with AutoDock Vina. The parameters are: `vina_gauss1`, `vina_gauss2`, `vina_repulsion`, `vina_hydrophobic`, `vina_hydrogen`, `vina_rot`.

