import argparse
from datetime import datetime
import hashlib
from os import chdir, listdir, makedirs, path, remove, system
from random import choice, seed
import re
import shutil
import subprocess
import sys
import traceback

sys.path.append('../MoleGuLAR/Optimizer')
sys.path.append('../RAscore/RAscore')

import AnalogueSearcher
import CandidateSelector
import CandidateSelector2
import DatasetMaker
import GeneratorInterface
import RandomSampler
import Screening
import Visualization


parser = argparse.ArgumentParser(description='Perform hit identification for a protein target')
parser.add_argument('config', type=str, help='Generator config file')
config_arg = parser.parse_args().config


class Main:
    chars = "ABCDEFGHIJKLMNPQRSTUVWXYZ123456789"

    def __init__(self, config_file):
        chdir(path.abspath("/workspace/Evaluation"))
        self.config_path = f"../data_and_results/{config_file}"
        self.config_name = config_file
        self.config = {"n_of_candidates": 100,
                       "logp_cutoff": 5,
                       "synthes_cutoff": 0,
                       "lower_weight": 0,
                       "upper_weight": 500,
                       "data_dir": "/workspace/data_and_results",
                       "generator_dir": "/workspace/MoleGuLAR/Optimizer",
                       # "results_dir_name": "results",  # "results_dir": "/workspace/data_and_results/results",
                       "figures_dir_name": "figures",  # "/workspace/data_and_results/figures",
                       "statistics_dir_name": "statistics",
                       "de_novo_docking_dir_name": "de_novo_docking",
                       "analogue_docking_dir_name": "analogue_docking",
                       "generated_name": "generated",
                       "dataset_name": "dataset",
                       "candidates_name": "candidates",
                       "screening_results_name": "screening",
                       "analogues_name": "analogues",
                       "run_id": None,
                       "protein": "6LU7",
                       "full_id": None,
                       "seed": 777,
                       "target_logP": None,
                       "gen_iterations": 175,
                       "wandb_logging": "yes",
                       "gen_reward": "exponential",
                       "start_at": "",
                       "dock_de_novo": "vina",
                       "vina_weights": None,
                       "similarity_threshold1": 0.9,
                       "similarity_threshold2": 0.7,
                       "analogue_mode": "sim",
                       "molecule_lib": "zinclib",
                       "lib_prefix": None,
                       "dock_screen": "vina",
                       "grid_center": (0.0, 0.0, 0.0),
                       "grid_size": (20, 20, 20),
                       "grid_spacing": 0.375,
                       "n_of_candidates2": 50,
                       "analogue_min_het": 4,
                       "sim_thresholds": 0.8,
                       "score_threshold": 0,
                       "bias_data": None,
                       "use_bias": False,
                       "de_novo_multi_mol": 1,
                       "match_rings": True,
                       "analogue_cap": 50}
        self.load_config()
        self.set_random_id()
        self.starting_point = 0
        self.get_starting_point()
        self.main_dir = None
        self.setup_directories()
        self.copy_generator_config()
        self.generate_config()
        if self.config["use_bias"]:
            self.generate_bias_file()
        self.prepare_files_for_docking()
        print(self.config)
        self.initialize_rng()

    def initialize_rng(self):
        seed(self.config["seed"])

    def set_random_id(self):
        seed()
        if self.config["run_id"] is None:
            char_list = list(self.chars)
            random_id = str()
            for _ in range(4):
                random_id += choice(char_list)
            self.config["run_id"] = random_id

    def setup_directories(self):
        if self.config["full_id"]:
            main_name = self.config["full_id"]
        else:
            main_name = f'{self.config["run_id"]}_{self.config["protein"]}'
        main_path = path.abspath(path.join(self.config["data_dir"], main_name))
        self.main_dir = main_path
        analogues_filename = self.config["analogues_name"] + self.config["run_id"] + ".txt"
        candidates_filename = self.config["candidates_name"] + self.config["run_id"] + ".txt"
        dataset_filename = self.config["dataset_name"] + self.config["run_id"] + ".txt"
        generated_filename = self.config["generated_name"] + self.config["run_id"] + ".txt"
        screening_filename = self.config["screening_results_name"] + self.config["run_id"] + ".txt"
        self.config["analogues_file"] = path.join(main_path, analogues_filename)
        self.config["candidates_file"] = path.join(main_path, candidates_filename)
        self.config["dataset_file"] = path.join(main_path, dataset_filename)
        self.config["generated_file"] = path.join(main_path, generated_filename)
        self.config["screening_results_file"] = path.join(main_path, screening_filename)
        if self.starting_point == 0:
            makedirs(main_path, exist_ok=True)
        for directory in ("de_novo_docking_dir", "analogue_docking_dir", "figures_dir", "statistics_dir"):
            complete_path = path.abspath(path.join(main_path, self.config[directory + "_name"]))
            self.config[directory] = complete_path
            if self.starting_point == 0:
                makedirs(complete_path, exist_ok=True)
            elif not path.exists(complete_path):
                makedirs(complete_path, exist_ok=True)
        if not path.exists(f"{self.config['statistics_dir']}/docking"):
            makedirs(f"{self.config['statistics_dir']}/docking", exist_ok=True)

    def prepare_files_for_docking(self):
        python_path = "/workspace/mgltools_x86_64Linux2_1.5.7/bin/pythonsh"
        mgl_path = "/workspace/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24"
        script_dir = path.dirname(path.abspath(__file__))
        protein_ext = path.abspath(path.join(self.config["data_dir"], self.config["protein"] + ".pdb"))
        protein_run = path.abspath(path.join(script_dir, self.config["protein"] + ".pdb"))
        shutil.copy(protein_ext, protein_run)
        print("Copied protein structure.")
        subprocess.run(["obabel", f"{self.config['protein']}.pdb", "-O", f"{self.config['protein']}_withH.pdb",
                        "-h"])
        subprocess.run([python_path, f"{mgl_path}/prepare_receptor4.py", "-r", f"{self.config['protein']}_withH.pdb",
                        "-o", f"{self.config['protein']}.pdbqt", "-U", "nphs_lps", "-v"], check=True)
        if self.config["dock_de_novo"] == "gpu4":
            self.create_grids()
        for filename in listdir(script_dir):
            if filename.startswith(self.config["protein"]):
                file_path = path.join(script_dir, filename)
                if path.isfile(file_path):
                    shutil.copy(file_path, self.config["generator_dir"])
                    shutil.copy(file_path, self.config["analogue_docking_dir"])
                    shutil.copy(file_path, f"{self.config['statistics_dir']}/docking")
                    if not filename.endswith((".pdb", ".pdbqt", ".gpf")):
                        remove(file_path)

    def copy_generator_config(self):
        target_path = f"{self.main_dir}/{self.config_name}"
        shutil.copy(self.config_path, target_path)

    def generate_config(self):
        lines = list()
        coord_names = "xyz"
        for index, entry in enumerate(self.config["grid_center"]):
            line = f"center_{coord_names[index]} = {entry}\n"
            lines.append(line)
        for index, entry in enumerate(self.config["grid_size"]):
            line = f"size_{coord_names[index]} = {entry}\n"
            lines.append(line)
        lines.append(f"spacing = {self.config['grid_spacing']}\n")
        lines.append("exhaustiveness = 8\n")
        lines.append("cpu = 15\n")
        config_path = path.abspath(path.join(self.config["generator_dir"], self.config["protein"] + "_config.txt"))
        with open(config_path, "w+") as f:
            f.writelines(lines)

    @staticmethod
    def is_number(x):
        try:
            int(x)
            return True
        except ValueError:
            try:
                float(x)
                return True
            except ValueError:
                return False

    def load_config(self):
        vina_options = {
            "vina_gauss1": "weight_gauss1",
            "vina_gauss2": "weight_gauss2",
            "vina_repulsion": "weight_repulsion",
            "vina_hydrophobic": "weight_hydrophobic",
            "vina_hydrogen": "weight_hydrogen",
            "vina_rot": "weight_rot"
        }
        raw_config = None
        try:
            with open(self.config_path, "r") as f:
                raw_config = f.readlines()
            print("Config successfully loaded")
        except Exception as e:
            detailed_error = traceback.format_exc()
            print(f"During config loading, the following exception was caught: {type(e).__name__}")
            print(f"Detailed error information: {detailed_error}")
            print("Falling back to default options.")
        if raw_config is not None:
            for line in raw_config:
                if line[0] != "#":
                    parts = [part for part in re.split(r'\t|\n| |=', line) if part]
                    if len(parts) > 1:
                        if parts[0] in self.config.keys():
                            if type(self.config[parts[0]]) in (int, float) or parts[0] == "target_logP":
                                try:
                                    entry = float(parts[-1])
                                    if isinstance(self.config[parts[0]], int):
                                        self.config[parts[0]] = int(entry)
                                    else:
                                        self.config[parts[0]] = entry
                                except ValueError:
                                    print(f"Value {parts[-1]} is invalid for option {parts[0]}. Please enter a number.")
                            elif parts[-1].lower() in ("y", "yes", "true"):
                                self.config[parts[0]] = True
                            elif parts[-1].lower() in ("n", "no", "false"):
                                self.config[parts[0]] = False
                            elif parts[0] == "analogue_mode":
                                if parts[-1].lower() in ("sim", "sub", "mixed"):
                                    self.config["analogue_mode"] = parts[-1].lower()
                                else:
                                    default_msg = "Using default: similarity"
                                    print(f"Invalid option '{parts[-1]}' for analogue searching mode. {default_msg}")
                                    self.config["analogue_mode"] = "sim"
                            elif parts[0] in ("grid_center", "grid_size", "sim_thresholds"):
                                no_parentheses = parts[-1].strip("()")
                                coords = re.split(r'/|,', no_parentheses)
                                if len(coords) != 3 and parts[0] != "sim_thresholds":
                                    print(f"Invalid number of coordinates ({len(coords)}) for option {parts[0]}")
                                    print(f"Using default value {self.config[parts[0]]} for option {parts[0]}")
                                    continue
                                if not all(self.is_number(x) for x in coords):
                                    print(f"coordinates contain invalid non-numerical information.")
                                    print(f"Using default value {self.config[parts[0]]} for option {parts[0]}")
                                    continue
                                if parts == "grid_size" and any(int(x) < 1 for x in coords):
                                    print(f"""Grid is too small: At least one axis length is < 1. Using default grid 
size of (20/20/20).""")
                                    continue
                                self.config[parts[0]] = tuple(int(x) if x.isdigit() else float(x) for x in coords)
                            else:
                                self.config[parts[0]] = parts[-1]
                        elif parts[0][:4] == "vina":
                            if parts[0] in vina_options.keys():
                                if self.config["vina_weights"] is None:
                                    self.config["vina_weights"] = dict()
                                try:
                                    weight = float(parts[-1])
                                    self.config["vina_weights"][vina_options[parts[0]]] = weight
                                except ValueError:
                                    print(f"Invalid vina weight {parts[-1]}")
                        elif parts[0] == "bias":
                            if self.config["bias_data"] is None:
                                self.config["bias_data"] = list()
                            entries = parts[-1].strip("()")
                            entries_clean = re.split(r'/|,', entries)
                            if len(entries_clean) != 4:
                                print(f"Invalid number of parameters ({len(entries_clean)}) for bias entry.")
                                print("Each entry needs x, y, and z coordinates for its center and a bias type")
                                continue
                            if not all(self.is_number(x) for x in entries_clean[:3]):
                                print(f"coordinates for bias contain invalid non-numerical information.")
                                continue
                            bias_entry = [float(c) for c in entries_clean[:3]]
                            bias_entry.append(entries_clean[3])
                            self.config["bias_data"].append(bias_entry)
                        else:
                            print(f"Unknown option {parts[0]}.")
        if type(self.config["sim_thresholds"]) == float:
            self.config["sim_thresholds"] = [self.config["sim_thresholds"]]
        if self.config["bias_data"] is not None:
            self.config["use_bias"] = True

    def get_starting_point(self):
        starts = {
            "denovo": 0,
            "dataset": 1,
            "visualization": 2,
            "selection": 3,
            "analogues": 4,
            "screening": 5,
            "screen_fast": 6,
            "best_analogues": 7,
            "validate_random": 8
        }
        if self.config["start_at"] in starts.keys():
            self.starting_point = starts[self.config["start_at"]]

    def run_generator(self):
        print(f"Started running de novo generation (protein {self.config['protein']}; run {self.config['run_id']})")
        logp_threshold = 2.5 if self.config["target_logP"] is None else self.config["target_logP"]
        generator_instance = GeneratorInterface.Generator(self.config,
                                                          switch=True,
                                                          logp_threshold=logp_threshold)
        generator_instance.run_de_novo()
        current_file_directory = path.dirname(path.abspath(__file__))
        chdir(current_file_directory)
        self.copy_de_novo_docking()
        print(f"Finished running de novo generation (protein {self.config['protein']}; run {self.config['run_id']})")

    def generate_and_evaluate(self):
        print("Protein: ", self.config["protein"])
        self.write_log(early=True)
        most_recent_step = "Startup"
        error_data = None
        try:
            if self.starting_point == 0:
                self.run_generator()
                most_recent_step = "De novo generation"
            if self.starting_point <= 1:
                self.make_dataset()
                most_recent_step = "Dataset generation"
            if self.starting_point <= 2:
                self.make_figures()
                most_recent_step = "Visualization of de novo results"
            if self.starting_point <= 3:
                self.select_candidates()
                most_recent_step = "Candidate selection"
            if self.starting_point <= 4:
                self.search_analogues()
                most_recent_step = "Analogue search"
            if self.starting_point <= 5:
                self.screen_analogues(True)
                most_recent_step = "Analogue screening (molecule prepartion included: yes)"
            elif self.starting_point == 6:
                self.screen_analogues(False)
                most_recent_step = "Analogue screening (molecule prepartion included: no)"
            if self.starting_point <= 7:
                self.select_best_analogues()
                most_recent_step = "Identification of top molecules"
            if self.starting_point <= 8:
                self.validate_with_random()
                most_recent_step = "Validation against random molecules"
        except Exception as e:
            error_data = self.log_error(str(e))
            print(self.log_error(str(e)))
        self.write_log(error_output=[error_data, most_recent_step])

    @staticmethod
    def log_error(error_message):
        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        error_content = f"""[ERROR]
Timestamp: {timestamp}
Message: {error_message}
Traceback:
{traceback.format_exc()}
 ---------------------------
"""
        return error_content

    def make_dataset(self):
        print(f"Started building molecule data set (protein {self.config['protein']}; run {self.config['run_id']})")
        dtsm = DatasetMaker.DatasetMaker(self.config["generated_file"], self.config["dataset_file"])
        dtsm.make_dataset()
        print(f"Finished building molecule data set (protein {self.config['protein']}; run {self.config['run_id']})")

    def select_candidates(self):
        print(f"Started candidate selection (protein {self.config['protein']}; run {self.config['run_id']})")
        cs = CandidateSelector.CandidateSelector(self.config)
        cs.handle_complete_selection()
        print(f"Finished candidate selection (protein {self.config['protein']}; run {self.config['run_id']})")

    def search_analogues(self):
        print(f"Started analogue search (protein {self.config['protein']}; run {self.config['run_id']})")
        complete_lib_path = path.join("/workspace/data_and_results/", self.config["molecule_lib"])
        zs = AnalogueSearcher.AnalogueSearcher(self.config["candidates_file"], self.config["analogues_file"],
                                               complete_lib_path,
                                               (self.config["similarity_threshold1"],
                                                self.config["similarity_threshold2"]),
                                               self.config["analogue_mode"],
                                               self.config["lib_prefix"],
                                               self.config["analogue_cap"],
                                               self.config["match_rings"],
                                               self.config["n_of_candidates"])
        zs.evaluate_list()
        print(f"Finished analogue search (protein {self.config['protein']}; run {self.config['run_id']})")

    def screen_analogues(self, full):
        print(f"Started analogues library screening (protein {self.config['protein']}; run {self.config['run_id']})")
        screening_instance = Screening.Screening(self.config["analogue_docking_dir"],
                                                 self.config["screening_results_file"],
                                                 [self.config["grid_center"],
                                                  self.config["grid_size"],
                                                  self.config["grid_spacing"]],
                                                 self.config["analogues_file"],
                                                 self.config["protein"],
                                                 self.config["run_id"],
                                                 self.config["dock_screen"],
                                                 self.config["vina_weights"],
                                                 self.config["seed"])
        # source_dir = path.abspath("/workspace/data_and_results")
        target_dir = self.config["analogue_docking_dir"]
        screening_instance.generate_config_file(target_dir, self.config["protein"] + "_config.txt")
        # screening_instance.copy_protein_pdbqt(source_dir, target_dir, self.config["protein"])
        print(self.config["analogues_file"])
        screening_instance.dock_all_smiles(full)
        print(f"Finished analogues library screening (protein {self.config['protein']}; run {self.config['run_id']})")

    def make_figures(self):
        print(f"Started visualization (protein {self.config['protein']}; run {self.config['run_id']})")
        vis = Visualization.Visualizer(self.config["figures_dir"], self.config["dataset_file"], self.config["run_id"])
        vis.complete_evaluation()
        print(f"Finished visualization (protein {self.config['protein']}; run {self.config['run_id']})")

    def write_log(self, error_output=None, early=False):
        starts = {0: "de novo generation with MoleGuLAR", 1: "de novo molecule evaluation and dataset generation",
                  2: "visualization / generation of figures", 3: "de novo molecule filtering and candidate selection",
                  4: "analogue search for candidates", 5: "analogue screening (full with preparation)",
                  6: "analogue screening (fast / molecules already prepared)", 7: "Top analogues identification",
                  8: "validation against random molecules"}
        line1 = f"Log for protein {self.config['protein']} and run {self.config['run_id']}.\n"
        line2 = f"Docking performed with center {self.config['grid_center']} and box size {self.config['grid_size']}.\n"
        line3 = f"Process was started at {starts[self.starting_point]}.\n"
        log_lines = list()
        if early:
            log_lines.append("This log was created before running the pipeline to document parameters.")
        log_lines.extend([line1, line2, line3])
        if self.starting_point == 0:
            information = f"rewards={self.config['gen_reward']}, target for logP {self.config['target_logP']}" \
                          f" and {self.config['gen_iterations']}.\n"
            log_lines.append("De novo generation with MoleGuLAR uses " + information)
        if self.starting_point <= 3:
            information = f"Molecule filtering uses logP < {self.config['logp_cutoff']}, synthesizability " \
                          f"> {self.config['synthes_cutoff']}, and {self.config['lower_weight']} < Mw " \
                          f"< {self.config['upper_weight']}.\n"
            log_lines.append(information)
        if self.starting_point <= 4:
            modes = {"sim": "Tanimoto similarity", "sub": "Bemis-Murcko scaffold substructure",
                     "mixed": "Bemis-Murcko substructure plus subsequent Tanimoto filtering"}
            mode = modes[self.config["analogue_mode"]]
            information = f"Analogue identification is performed using the library {self.config['molecule_lib']} " \
                          f"with {self.config['n_of_candidates']} candidates and {mode} mode.\n"
            log_lines.append(information)
        log_lines.append(f"Screening is performed with {self.config['dock_screen']}.\n")
        if error_output is not None:
            log_lines.append(f"Process crashed at step: {error_output[1]}")
            log_lines.append(error_output[1])
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        log_filename = f"log_{self.config['protein']}_{self.config['run_id']}_{timestamp}.txt"
        log_path = path.join(self.main_dir, log_filename)
        with open(log_path, 'w') as f:
            f.writelines(log_lines)
        print(f"Log file created: {log_filename}")

    def select_best_analogues(self):
        print("Started selecting best analogues.")
        analogue_selector = CandidateSelector2.CandidateSelector2(self.main_dir, self.config["protein"],
                                                                  self.config["run_id"],
                                                                  self.config["screening_results_file"],
                                                                  self.config["n_of_candidates2"],
                                                                  self.config["analogue_min_het"],
                                                                  self.config["score_threshold"],
                                                                  self.config["sim_thresholds"])
        analogue_selector.run()
        print("Finished selecting best analogues.")

    def validate_with_random(self):
        print("Started validation against randomly selected molecules.")
        complete_lib_path = path.join("/workspace/data_and_results/", self.config["molecule_lib"])
        with open(self.config["analogues_file"], 'r') as f:
            sample_size = sum(1 for _ in f)
        sampler = RandomSampler.RandomSampler(complete_lib_path, sample_size, self.config["protein"],
                                              self.config["run_id"], self.config["statistics_dir"],
                                              [self.config["grid_center"], self.config["grid_size"],
                                               self.config["grid_spacing"]], self.config["dock_screen"],
                                              self.config["vina_weights"], self.config["seed"])
        random_mols = sampler.run()
        print("Finished validation against randomly selected molecules.")
        print("Started statistical evaluation")
        statistics = RandomSampler.Statistics(random_mols, self.config["screening_results_file"],
                                              self.config["statistics_dir"])
        statistics.complete_evaluation()
        print("Finished statistical evaluation")

    def generate_gpf(self):
        python_path = "/workspace/mgltools_x86_64Linux2_1.5.7/bin/pythonsh"
        mgl_path = "/workspace/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24"
        gpf_creation_parameters = [python_path, f"{mgl_path}/prepare_gpf4.py", "-r", f"{self.config['protein']}.pdbqt",
                                   "-l", "test_molecule.pdbqt"]
        subprocess.run(gpf_creation_parameters, check=True)
        self.modify_gpf()

    def modify_gpf(self):
        with open(f"{self.config['protein']}.gpf", "r") as f:
            lines = f.readlines()
        new_lines = list()
        map_types = ["A", "C", "NA", "OA", "N", "P", "S", "SA", "HD", "F", "Cl"]
        maps_added = False
        for line in lines:
            if "npts" in line:
                s = [int(round(x/self.config["grid_spacing"], 0)) for x in self.config["grid_size"]]
                size_line = f"npts {s[0]} {s[1]} {s[2]}\n"
                new_lines.append(size_line)
            elif "gridcenter" in line:
                c = self.config["grid_center"]
                center_line = f"gridcenter {c[0]} {c[1]} {c[2]}\n"
                new_lines.append(center_line)
            elif "ligand_types" in line:
                type_line = "ligand_types A C NA OA N P S SA HD F Cl\n"
                new_lines.append(type_line)
            elif "atom-specific" in line:
                if not maps_added:
                    for map_type in map_types:
                        map_line = f"map {self.config['protein']}.{map_type}.map\n"
                        new_lines.append(map_line)
                    maps_added = True
            else:
                new_lines.append(line)
        with open(f"{self.config['protein']}.gpf", "w") as f:
            f.writelines(new_lines)

    def create_grids(self):
        python_path = "/workspace/mgltools_x86_64Linux2_1.5.7/bin/pythonsh"
        mgl_path = "/workspace/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24"
        preparebias_path = path.join(mgl_path, "contrib/adbias/prepare_bias.py")
        self.generate_gpf()
        subprocess.run(["autogrid4", "-p", f"{self.config['protein']}.gpf"], check=True)
        if self.config["use_bias"]:
            subprocess.run([python_path, preparebias_path, "-b", f"{self.config['protein']}_bias.bpf", "-g",
                            f"{self.config['protein']}.gpf"], check=True)
            self.modify_fld()

    def generate_bias_file(self):
        lines = list()
        for b in self.config["bias_data"]:
            radius = 0.8 if b[3] == "acc" else 0.6
            line = f"{b[0]} {b[1]} {b[2]} -2.00 {radius} {b[3]}\n"
            lines.append(line)
        with open(self.config["protein"] + "_bias.bpf", "w+") as f:
            f.writelines(lines)

    def modify_fld(self):
        use_acc = False
        use_don = False
        for entry in self.config["bias_data"]:
            if entry[3] == "acc":
                use_acc = True
            elif entry[3] == "don":
                use_don = True
        with open(f"{self.config['protein']}.maps.fld", "r") as f:
            lines = f.readlines()
        new_lines = list()
        for line in lines:
            if ".HD.map" in line and use_don:
                line = line.replace(".HD.map", ".HD.biased.map")
            if ".NA.map" in line and use_acc:
                line = line.replace(".NA.map", ".NA.biased.map")
            if ".OA.map" in line and use_acc:
                line = line.replace(".OA.map", ".OA.biased.map")
            new_lines.append(line)
        with open(f"{self.config['protein']}.maps.fld", "w") as f:
            f.writelines(new_lines)

    def copy_de_novo_docking(self):
        starting_directory = f"{self.config['generator_dir']}/logs_exponential"
        target_directory = f"{self.main_dir }/de_novo_docking"
        for filename in listdir(starting_directory):
            if filename.endswith(".pdbqt"):
                source_path = path.join(starting_directory, filename)
                destination_path = path.join(target_directory, filename)
                shutil.copy(source_path, destination_path)


pipeline = Main(config_arg)
pipeline.generate_and_evaluate()
