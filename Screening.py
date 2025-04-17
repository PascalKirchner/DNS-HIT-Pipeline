from os import chdir, path, popen, system, makedirs, listdir
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles
import re
import shutil
import subprocess
import traceback


class Screening:
    mgl_path = "/workspace/mgltools_x86_64Linux2_1.5.6/bin/pythonsh /workspace/mgltools_x86_64Linux2_1.5.6/" + \
               "MGLToolsPckgs/AutoDockTools/Utilities24"

    def __init__(self, docking_dir, screening_results_file, grid_info, smiles_file, protein, run_id,
                 docking_tool="vina", vina_weights=None, seed=777):
        chdir(docking_dir)
        self.docking_dir = docking_dir
        self.screening_results = screening_results_file
        self.docking_file_index = 0
        self.docking_files = dict()
        self.smiles_file = smiles_file
        self.protein_pdbqt = None
        self.prot = protein
        self.grid_info = grid_info
        self.vina_config = None
        self.vina_weights = vina_weights
        self.docking_tool = docking_tool
        self.docking_results = dict()
        self.seed = seed
        self.run_id = run_id
        self.longest_id = 0

    def get_docking_index(self):
        str_index = str(self.docking_file_index)
        zeros_to_add = 6-len(str(str_index))
        current_index = zeros_to_add * "0" + str_index
        self.docking_file_index += 1
        return current_index

    def dock_all_smiles(self, full=True):
        print(f"Docking tool: {self.docking_tool}")
        if full:
            data = self.load_smiles()[:50]
            print(f"Loaded {len(data)} molecules for screening")
            self.prepare_data(data)
        self.reload_preliminary_data()
        if self.docking_tool == "vinagpu":
            self.dock_with_vinagpu()
        elif self.docking_tool == "gpu4":
            self.dock_with_4gpu()
        else:
            self.dock_with_vina()
        self.filter_and_sort_by_score()
        self.save_screening_results()

    def load_smiles(self):
        loaded_data = list()
        with open(self.smiles_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            parts = re.split("\t|\n| ", line)
            parts_clean = [p for p in parts if p != ""]
            loaded_data.append(parts_clean)
        return loaded_data

    def prepare_data(self, data):
        n_total = len(data)
        for n, entry in enumerate(data):
            print("Entry: ", entry)
            try:
                if entry[0] != "":
                    next_index = self.get_docking_index()
                    if len(entry) == 2:
                        smiles = entry[0].strip("'")
                        identifier = entry[1].strip("'")
                    else:
                        smiles = entry[1].strip("'")
                        identifier = entry[2].strip("'")
                    self.docking_results[next_index] = {"smiles": smiles,
                                                        "identifier": identifier,
                                                        }
                    self.longest_id = max(len(d["identifier"]) for d in list(self.docking_results.values()))
                    self.prepare_pdb_file(smiles, identifier)
                    self.prepare_pdbqt(identifier)
            except Exception as e:
                tr = traceback.format_exc()
                print(f"Docking preparation error @ {entry[0]}: {tr}")
            print(f"Prepared {n+1} / {n_total}")
        self.save_screening_results(preliminary=True)
        self.docking_results = dict()

    def reload_preliminary_data(self):
        with open(self.screening_results, "r") as f:
            data_lines = f.readlines()
        for line in data_lines:
            parts_clean = [part.strip() for part in line.split() if part]
            self.docking_results[parts_clean[0]] = {"smiles": parts_clean[3], "identifier": parts_clean[2]}
        if self.longest_id == 0:
            self.longest_id = self.longest_id = max(len(d["identifier"]) for d in list(self.docking_results.values()))
        print(f"Finished loading molecules for docking: {len(list(self.docking_results.keys()))} items")

    def move_all_pdbqts_and_return_dir(self):
        pdb_dir = path.join(self.docking_dir, "docking_input")
        makedirs(pdb_dir, exist_ok=True)
        for filename in listdir(self.docking_dir):
            if filename.endswith(".pdbqt"):
                src_path = path.join(self.docking_dir, filename)
                dest_path = path.join(pdb_dir, filename)
                shutil.move(src_path, dest_path)
        return pdb_dir

    def prepare_pdb_file(self, smile, mol_id):
        """Create ligand files in pdb format from molecular smiles.

        Using rdkit, generate a molecule representation, then save it as pdb.
        """
        mol = Chem.MolFromSmiles(smile)
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=50)
        except Exception as e:
            print("Embedding error, skipping embedding")
        print(f"Run: {self.run_id}")
        rdmolfiles.MolToPDBFile(mol, f"{self.docking_dir}/{mol_id}_{self.run_id}.pdb")

    def prepare_pdbqt(self, index):
        """Transform a pdb files to dock into pdbqt file

        Use prepare ligand script from mgltools. The complex path is necessary because the script doesn't run on
        current python.
        """
        system(
            f"{self.mgl_path}/prepare_ligand4.py -l {self.docking_dir}/{index}_{self.run_id}.pdb -o {self.docking_dir}/{str(index)}_{self.run_id}.pdbqt")

    def copy_protein_pdbqt(self, directory1, directory2, protein):
        full_source_path = path.abspath(path.join(directory1, protein + ".pdbqt"))
        full_target_path = path.abspath(path.join(directory2, protein + ".pdbqt"))
        if not path.exists(full_source_path):
            pdb_source_path = path.abspath(path.join(directory1, protein + ".pdb"))
            pdbqt_target_path = path.abspath(path.join(directory1, protein + ".pdbqt"))
            parameter_list = [f"{self.mgl_path}/prepare_receptor4.py", "-r", pdb_source_path, "-o", pdbqt_target_path,
                              "-U", "nphs_lps", "-v"]
            subprocess.run(parameter_list, check=True)
        shutil.copy(full_source_path, full_target_path)
        self.protein_pdbqt = full_target_path
        self.prot = protein
        print("Copied protein structure.")

    def generate_config_file(self, directory, file_name):
        config_path = path.abspath(path.join(directory, file_name))
        lines = list()
        coord_names = "xyz"
        for index, entry in enumerate(self.grid_info[0]):
            line = f"center_{coord_names[index]} = {entry}\n"
            lines.append(line)
        for index, entry in enumerate(self.grid_info[1]):
            line = f"size_{coord_names[index]} = {entry}\n"
            lines.append(line)
        lines.append("thread = 8000\n")
        with open(config_path, "w+") as f:
            f.writelines(lines)
        self.vina_config = config_path

    def dock_with_vinagpu(self):
        print("Started docking with Vina GPU")
        input_path = self.move_all_pdbqts_and_return_dir()
        output_path = path.join(self.docking_dir, "docking_output")
        protein_pdbqt = path.join(input_path, path.basename(self.protein_pdbqt))
        if not path.exists(output_path):
            makedirs(output_path)
        cmd_part1 = f"AutoDock-Vina-GPU-2-1 --config {self.vina_config} --receptor {protein_pdbqt} --seed {self.seed} "
        cmd_part2 = f"--ligand_directory {input_path} "
        cmd_part3 = f"--output_directory {output_path}"
        complete_cmd = cmd_part1 + cmd_part2 + cmd_part3
        system(complete_cmd)
        self.readout_results(output_path)

    def readout_results(self, output_path):
        filename_wrong = 0
        print(f"Reading docking results: {len(list(self.docking_results.keys()))} items")
        for key, value in self.docking_results.items():
            try:
                file_path = path.join(output_path, value["identifier"] + "_" + self.run_id + "_out.pdbqt")
                if path.isfile(file_path):
                    with open(file_path, 'r') as file:
                        for line in file:
                            if line.startswith("REMARK VINA RESULT"):
                                parts = re.split("\t|\n| ", line)
                                parts_clean = [p for p in parts if p != ""]
                                if len(parts_clean) > 1:
                                    try:
                                        value = round(float(parts_clean[3]), 3)
                                        self.docking_results[key]["score"] = value
                                    except ValueError:
                                        print(f"Warning: Could not convert '{parts_clean[1]}'")
                                break
                else:
                    filename_wrong += 1
            except Exception as e:
                print(f"There was a problem with docking compound {value['identifier']}")
                tr = traceback.format_exc()
                print(f"Docking error @ {value['identifier']}: {tr}")
        print(f"Filename wrong: {filename_wrong}")

    def dock_with_vina(self):
        print("Started docking with Vina")
        vina_cmd1 = f"vina --config {self.vina_config} --receptor {self.protein_pdbqt} --seed {self.seed} "
        for number in self.docking_results.keys():
            identifier = self.docking_results[number]["identifier"]
            try:
                self.dock_one_vina(number, vina_cmd1)
            except Exception as e:
                print(f"There was a problem with docking compound {identifier}")
                tr = traceback.format_exc()
                print(f"Docking error @ {identifier}: {tr}")

    def dock_one_vina(self, number, cmd1):
        identifier = self.docking_results[number]["identifier"]
        if path.exists(f"{self.docking_dir}/{str(identifier)}_{self.run_id}.pdbqt"):
            vina_command_part_2 = f"--ligand {self.docking_dir}/{str(identifier)}_{self.run_id}.pdbqt "
            complete_vina_command = cmd1 + vina_command_part_2
            if self.vina_weights is not None:
                for key, value in self.vina_weights.items():
                    complete_vina_command += f"--{key} {value}"
            system(complete_vina_command)
            readout_cmd = f"cat {self.docking_dir}/{str(identifier)}_{self.run_id}_out.pdbqt | grep -i RESULT | tr -s '\t' ' ' | cut -d ' ' -f 4 | head -n1"
            stream = popen(readout_cmd)
            output = float(stream.read().strip())
            self.docking_results[number]["score"] = round(output, 3)
        else:
            print(f"Skipped docking compound {str(identifier)} because it could apparently not be correctly prepared")

    def dock_with_4gpu(self):
        print("Started docking with Autodock GPU")
        for number in self.docking_results.keys():
            try:
                self.dock_one_4gpu(number)
            except Exception as e:
                print(f"There was a problem with docking compound {number}")
                tr = traceback.format_exc()
                print(f"Docking error @ {number}: {tr}")

    def dock_one_4gpu(self, number):
        identifier = self.docking_results[number]["identifier"]
        if path.exists(f"{self.docking_dir}/{str(identifier)}_{self.run_id}.pdbqt"):
            parameters_list = ["/workspace/AutoDock-GPU/bin/autodock_gpu_128wi", "-ffile", f"{self.prot}.maps.fld",
                               "-lfile", f"{self.docking_dir}/{str(identifier)}_{self.run_id}.pdbqt", "-resnam",
                               f"{self.docking_dir}/{str(identifier)}_{self.run_id}", "--gbest", "1", "-nrun", "15",
                               "-devnum" "1"]
            subprocess.run(parameters_list, check=True)
            cmd = f"cat {self.docking_dir}/{str(identifier)}_{self.run_id}.dlg | grep -i ranking | tr -s '\t' ' ' | cut -d ' ' -f 5 | head -n1"
            stream = popen(cmd)
            output = float(stream.read().strip())
            self.docking_results[number]["score"] = round(output, 3)
        else:
            print(f"Skipped docking compound {str(identifier)} because it could apparently not be correctly prepared")

    def filter_and_sort_by_score(self):
        missing_score = 0
        filtered_results = dict()
        for key, value in self.docking_results.items():
            if "score" in value.keys():
                filtered_results[key] = value
            else:
                missing_score += 1
        print("Scores missing due to docking errors: ", missing_score)
        sorted_screening_items = sorted(filtered_results.items(), key=lambda x: x[1]["score"])
        sorted_screening_dict = {k: v for k, v in sorted_screening_items}
        self.docking_results = sorted_screening_dict

    def save_screening_results(self, preliminary=False):
        lines = list()
        for key, value in self.docking_results.items():
            score = str(round(value["score"], 3)) if not preliminary else "n/a"
            line = str()
            line += key
            line += "  "
            line += self.make_block(score, 8)
            id_length = self.longest_id // 4 * 4 + 6
            line += self.make_block(value["identifier"], id_length)
            line += "  "
            line += value["smiles"]
            line += "\n"
            lines.append(line)
        with open(self.screening_results, "w") as f:
            f.writelines(lines)

    @staticmethod
    def make_block(text, size, align="r"):
        text = str(text)
        if text[-1] == "\n":
            text = text[:-1]
        spaces = size - len(text)
        if align == "r":
            block = spaces * " " + text
        else:
            block = text + spaces * " "
        return block
