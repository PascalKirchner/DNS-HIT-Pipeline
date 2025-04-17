import numpy as np
from os import makedirs, path
import re
import shutil

from rdkit import Chem
from rdkit.Chem import Draw, rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold


class CandidateSelector2:
    def __init__(self, working_dir, protein_id, run_id, screening_file, n_top, min_het, threshold=0, sim_thresholds=None):
        self.protein_id = protein_id
        self.run_id = run_id
        self.working_dir = working_dir
        self.screening_file = screening_file
        self.n_top = n_top
        self.min_het = min_het
        self.threshold = threshold
        self.sim_thresholds = list()
        self.set_sim_thresholds(sim_thresholds)
        self.data = list()
        self.scaffolds = list()
        self.top_smiles = list()
        self.top_ids = list()
        self.diverse_compounds = dict()
        self.diverse_top = list()
        self.next_index = 0

    def set_sim_thresholds(self, thrsh):
        if thrsh is None or len(thrsh) == 0:
            self.sim_thresholds = [0.75]
        elif all(isinstance(x, float) and 0 <= x <= 1 for x in thrsh):
            self.sim_thresholds = thrsh
        else:
            self.sim_thresholds = [0.75]

    def run(self):
        self.load_molecule_info()
        self.set_threshold()
        self.extract_scaffolds()
        self.identify_top()
        self.visualize_top("top_molecules")
        self.collect_top_pdbqts(self.top_ids, "best")
        for entry in self.sim_thresholds:
            self.diverse_compounds = dict()
            self.diverse_top = list()
            self.top_smiles = list()
            self.identify_diverse(self.threshold, self.min_het, entry)
            self.identify_top_diverse()
            id_list = [e2["id"] for e2 in self.diverse_top]
            sim_thresh_number = int(entry * 100)
            self.collect_top_pdbqts(id_list, f"diverse_{sim_thresh_number}")
            self.top_smiles = [e2["smiles"] for e2 in self.diverse_top]
            self.visualize_top(f"diverse_molecules_{sim_thresh_number}")
            self.write_top_smiles(f"{self.working_dir}/diverse_{sim_thresh_number}/smiles_and_ids.txt")

    def load_molecule_info(self):
        with open(self.screening_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            if "Molecular Weight" in line:
                continue
            parts = re.split("\t|\n| |;", line)
            parts_clean = [p for p in parts if p != ""]
            self.data.append({"smiles": parts_clean[3],
                              "score": float(parts_clean[1]),
                              "id": parts_clean[2]
                              })
        print("Loaded smiles")
        self.check_molecule_info([entry['id'] for entry in self.data])

    def set_threshold(self):
        scores = [d["score"] for d in self.data]
        med = np.median(scores)
        self.threshold = round(med.item(), 1)

    def check_molecule_info(self, data_list):
        input_directory_1 = f"{self.working_dir}/analogue_docking/docking_output"
        input_directory_2 = f"{self.working_dir}/analogue_docking"
        missing = 0
        for entry in data_list:
            source_filename_1 = path.join(input_directory_1, f"{entry}_{self.run_id}_out.pdbqt")
            source_filename_2 = path.join(input_directory_2, f"{entry}_{self.run_id}.pdbqt")
            if not path.exists(source_filename_1):
                if not path.exists(source_filename_2):
                    print(f"Missing: {entry}")
                    missing += 1
        print(f"Missing total: {missing}")

    def extract_scaffolds(self):
        """Extracts Bemis-Murcko scaffolds from a list of SMILES strings."""
        for index, entry in enumerate(self.data):
            smiles = entry["smiles"]
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                if scaffold:
                    re_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=True)
                    if re_smiles not in self.scaffolds:
                        self.scaffolds.append(re_smiles)
        print("Extracted scaffolds")

    def visualize_scaffolds(self, output_dir, per_row=5, mol_size=(200, 200), per_file=60):
        """Displays the scaffolds using RDKit's drawing functions."""
        if not path.exists(output_dir):
            makedirs(output_dir)
        for i in range(0, len(self.scaffolds), per_file):
            chunk = self.scaffolds[i:i + per_file]
            img = Draw.MolsToGridImage(chunk, molsPerRow=per_row, subImgSize=mol_size)
            img_path = path.join(output_dir, f"scaffolds_{i // per_file + 1}.png")
            img.save(img_path)
            print(f"Saved: {img_path}")

    def identify_top(self):
        for entry in self.data:
            if entry["smiles"] not in self.top_smiles:
                self.top_smiles.append(entry["smiles"])
                self.top_ids.append(entry["id"])
            if len(self.top_smiles) == self.n_top:
                break

    def identify_diverse(self, threshold, min_het, sim_thresh):
        for molecule in self.data:
            if molecule["score"] > threshold:
                continue
            smiles1 = molecule["smiles"]
            mol1 = Chem.MolFromSmiles(smiles1)
            n_het = sum(1 for atom in mol1.GetAtoms() if atom.GetAtomicNum() not in [1, 6, 9, 17, 35, 53])
            if n_het < min_het:
                continue
            heavy1 = sum(1 for atom in mol1.GetAtoms() if atom.GetAtomicNum() > 1)
            found_similar_cluster = False
            for key, value in self.diverse_compounds.items():
                smiles2 = value[0]["smiles"]
                mol2 = Chem.MolFromSmiles(smiles2)
                heavy2 = sum(1 for atom in mol2.GetAtoms() if atom.GetAtomicNum() > 1)
                mcs = rdFMCS.FindMCS([mol1, mol2], matchValences=False, ringMatchesRingOnly=False)
                mcs_smarts = mcs.smartsString
                mcs_mol = Chem.MolFromSmarts(mcs_smarts)
                heavy_common = sum(1 for atom in mcs_mol.GetAtoms() if atom.GetAtomicNum() > 1)
                percentage_common = round(2.0 * heavy_common / (heavy1 + heavy2), 1)
                if percentage_common > sim_thresh:
                    self.diverse_compounds[key].append(molecule)
                    found_similar_cluster = True
                    break
            if not found_similar_cluster:
                self.diverse_compounds[self.next_index] = [molecule]
                self.next_index += 1

    def identify_top_diverse(self):
        diverse_top = list()
        for entry in self.diverse_compounds.values():
            if len(entry) == 1:
                diverse_top.extend(entry)
                continue
            top = None
            for molecule in entry:
                if top is None or molecule["score"] < top["score"]:
                    top = molecule
            diverse_top.append(top)
        ids_taken = list()
        for entry in diverse_top:
            if entry["id"] not in ids_taken:
                self.diverse_top.append(entry)
                ids_taken.append(entry["id"])

    def visualize_top(self, output_name, per_row=5, mol_size=(200, 200)):
        """Displays the scaffolds using RDKit's drawing functions."""
        top_mols = [Chem.MolFromSmiles(smiles) for smiles in self.top_smiles]
        img = Draw.MolsToGridImage(top_mols, molsPerRow=per_row, subImgSize=mol_size)
        img_path = f"{self.working_dir}/{output_name}.png"
        img.save(img_path)
        print(f"Saved: {img_path}")

    def collect_top_pdbqts(self, id_list, output_name):
        self.check_molecule_info(id_list)
        input_directory_1 = f"{self.working_dir}/analogue_docking/docking_output"
        input_directory_2 = f"{self.working_dir}/analogue_docking"
        target_directory = f"{self.working_dir}/{output_name}"
        if not path.exists(target_directory):
            makedirs(target_directory)
        output_file_no_found = 0
        for entry in id_list:
            source_filename_1 = path.join(input_directory_1, f"{entry}_{self.run_id}_out.pdbqt")
            source_filename_2 = path.join(input_directory_2, f"{entry}_{self.run_id}.pdbqt")
            target_filename = path.join(target_directory, f"{entry}_{self.run_id}.pdbqt")
            if path.exists(source_filename_1):
                shutil.copy(source_filename_1, target_filename)
            elif path.exists(source_filename_2):
                shutil.copy(source_filename_2, target_filename)
            else:
                output_file_no_found += 1
        print(f"Pdbqt not found {output_file_no_found}/{len(id_list)}")
        self.check_molecule_info(id_list)

    def write_top_smiles(self, filename):
        print(f"Saving top smiles to {filename}")
        list_of_lines = list()
        for entry in self.diverse_top:
            line = f"{entry['smiles']} {entry['id']}\n"
            list_of_lines.append(line)
        with open(filename, "w+") as f:
            f.writelines(list_of_lines)
