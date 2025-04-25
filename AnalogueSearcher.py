import multiprocessing
import os
from pathlib import Path
from rdkit import Chem as chem
from rdkit import DataStructs as tanimoto
from rdkit.Chem.rdMolDescriptors import CalcNumHeavyAtoms
from rdkit.Chem.Scaffolds import MurckoScaffold as murcko
from rdkit.Chem import rdFMCS
import re


class NoLibraryFilesException(Exception):
    def __init__(self, message="No library files loaded. Double check the 'molecule_lib' and 'lib_prefix' parameters"):
        self.message = message
        super().__init__(self.message)


class AnalogueSearcher:
    def __init__(self, candidate_file, analogue_file, lib_path, thresholds, analogue_mode, lib_prefix, analogue_cap=50,
                 match_rings=True, no_cand=None):
        self.candidate_file = candidate_file
        self.analogue_file = analogue_file
        self.library_path = lib_path
        self.smiles_similar = dict()
        self.smiles_filtered = list()
        self.similarity_thresholds = thresholds
        self.analogue_mode = analogue_mode
        self.lib_prefix = lib_prefix
        self.no_cand = no_cand
        self.analogue_cap = analogue_cap
        self.match_rings = match_rings
        self.data_dict = dict()

    def load_smiles(self):
        with open(self.candidate_file, "r") as f:
            data = f.readlines()
        if self.no_cand != 0:
            if self.no_cand <= len(data):
                data = data[:self.no_cand]
            else:
                print(f"Not enough candidate molecules ({len(data)}/{self.no_cand}). Using all.")
        print(f"Analogue search: Finished loading candidate smiles: {len(data)} entries")
        for entry in data:
            entry = entry.rstrip("\n")
            parts = entry.split(" ")
            self.data_dict[parts[0]] = parts[1]
        return list(self.data_dict.keys())

    def get_all_index_files(self):
        dir_content = os.listdir(self.library_path)
        list_of_files = list()
        for entry in dir_content:
            if entry[-3:] == ".fs":
                if not self.lib_prefix:
                    list_of_files.append(entry)
                elif entry.startswith(self.lib_prefix):
                    list_of_files.append(entry)
        print(f"Registered {len(list_of_files)} library files")
        if not list_of_files:
            raise NoLibraryFilesException()
        return list_of_files

    @staticmethod
    def split_entry(line):
        parts = re.split("\t|\n| ", line)
        relevant_parts = list()
        for part in parts:
            if len(part) > 2:
                relevant_parts.append(part)
        if not len(relevant_parts) == 2:
            print("Problem: ", parts)
            return "error"
        return relevant_parts

    def find_similar_molecules(self, smile_clean):
        all_index_files = self.get_all_index_files()
        mol_original = chem.MolFromSmiles(smile_clean)
        fp_original = chem.RDKFingerprint(mol_original)
        analogues = list()
        analogues.append(self.find_analogues(all_index_files, smile_clean, mol_original, fp_original))
        return analogues

    def find_analogues(self, file, smile_clean, mol_original, fp_original):
        if self.analogue_mode == "sim":
            print("Running similarity")
            entries = self.run_openbabel(file, smile_clean, "sim")
        else:
            print("Running substructure")
            entries = self.run_openbabel(file, smile_clean, "sub")
        analogues = list()
        for line in entries:
            split_eval = self.split_entry(line)
            if split_eval == "error":
                continue
            analogue_smi = split_eval[0]
            analogue_id = split_eval[1]
            analogue_mol = chem.MolFromSmiles(analogue_smi)
            analogue_fp = chem.RDKFingerprint(analogue_mol)
            tani = tanimoto.TanimotoSimilarity(fp_original, analogue_fp)
            analogue_info = [smile_clean, analogue_smi, analogue_id, round(tani, 4), self.data_dict[smile_clean]]
            scaf = murcko.GetScaffoldForMol(mol_original)
            max_common = rdFMCS.FindMCS((mol_original, analogue_mol), ringMatchesRingOnly=True)
            number_in_scaf = CalcNumHeavyAtoms(scaf)
            if max_common.numAtoms >= number_in_scaf or self.match_rings is False:
                if tani > 0.5 or self.analogue_mode == "sub":
                    analogues.append(analogue_info)
        return analogues

    def run_openbabel(self, index_files, smile, mode, multi=False):
        all_entries = list()
        if multi:
            tasks = [
                (os.path.join(self.library_path, file), smile, f"temporary_{i}.smi", mode)
                for i, file in enumerate(index_files)
            ]
            with multiprocessing.Pool(processes=7) as pool:
                results = pool.starmap(self.process_file, tasks)
        else:
            for i, file in enumerate(index_files):
                input_path = os.path.join(self.library_path, file)
                output_path = f"temporary_{i}.smi"
                results = self.process_file(input_path, smile, output_path, mode)
        all_entries.extend(results)
        return all_entries

    @staticmethod
    def process_file(library_file_path, smile, result_file_smi, mode):
        if mode == "sub":
            mol = chem.MolFromSmiles(smile)
            scaf = murcko.GetScaffoldForMol(mol)
            resmile = chem.MolToSmiles(scaf)
            if resmile == str():
                resmile = smile
            obabal_command = f'obabel {library_file_path} -O {result_file_smi} -s "{resmile}"'
            os.system(obabal_command)
        else:
            obabal_command = f'obabel {library_file_path} -O {result_file_smi} -at0.7 -xt -s "{smile}"'
            os.system(obabal_command)
        with open(result_file_smi, "r") as f:
            entries = f.readlines()
        Path.unlink(Path(result_file_smi))
        return entries

    def evaluate_list(self):
        smiles = self.load_smiles()
        for index, smile in enumerate(smiles):
            print(f"evaluate: {smile}")
            smile_clean = smile.rstrip("\n")
            print(f"Searching {index + 1} / {len(smiles)}")
            similars_for_smile = list()
            similars_primary = self.find_similar_molecules(smile_clean)
            for entry in similars_primary:
                if len(entry) > 0:
                    similars_for_smile.extend(entry)
            similars_for_smile.sort(key=lambda x: x[3], reverse=True)
            if len(similars_for_smile) > self.analogue_cap != 0:
                complete_len = len(similars_for_smile)
                similars_for_smile = similars_for_smile[:self.analogue_cap]
                print(f"Analogue list truncated from {complete_len} to {self.analogue_cap} with highest Tanimoto")
            else:
                print(f"Analogue search for this candidate yielded {len(similars_for_smile)} molecules")
            self.smiles_similar[smile_clean] = similars_for_smile
        print(f"Loaded: {smiles[0] if len(smiles) > 0 else 'none'}")
        if self.analogue_mode == "sim":
            self.filter_analogues_sim()
        else:
            self.filter_analogues_sub()
        lines = self.format_data()
        self.write_analogue_list(lines)

    def filter_analogues_sim(self):
        screening_candidates = list()
        for seed, analogues in self.smiles_similar.items():
            highest_similarity = 0
            for analogue in analogues:
                if analogue[3] > highest_similarity:
                    highest_similarity = analogue[3]
            if highest_similarity < self.similarity_thresholds[1]:
                continue
            for analogue in analogues:
                if analogue[3] > self.similarity_thresholds[0] or (highest_similarity < self.similarity_thresholds[0]
                                                                   and analogue[3] > self.similarity_thresholds[1]):
                    complete_info_clean = [s.strip() if isinstance(s, str) else s for s in analogue]
                    screening_candidates.append(complete_info_clean)
        self.smiles_filtered = screening_candidates

    def filter_analogues_sub(self):
        screening_candidates = list()
        for seed, analogues in self.smiles_similar.items():
            for analogue in analogues:
                complete_info_clean = [s.strip() if isinstance(s, str) else s for s in analogue]
                screening_candidates.append(complete_info_clean)
        self.smiles_filtered = screening_candidates

    @staticmethod
    def make_line(line_data):
        line = str()
        for index, entry in enumerate(line_data):
            if index != 0:
                line += "\t"
            line += str(entry)
        line += "\n"
        return line

    def format_data(self):
        list_of_lines = list()
        for entry in self.smiles_filtered:
            # ACHTUNG!
            if isinstance(entry, list) and len(entry) == 5:
                line = self.make_line(entry)
                list_of_lines.append(line)
        return list_of_lines

    def write_analogue_list(self, lines):
        print("Write file", len(lines))
        mode = "r+" if os.path.exists(self.analogue_file) else "w+"
        with open(self.analogue_file, mode) as f:
            previous = f.readlines() if mode == "r+" else list()
            f.seek(0)
            previous.extend(lines)
            f.truncate()
            f.writelines(previous)
