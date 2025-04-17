import rdkit.Chem as Chem
from rdkit.Chem.Crippen import MolLogP
import rdkit.Chem.Descriptors as Descr
import traceback

import SynthiInterface


class DatasetMaker:
    def __init__(self, sm_path, eval_path):
        self.data = dict()
        self.smiles_path = sm_path
        self.eval_path = eval_path
        self.synthi_interface = SynthiInterface.SynthiInterface()

    def load_smiles_and_scores(self):
        with open(self.smiles_path, "r") as f:
            raw_data = f.readlines()
        for index, entry in enumerate(raw_data):
            subentries = entry.split(" ")
            self.data[index] = dict()
            self.data[index]["smiles"] = subentries[0]
            try:
                self.data[index]["score"] = float(subentries[1])
            except ValueError:
                self.data[index]["score"] = None

    @staticmethod
    def get_weight(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            weight = round(Descr.ExactMolWt(mol), 1)
            return weight
        except Exception as e:
            tr = traceback.format_exc()
            print("Mol weight error", tr)
            return 0

    @staticmethod
    def get_log_p(smiles):
        mol = Chem.MolFromSmiles(smiles)
        logp = round(MolLogP(mol), 3)
        return logp

    @staticmethod
    def get_heavy(smiles):
        mol = Chem.MolFromSmiles(smiles)
        heavy = mol.GetNumHeavyAtoms()
        return heavy

    @staticmethod
    def get_efficiency(score, heavy):
        result1 = round(float(score) / heavy, 3)
        result2 = result1 * -1.0
        return result2

    def get_synthesizability(self, smiles):
        synthi_score = self.synthi_interface.predict_synthesizability(smiles)
        return synthi_score

    @staticmethod
    def get_formula(smiles):
        formula = dict()
        mol_without_h = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol_without_h)
        for atom in mol.GetAtoms():
            atom_type = atom.GetSymbol()
            if atom_type in formula.keys():
                formula[atom_type] += 1
            else:
                formula[atom_type] = 1
        return formula

    @staticmethod
    def reformat_formula(formula):
        common_atoms = ("C", "H", "N", "O", "S", "F", "Cl", "Br", "I")
        new_formula = str()
        for atom in common_atoms:
            if atom in formula.keys():
                new_formula += atom
                if formula[atom] != 1:
                    new_formula += str(formula[atom])
        for atom, count in formula.items():
            if atom not in common_atoms:
                new_formula += atom
                if count != 1:
                    new_formula += str(count)
        return new_formula

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

    def format_data(self):
        lines = list()
        lines.append(self.make_line(["Index", "Aff.", "LogP", "Weight", "Heavy", "Eff.", "Synthi", "Formula",
                                     "SMILES"], 8, 20))
        for index, molecule in self.data.items():
            line_data = list()
            line_data.append(index)  # 0
            line_data.append(molecule["score"])  # 1
            line_data.append(molecule["logp"])  # 2
            line_data.append(molecule["weight"])  # 3
            line_data.append(molecule["heavy"])  # 4
            line_data.append(molecule["efficiency"])  # 5
            line_data.append(molecule["synthi"])  # 6
            line_data.append(molecule["formula"])  # 7
            line_data.append(molecule["smiles"])  # 8
            line = self.make_line(line_data, 8, 20)
            lines.append(line)
        return lines

    def make_line(self, line_data, default_len, formula_len):
        line = str()
        for index, entry in enumerate(line_data):
            if index == 7:
                block = self.make_block(entry, formula_len)
            elif entry == line_data[-1]:
                line += "   "
                block = entry
            else:
                block = self.make_block(entry, default_len)
            line += block
        line += "\n"
        return line

    def evaluate_all(self):
        self.load_smiles_and_scores()
        for entry in self.data.values():
            valid_entry = True
            if entry["score"] is not None:
                try:
                    entry["logp"] = self.get_log_p(entry["smiles"])
                    entry["weight"] = self.get_weight(entry["smiles"])
                    entry["heavy"] = self.get_heavy(entry["smiles"])
                    entry["efficiency"] = self.get_efficiency(entry["score"], entry["heavy"])
                    entry["synthi"] = self.get_synthesizability(entry["smiles"])
                    raw_formula = self.get_formula(entry["smiles"])
                    entry["formula"] = self.reformat_formula(raw_formula)
                except Exception as e:
                    tr = traceback.format_exc()
                    print("dataset error", tr)
                    valid_entry = False
            else:
                valid_entry = False
            if not valid_entry:
                entry["logp"] = None
                entry["weight"] = None
                entry["heavy"] = None
                entry["efficiency"] = None
                entry["synthi"] = None
                entry["formula"] = None

    def write_dataset_file(self, lines):
        with open(self.eval_path, "w") as f:
            f.writelines(lines)

    def make_dataset(self):
        self.evaluate_all()
        lines = self.format_data()
        self.write_dataset_file(lines)
