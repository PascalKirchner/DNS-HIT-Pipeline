import numpy as np
from pathlib import Path
import os
import re
from PIL import Image
import rdkit.Chem as Chem
import rdkit.Chem.Draw as Draw


class CandidateSelector:
    def __init__(self, config):
        self.config = config
        self.raw_data = None
        self.clean_data = None
        self.best_ids = None
        self.candidate_dict = None

    def handle_complete_selection(self):
        raw_lines = self.load_file()
        self.extract_data(raw_lines)
        self.make_clean_data()
        self.assign_candidate_status()
        self.make_candidate_dict()
        self.get_best()
        self.list_smiles()

    def load_file(self):
        with open(self.config["dataset_file"], "r") as f:
            data_lines = f.readlines()
        return data_lines

    def extract_data(self, raw_lines):
        data_list = list()
        for line in raw_lines[1:]:
            if len(line) > 5:
                parts = re.split("\t|\n| ", line)
                parts_clean = list()
                for part in parts:
                    if part not in ("", " ", "\t", "\n"):
                        parts_clean.append(part)
                data_list.append(parts_clean)

        self.raw_data = data_list

    def make_clean_data(self):
        data_clean = list()
        for entry in self.raw_data:
            try:
                new_entry = dict()
                new_entry["id"] = int(entry[0])
                new_entry["affinity"] = float(entry[1])
                new_entry["logp"] = float(entry[2])
                new_entry["weight"] = float(entry[3])
                new_entry["synthi"] = float(entry[6])
                new_entry["smiles"] = entry[8]
                data_clean.append(new_entry)
            except ValueError:
                print(entry)
        self.clean_data = data_clean

    def assign_candidate_status(self):
        for entry in self.clean_data:
            if entry["logp"] <= self.config["logp_cutoff"]:
                if entry["weight"] >= self.config["lower_weight"]:
                    if entry["weight"] <= self.config["upper_weight"]:
                        if entry["synthi"] >= self.config["synthes_cutoff"]:
                            entry["candidate"] = True
                            continue
            entry["candidate"] = False

    def make_candidate_dict(self):
        candidate_dict = dict()
        molecules_used = list()
        for entry in self.clean_data:
            if entry["candidate"] and entry["smiles"] not in molecules_used:
                candidate_dict[entry["id"]] = entry["affinity"]
                molecules_used.append(entry["smiles"])
        self.candidate_dict = candidate_dict

    def get_best(self):
        sorted_candidates_items = sorted(self.candidate_dict.items(), key=lambda x: x[1])
        sorted_candidates_dict = {k: v for k, v in sorted_candidates_items}
        sorted_candidates_list = list(sorted_candidates_dict.keys())
        if self.config["n_of_candidates"] >= len(sorted_candidates_list):
            self.best_ids = sorted_candidates_list
        else:
            self.best_ids = sorted_candidates_list[:self.config["n_of_candidates"]]

    def reformat_candidate_list(self):
        reformatted = dict()
        for entry in self.clean_data:
            reformatted[entry["id"]] = entry
        self.clean_data = reformatted

    def list_smiles(self):
        self.reformat_candidate_list()
        list_of_smiles = list()
        for entry in self.best_ids:
            canddata = self.clean_data[entry]
            candidate_line = f"{canddata['smiles']} {canddata['id']} {canddata['affinity']} {canddata['weight']}\n"
            list_of_smiles.append(candidate_line)
        with open(self.config["candidates_file"], "w") as f:
            f.writelines(list_of_smiles)

    @staticmethod
    def visualize_best(smiles, image_path, exp_no, non_legit=None):
        if not Path(image_path).is_dir():
            os.mkdir(image_path)
        list_of_images = list()
        for index, smile in enumerate(smiles):
            if not non_legit or index not in non_legit:
                smile_path = os.path.abspath(os.path.join(image_path, str(index) + ".png"))
                mol = Chem.MolFromSmiles(smile)
                Draw.MolToImageFile(mol, smile_path)
                list_of_images.append(smile_path)
        combine_images(6, list_of_images, "matrix" + str(exp_no) + ".png")

    @staticmethod
    def combine_images(columns, images, filename):
        rows = len(images) // columns
        if len(images) % columns:
            rows += 1
        x, y = 0, 0
        width = (columns - 1) * 350 + 300
        height = (rows - 1) * 350 + 300
        full_image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
        for index, image_name in enumerate(images):
            image = Image.open(image_name)
            full_image.paste(image, (x, y))
            x += 350
            if (index + 1) % columns == 0:
                x = 0
                y += 350
        full_image.save(filename)
