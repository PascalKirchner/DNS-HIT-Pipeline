import matplotlib.pyplot as plt
import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.Descriptors as descr
from rdkit.Chem.QED import qed
import rdkit.Chem.Draw as draw
import os
from pathlib import Path
from PIL import Image
import re
import seaborn
import pandas as pd


class Visualizer:
    group_map = {
        0: (0.0, 0.9),
        1: (0.9, 0.96),
        2: (0.96, 1.0)
    }

    def __init__(self, figure_dir, dataset_path, run_id):
        self.figure_dir = figure_dir
        self.dataset_path = dataset_path
        self.run_id = run_id
        self.raw_data = None
        self.clean_data = None
        self.load_data()

    def load_data(self):
        with open(self.dataset_path, "r") as f:
            data_lines = f.readlines()
        self.extract_data(data_lines)
        self.make_clean_data()

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
        data_clean = dict()
        for entry in self.raw_data:
            try:
                new_entry = dict()
                new_entry["affinity"] = float(entry[1])
                new_entry["logp"] = float(entry[2])
                new_entry["weight"] = float(entry[3])
                new_entry["heavy"] = float(entry[4])
                new_entry["efficiency"] = float(entry[5])
                new_entry["synthi"] = float(entry[6])
                new_entry["smiles"] = entry[8]
                data_clean[int(entry[0])] = new_entry
            except ValueError:
                continue
        self.clean_data = data_clean

    def make_path(self, output_type, extension=".txt"):
        new_filename = output_type + str(self.run_id) + extension
        path = os.path.join(self.figure_dir, new_filename)
        return path

    @staticmethod
    def get_tick_range(data):
        distance = len(data) / 8.0
        d_clean = int(round(distance / 1000, 0) * 1000)
        if d_clean < 1:
            d_clean = 1
        return list(range(0, len(data), d_clean))

    def make_regression_plot(self, y_name, name, y_axis, title):
        y_data = list()
        for entry in self.clean_data.values():
            y_data.append(entry[y_name])
        plt.close()
        reg_y = np.array(y_data)
        reg_x = np.array(range(len(reg_y)))
        fig, ax = plt.subplots()
        fit = np.polyfit(reg_x, reg_y, 1)
        m, b = fit[0], fit[1]
        plt.scatter(reg_x, reg_y)
        plt.plot(reg_x, reg_y, 'yo', reg_x, m * reg_x + b, '--k')
        ax.set_xticks(self.get_tick_range(reg_x))
        plt.xlabel("Numerical index")
        plt.ylabel(y_axis)
        plt.title(title)
        output_path = self.make_path(name, ".png")
        plt.savefig(output_path)

    def complete_evaluation(self):
        self.make_regression_plot("affinity", "RegressionAffinity", "Binding affinity",
                                  "Progression of binding affinity")
        self.make_regression_plot("heavy", "RegressionNumberHeavy", "Number of heavy atoms",
                                  "Progression of heavy atom number")
        self.make_regression_plot("efficiency", "RegressionEfficicency", "Ligand efficiency",
                                  "Progression of ligand efficiency")
        self.make_contour()
        self.make_weight_histogram()
        self.make_triple_scatter()
        self.make_scatter_3d()

    def visualize_best(self, smiles, mol_identifier, series_identifier):
        image_path = os.path.abspath(os.path.join(self.main_path, series_identifier + str(mol_identifier)))
        matrix_path = os.path.abspath(os.path.join(image_path, "matrix.png"))
        if not Path(image_path).is_dir():
            os.mkdir(image_path)
        list_of_images = list()
        for index, smile in enumerate(smiles):
            smile_path = os.path.abspath(os.path.join(image_path, str(index) + ".png"))
            mol = Chem.MolFromSmiles(smile)
            draw.MolToImageFile(mol, smile_path)
            list_of_images.append(smile_path)
        self.combine_images(6, list_of_images, matrix_path)

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

    def make_contour(self):
        plt.close()
        warning = str()
        list_of_molecules = list(self.clean_data.keys())
        list_sections = (list_of_molecules[:500], list_of_molecules[-500:])
        section_data = {"x": list(), "y": list(), "name": list()}
        for n in range(2):
            for entry in list_sections[n]:
                try:
                    smiles = self.clean_data[entry]["smiles"]
                    mol = Chem.MolFromSmiles(smiles)
                    qe = qed(mol)
                    lo = self.clean_data[entry]["logp"]
                    section_data["y"].append(qe)
                    section_data["x"].append(lo)
                    section_data["name"].append("Early" if n == 0 else "Late")
                except Chem.rdchem.KekulizeException:
                    print("Skipped one molecule due to failed aromatization.")
        if len(section_data["x"]) == 0 or len(section_data["y"]) == 0:
            warning = "Warning: No valid data points for contour plot."
        g = seaborn.jointplot(x=section_data["x"], y=section_data["y"], hue=section_data["name"], kind="kde",
                              palette={"Early": "blue", "Late": "orange"})
        g.fig.suptitle("QED and logP distributions of early and late molecules" + warning)
        plt.xlabel("Quantitative estimate of drug-likeness (QED)")
        plt.ylabel("Octanol/water partition coefficient (logP)")
        output_path = self.make_path("contour", extension=".svg")
        plt.savefig(output_path)

    def make_weight_histogram(self):
        plt.close()
        weights = list()
        for value in self.clean_data.values():
            weights.append(value["weight"])
        counts, bins = np.histogram(weights)
        plt.stairs(counts, bins)
        output_path = self.make_path("WeightHistogram", extension=".svg")
        plt.title("Distribution of molecular weights")
        plt.savefig(output_path)

    def sort_data_points(self):
        x = {0: [], 1: [], 2: []}
        y = {0: [], 1: [], 2: []}
        z = {0: [], 1: [], 2: []}
        for index, (score_min, score_max) in self.group_map.items():
            for value in self.clean_data.values():
                if score_min <= value["synthi"] <= score_max:
                    x[index].append(value["affinity"])
                    y[index].append(value["logp"])
                    z[index].append(value["synthi"])
        return x, y, z

    def get_axis_limits(self):
        data = list(self.clean_data.values())
        min_log = int(round(min(data, key=lambda q: q["logp"])["logp"], 0))-1
        max_log = int(round(max(data, key=lambda q: q["logp"])["logp"], 0))+1
        min_aff = int(round(min(data, key=lambda q: q["affinity"])["affinity"], 0))-1
        max_aff = int(round(max(data, key=lambda q: q["affinity"])["affinity"], 0))+1
        min_syn = int(round(min(data, key=lambda q: q["synthi"])["synthi"], 0))-1
        max_syn = int(round(max(data, key=lambda q: q["synthi"])["synthi"], 0))+1
        return [(min_aff, max_aff), (min_log, max_log), (min_syn, max_syn)]

    def make_triple_scatter(self):
        data_sorted = self.sort_data_points()
        colors = ("red", "yellow", "green")
        fig, ax = plt.subplots()
        axis_limits = self.get_axis_limits()
        ax.set_xlim(axis_limits[0])
        ax.set_ylim(axis_limits[1])
        for index, (score_min, score_max) in self.group_map.items():
            ax.scatter(data_sorted[0][index], data_sorted[1][index], c=colors[index],
                       label=f'Synthesizability: {score_min}-{score_max}', alpha=0.5)
        ax.set_xlabel('Affinity')
        ax.set_ylabel('LogP')
        ax.set_title('Scatter Plot: Affinity, logP, Synthesizability')
        ax.legend()
        plt.tight_layout()
        output_path = self.make_path("triple", extension=".svg")
        plt.savefig(output_path)

    def make_scatter_3d(self):
        """
        Create a 3D scatter plot using 'Aff.', 'LogP', and 'PredictedScore' columns from data.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(30, 40)
        # fig, ax = plt.subplots()
        axis_limits = self.get_axis_limits()
        ax.set_xlim(axis_limits[0])
        ax.set_ylim(axis_limits[1])
        ax.set_zlim(axis_limits[2])
        data_sorted = self.sort_data_points()
        ax.scatter(data_sorted[0][0], data_sorted[1][0], data_sorted[2][0], c='red',
                   label=f'Synthesizability {self.group_map[0][0]} - {self.group_map[0][1]}', alpha=0.5, s=20)
        ax.scatter(data_sorted[0][1], data_sorted[1][1], data_sorted[2][1], c='yellow',
                   label=f'Synthesizability {self.group_map[1][0]} - {self.group_map[1][1]}', alpha=0.8, s=20)
        ax.scatter(data_sorted[0][2], data_sorted[1][2], data_sorted[2][2], c='green',
                   label=f'Synthesizability {self.group_map[2][0]} - {self.group_map[2][1]}', alpha=0.8, s=20)
        ax.set_xlabel('Affinity')
        ax.set_ylabel('LogP')
        ax.set_zlabel('Synthesizability')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=-3)
        output_path = self.make_path("Aff_log_syn_3d", extension=".svg")
        plt.savefig(output_path)
