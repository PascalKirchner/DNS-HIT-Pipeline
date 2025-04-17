import argparse
import ast
from decimal import Decimal, getcontext
import matplotlib.pyplot as plt
import numpy as np
from os import listdir, makedirs, path
from random import choices
import re
from scipy import stats
import subprocess

import Screening


class RandomSampler:
    main_path = "../data_and_results/"

    def __init__(self, libdir, sample_size, protein, run_id, sampler_dir, grid_info,
                 docking_tool="vina", vina_weights=None, seed=777):
        self.lib_path = path.join(self.main_path, libdir)
        self.sampler_dir = sampler_dir
        self.sample_size = sample_size
        self.protein = protein
        self.run_id = run_id
        self.grid_info = grid_info
        self.line_count_dict = dict()
        self.weights_dict = dict()
        self.libfile_selection = dict()
        self.docking_dir = path.join(self.sampler_dir, "docking")
        self.selection_file = None
        self.result_file = None
        self.vina_weights = vina_weights
        self.docking_tool = docking_tool
        self.seed = seed

    def collect_smi_files(self):
        print("Started collecting library files.")
        for filename in listdir(self.lib_path):
            if filename.endswith(".smi"):
                file_path = path.join(self.lib_path, filename)
                result = subprocess.run(["wc", "-l", file_path], capture_output=True, text=True, check=True)
                line_count = int(result.stdout.split()[0])
                self.line_count_dict[filename] = line_count
        print("Finished collecting library files.")

    def assign_weights(self):
        print("Started assigning weights to library files.")
        total_lines = sum(self.line_count_dict.values())
        for filename, line_count in self.line_count_dict.items():
            weight = line_count * 1.0 / total_lines
            self.weights_dict[filename] = weight
        print("Finished assigning weights to library files.")

    def select_files_weighted(self):
        print("Started distribution of sampling to library files.")
        for filename in self.line_count_dict.keys():
            self.libfile_selection[filename] = 0
        filenames = list(self.weights_dict.keys())
        weights = list(self.weights_dict.values())
        for _ in range(self.sample_size):
            selection = choices(filenames, weights, k=1)[0]
            self.libfile_selection[selection] += 1
        print("Finished distribution of sampling to library files.")

    def select_random_mols(self):
        print("Started random molecule selection.")
        all_lines = list()
        for filename, number in self.libfile_selection.items():
            filepath = path.join(self.lib_path, filename)
            try:
                result = subprocess.run(
                    ['shuf', '-n', str(number), filepath],
                    capture_output=True, text=True, check=True
                )
                random_lines = result.stdout.splitlines()
                for line in random_lines:
                    all_lines.append(line + "\n")
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while running the command:")
                print(f"Command: {' '.join(e.cmd)}")
                print(f"Return code: {e.returncode}")
                print(f"Error message: {e.stderr}")
        self.selection_file = path.join(self.sampler_dir, "random_selection.txt")
        with open(self.selection_file, "w+") as f:
            f.writelines(all_lines)
        print("Finished random molecule selection.")

    def dock_molecules(self):
        print("Started random molecule docking.")
        self.result_file = path.join(self.sampler_dir, f"screening{self.run_id}.txt")
        screening_instance = Screening.Screening(self.docking_dir,
                                                 self.result_file,
                                                 self.grid_info,
                                                 self.selection_file,
                                                 self.protein,
                                                 self.run_id,
                                                 self.docking_tool,
                                                 self.vina_weights,
                                                 self.seed)
        source_dir = path.abspath("/workspace/data_and_results")
        screening_instance.generate_config_file(self.docking_dir, self.protein + "_config.txt")
        # screening_instance.copy_protein_pdbqt(source_dir, self.docking_dir, self.protein)
        screening_instance.dock_all_smiles()
        print("Finished random molecule docking.")

    def run(self):
        self.collect_smi_files()
        self.assign_weights()
        self.select_files_weighted()
        self.select_random_mols()
        self.dock_molecules()
        return self.result_file


class Statistics:
    """Results for both datasets are ordered random, targeted."""
    def __init__(self, filename_random, filename_target, directory):
        self.directory = directory
        self.filename_random = filename_random
        self.filename_target = filename_target
        self.random_scores = list()
        self.target_scores = list()
        self.means = list()
        self.medians = list()
        self.variances = list()
        self.shapiro_stats = list()
        self.shapiro_p = list()
        self.normality = list()
        self.mw_results = None
        self.significant = None
        self.rb_result = None
        self.effect_size = None

    def complete_evaluation(self):
        self.load_scores(self.filename_random, self.random_scores)
        self.load_scores(self.filename_target, self.target_scores)
        print("Evaluating random selection")
        self.mean(self.random_scores)
        self.median(self.random_scores)
        self.variance(self.random_scores)
        self.shapiro_wilk(self.random_scores)
        print("Evaluating targeted selection")
        self.mean(self.target_scores)
        self.median(self.target_scores)
        self.variance(self.target_scores)
        self.shapiro_wilk(self.target_scores)
        self.mann_whitney(self.random_scores, self.target_scores)
        self.rank_biserial_correlation(self.random_scores, self.target_scores)
        list_of_datasets = [self.random_scores, self.target_scores]
        self.boxplots(list_of_datasets)
        self.histogram(list_of_datasets)
        self.print_statistical_results()

    @staticmethod
    def load_scores(filename, whereto):
        print(f"File: {filename}")
        with open(filename, "r") as f:
            lines = f.readlines()
        error_printed = False
        for line in lines:
            parts = re.split("\t|\n| ", line)
            parts_clean = [p for p in parts if p != ""]
            try:
                score = float(parts_clean[1].strip())
                whereto.append(score)
            except ValueError:
                if not error_printed:
                    print(line)
                    print(parts_clean)
                    error_printed = True

    def mean(self, data):
        mean = np.mean(data)
        print(f"Mean: {mean}")
        self.means.append(mean)

    def median(self, data):
        median = np.median(data)
        print(f"Median: {median}")
        self.medians.append(median)

    def variance(self, data):
        sample_variance = np.var(data, ddof=1)
        print(f"Sample Variance: {sample_variance}")
        self.variances.append(sample_variance)

    def shapiro_wilk(self, data):
        stat, p_value = stats.shapiro(data)
        print(f"Shapiro-Wilk test statistic: {stat}, p-value: {p_value}")
        if p_value > 0.05:
            print("The data is normally distributed (fail to reject H0).")
        else:
            print("The data is not normally distributed (reject H0).")
        self.shapiro_stats.append(stat)
        self.shapiro_p.append(p_value)
        self.normality.append("normal" if p_value > 0.05 else "non_normal")

    def mann_whitney(self, data1, data2):
        getcontext().prec = 100000
        stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        high_precision_p = Decimal(float(p_value))
        print(f"Mann-Whitney U test statistic: {stat}, p-value: {high_precision_p}")
        if p_value < 0.05:
            print("There is a statistically significant difference between the two groups.")
        else:
            print("There is no statistically significant difference between the two groups.")
        self.mw_results = [stat, p_value]
        self.significant = "significant" if p_value < 0.05 else "non_significant"

    def rank_biserial_correlation(self, data1, data2):
        mwu = self.mw_results[0]
        r_rb = 1 - (2 * mwu) / (len(data1) * len(data2))
        print(f"Rank-biserial correlation (effect size): {r_rb}")
        self.rb_result = r_rb
        if r_rb < -0.5 or r_rb > 0.5:
            self.effect_size = "large"
        elif r_rb < -0.3 or r_rb > 0.3:
            self.effect_size = "medium"
        elif r_rb < -0.1 or r_rb > 0.1:
            self.effect_size = "small"
        else:
            self.effect_size = "negligible"

    def boxplots(self, list_of_datasets, minlim=-4, maxlim=-15):
        plt.boxplot(list_of_datasets)
        plt.ylim(minlim, maxlim)
        plt.title('Boxplot of docking scores')
        plt.xlabel('Dataset')
        plt.ylabel('Values')
        filename = path.join(self.directory, 'boxplot_image.png')
        plt.savefig(filename)
        plt.show()

    def histogram(self, list_of_datasets):
        for index, entry in enumerate(list_of_datasets):
            median_value = np.median(entry)
            plt.axvline(median_value, color='red', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')
            plt.hist(entry, bins=30, edgecolor='black')
            plt.title('Histogram of docking scores')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            filename = path.join(self.directory, f'histo_{index}.png')
            plt.savefig(filename)
            plt.show()

    def print_statistical_results(self):
        lines = list()
        lines.append(f"Mean: {self.means[0]} (random), {self.means[1]} (targeted)\n")
        lines.append(f"Median: {self.medians[0]} (random), {self.medians[1]} (targeted)\n")
        lines.append(f"Variance: {self.variances[0]} (random), {self.variances[1]} (targeted)\n")
        lines.append(f"Shapiro-Wilk (random): {self.shapiro_stats[0]}; p {self.shapiro_p[0]} ({self.normality[0]})\n")
        lines.append(f"Shapiro-Wilk (targeted): {self.shapiro_stats[1]}; p {self.shapiro_p[1]} ({self.normality[1]})\n")
        mw_text = f"U = {self.mw_results[0]}, p = {self.mw_results[1]}"
        lines.append(f"Difference according to Mann-Whitney U test is {self.significant }, {mw_text}\n")
        lines.append(f"Effect size (Rank-Biserial correlation) is {self.effect_size}, value = {self.rb_result}\n")
        filepath = path.join(self.directory, "statistical_results.txt")
        with open(filepath, "w+") as f:
            f.writelines(lines)
