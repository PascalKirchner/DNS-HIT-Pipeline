"""
Original file model_logP_QED_switch.py from MoleGuLAR
M. Goel et. Al.: MoleGuLAR: Molecule Generation Using Reinforcement Learning with Alternating Rewards
J. Chem. Inf. Model. 2021, 61, 12, 5815–5826, DOI: 10.1021/acs.jcim.1c01341

Permission has been granted by the authors of MoleGuLAR to use their code for academic, non-commercial purposes.

The present file is derived from model_logP_QED_switch.py and was written by P. Kirchner for use in the DNS-HIT pipeline.
"""

import os
import sys

current_file_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_file_directory)
sys.path.append('./release')

# import pickle
import random
import shutil
import subprocess

import numpy as np
# import seaborn as sns
import torch
import wandb
from data import GeneratorData
# from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdmolfiles
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.QED import qed
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from reinforcement import Reinforcement
from stackRNN import StackAugmentedRNN
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from tqdm import tqdm, trange
from utils import canonical_smiles

import rewards as rwds
from Predictors.GINPredictor import Predictor as GINPredictor
from Predictors.RFRPredictor import RFRPredictor
from Predictors.SolvationPredictor import FreeSolvPredictor

RDLogger.DisableLog('rdApp.info')


mlg_inst = None


class Predictor(object):
    def __init__(self, path):
        super(Predictor, self).__init__()
        self.path = path
        self.isdocker = True

    @staticmethod
    def predict(smiles, use_tqdm=False):
        canonical_indices = []
        invalid_indices = []
        if use_tqdm:
            pbar = tqdm(range(len(smiles)))
        else:
            pbar = range(len(smiles))
        for i in pbar:
            sm = smiles[i]
            if use_tqdm:
                pbar.set_description("Calculating predictions...")
            try:
                sm = Chem.MolToSmiles(Chem.MolFromSmiles(sm))
                if len(sm) == 0:
                    invalid_indices.append(i)
                else:
                    canonical_indices.append(i)
            except:
                invalid_indices.append(i)
        canonical_smiles = [smiles[i] for i in canonical_indices]
        invalid_smiles = [smiles[i] for i in invalid_indices]
        if len(canonical_indices) == 0:
            return canonical_smiles, [], invalid_smiles
        prediction = list()
        for index in canonical_indices:
            score_for_one = dock_and_get_score(smiles[index])
            prediction.append(score_for_one)
        return canonical_smiles, prediction, invalid_smiles


def dock_and_get_score(smiles):
    print(f"switch: {mlg_inst.switch}")
    print(f"frequency: {mlg_inst.switch_frequency}")
    print(f"logP: {mlg_inst.use_logP}")
    print(f"threshold: {mlg_inst.thresholds['LogP']}")
    try:
        # smiles = "C[C@@]12[C@](C[C@@H](O1)n3c4ccccc4c5c3c6n2c7ccccc7c6c8c5C(=O)NC8)(CO)O"
        python_path = "/workspace/mgltools_x86_64Linux2_1.5.6/bin/pythonsh"
        mgl_path = "/workspace/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24"
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        rdmolfiles.MolToPDBFile(mol, f"{mlg_inst.MOL_DIR}/{str(mlg_inst.mol_index)}.pdb")
        prepare_parameters = [python_path, f"{mgl_path}/prepare_ligand4.py", "-l",
                              f"{mlg_inst.MOL_DIR}/{str(mlg_inst.mol_index)}.pdb", "-o",
                              f"{mlg_inst.MOL_DIR}/{str(mlg_inst.mol_index)}.pdbqt"]
        subprocess.run(prepare_parameters, check=True)
        if mlg_inst.dock_mode == "vina":
            output = dock_vina()
        elif mlg_inst.dock_mode == "gpu4":
            output = dock_gpu()
        else:
            print("Unknown docking mode. Possible options: 'vina' and 'gpu'")
            return 0
        mlg_inst.mol_index += 1
        return output
    except Exception as e:
        print(smiles)
        mlg_inst.mol_index += 1
        print(f"Did Not Complete because of {e}")
        return 0


def dock_vina():
    vina_command = f"vina --config {mlg_inst.receptor}_config.txt --receptor {mlg_inst.receptor}.pdbqt "
    vina_command += f"--ligand {mlg_inst.MOL_DIR}/{str(mlg_inst.mol_index)}.pdbqt --seed {mlg_inst.seed} "
    if mlg_inst.vina_weights is not None:
        for key, value in mlg_inst.vina_weights.items():
            vina_command += f"--{key} {value}"
    os.system(vina_command)
    cmd = f"cat {mlg_inst.MOL_DIR}/{str(mlg_inst.mol_index)}_out.pdbqt | grep -i RESULT | tr -s '\t' ' ' | cut -d ' ' -f 4 | head -n1"
    stream = os.popen(cmd)
    output = float(stream.read().strip())
    return output


def dock_gpu():
    print(f"Docking molecule {mlg_inst.mol_index} of run {mlg_inst.run_id}")
    parameters_list = ["/workspace/AutoDock-GPU/bin/autodock_gpu_128wi", "-ffile", f"{mlg_inst.receptor}.maps.fld",
                       "-lfile", f"{mlg_inst.MOL_DIR}/{str(mlg_inst.mol_index)}.pdbqt", "-resnam",
                       f"{mlg_inst.LOGS_DIR}/{str(mlg_inst.mol_index)}", "--gbest", "1", "-nrun", "15", "-devnum" "1"]
    subprocess.run(parameters_list, check=True)
    cmd = f"cat {mlg_inst.LOGS_DIR}/{str(mlg_inst.mol_index)}.dlg | grep -i ranking | tr -s '\t' ' ' | cut -d ' ' -f 5 | head -n1"
    stream = os.popen(cmd)
    output = float(stream.read().strip())
    return output


class Generator:
    tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
              '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
              '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']
    
    def __init__(self, config,
                 device="GPU", 
                 gen_data='./random.smi',
                 use_checkpoint=True,
                 adaptive_reward=True,
                 use_logp=True,
                 use_qed=False, 
                 use_tpsa=False, 
                 use_solvation=False, 
                 switch=True,
                 tpsa_threshold=100, 
                 solvation_threshold=-10, 
                 qed_threshold=0.8,
                 logp_threshold=2.5,
                 switch_frequency=35
                 ):
        print("Started running MoleGuLAR")
        global mlg_inst
        os.chdir(os.path.abspath("../MoleGuLAR/Optimizer"))
        self.switch_frequency = switch_frequency
        self.thresholds = {
            'TPSA': tpsa_threshold,
            'LogP': logp_threshold,
            'solvation': solvation_threshold,
            'QED': qed_threshold
        }

        self.receptor = config["protein"]
        self.seed = config["seed"]
        if device == "CPU":
            self.device = torch.device('cpu')
            self.use_cuda = False
        elif device == "GPU":
            if torch.cuda.is_available():
                self.use_cuda = True
                self.device = torch.device('cuda:0')
            else:
                print("Sorry! GPU not available Using CPU instead")
                self.device = torch.device('cpu')
                self.use_cuda = False
        else:
            print("Invalid Device")
            quit(0)

        self.mol_index = 0
        self.use_wandb = config["wandb_logging"]

        if config["gen_reward"] == 'linear':
            self.get_reward = rwds.linear
        if config["gen_reward"] == 'exponential':
            self.get_reward = rwds.exponential
        if config["gen_reward"] == 'logarithmic':
            self.get_reward = rwds.logarithmic
        if config["gen_reward"] == 'squared':
            self.get_reward = rwds.squared

        self.use_qed = use_qed
        self.use_tpsa = use_tpsa
        self.use_solvation = use_solvation
        self.use_logP = use_logp

        self.smiles_file = config["generated_file"]

        self.get_reward = rwds.MultiReward(self.get_reward, True, use_logp, use_qed, use_tpsa, use_solvation,
                                           **self.thresholds)

        self.MODEL_NAME = f"./models/model_{config['gen_reward']}"
        self.LOGS_DIR = f"./logs_{config['gen_reward']}"
        self.MOL_DIR = f"./molecules_{config['gen_reward']}"

        self.TRAJ_FILE = None
        self.LOSS_FILE = f"./losses/{config['gen_reward']}"
        self.REWARD_FILE = f"./rewards/{config['gen_reward']}"
        self.PRED_FILE = f"./predictions/{config['gen_reward']}"

        self.gen_data_path = gen_data
        self.gen_data = GeneratorData(training_data_path=self.gen_data_path, delimiter='\t',
                                      cols_to_read=[0], keep_header=True, tokens=self.tokens)
        self.setup_directories(config['gen_reward'])
        self.predictor = Predictor("")
        self.model_path = './checkpoints/generator/checkpoint_biggest_rnn'
        if use_checkpoint and os.path.exists(f"./models/model_{config['gen_reward']}"):
            self.model_path = f"./models/model_{config['gen_reward']}"
        self.generator = None
        self.setup_generator()
        self.rl_instance = Reinforcement(self.generator, self.predictor, self.get_reward)
        self.n_to_generate = 100
        self.n_policy_replay = 10
        self.n_policy = 15
        self.n_iterations = config["gen_iterations"]
        self.reward_function = config['gen_reward']
        self.use_checkpoint = use_checkpoint
        self.adaptive_reward = adaptive_reward
        self.switch = switch
        self.logp_threshold = logp_threshold

        self.rewards = []
        self.rl_losses = []
        self.preds = []
        self.logp_iter = []
        self.solvation_iter = []
        self.qed_iter = []
        self.tpsa_iter = []
        self.solvation_predictor = FreeSolvPredictor('./Predictors/SolvationPredictor.tar')
        self.dock_mode = config['dock_de_novo']
        self.vina_weights = config["vina_weights"]
        self.c = config["grid_center"]
        self.s = config["grid_size"]
        self.sp = config["grid_spacing"]
        self.use_bias = config["use_bias"]
        self.run_id = config["run_id"]
        self.multi = config["de_novo_multi_mol"]
        self.biases = config["bias_data"]
        self.use_biases = False
        if self.biases is not None:
            if self.biases != list():
                self.use_biases = True
        mlg_inst = self

    @staticmethod
    def setup_directories(reward_function):
        if os.path.exists(f"./logs_{reward_function}"):
            shutil.rmtree(f"./logs_{reward_function}")
        os.mkdir(f"./logs_{reward_function}")
        if os.path.exists(f"./molecules_{reward_function}"):
            shutil.rmtree(f"./molecules_{reward_function}")
        os.mkdir(f"./molecules_{reward_function}")
        if not os.path.exists("./trajectories"):
            os.mkdir(f"./trajectories")
        if not os.path.exists("./rewards"):
            os.mkdir(f"./rewards")
        if not os.path.exists("./losses"):
            os.mkdir(f"./losses")
        if not os.path.exists("./models"):
            os.mkdir(f"./models")
        if not os.path.exists("./predictions"):
            os.mkdir("./predictions")

    def setup_generator(self):
        hidden_size = 1500
        stack_width = 1500
        stack_depth = 200
        layer_type = 'GRU'
        lr = 0.001
        optimizer_instance = torch.optim.Adadelta

        self.generator = StackAugmentedRNN(input_size=self.gen_data.n_characters,
                                           hidden_size=hidden_size,
                                           output_size=self.gen_data.n_characters,
                                           layer_type=layer_type,
                                           n_layers=1, is_bidirectional=False, has_stack=True,
                                           stack_width=stack_width, stack_depth=stack_depth,
                                           use_cuda=self.use_cuda,
                                           optimizer_instance=optimizer_instance, lr=lr)
        self.generator.load_model(self.model_path)

    def estimate_and_update(self):
        generated = []
        pbar = tqdm(range(self.n_to_generate))
        for _ in pbar:
            pbar.set_description("Generating molecules...")
            generated.append(self.generator.evaluate(self.gen_data, predict_len=120)[1:-1])
        sanitized = canonical_smiles(generated, sanitize=False, throw_warning=False)[:-1]
        unique_smiles = list(np.unique(sanitized))[1:]
        smiles, prediction, nan_smiles = self.predictor.predict(unique_smiles, use_tqdm=True)
        print("gen_predicted: ", len(smiles))
        return smiles, prediction

    @staticmethod
    def simple_moving_average(previous_values, new_value, ma_window_size=10):
        value_ma = np.sum(previous_values[-(ma_window_size - 1):]) + new_value
        value_ma = value_ma / (len(previous_values[-(ma_window_size - 1):]) + 1)
        return value_ma

    def evaluate_round(self, smiles_cur, prediction_cur):
        try:
            with open(self.smiles_file, "r") as f:
                content = f.readlines()
                print(f"Currently, there are {len(content)} smiles registered from the generator.")
        except FileNotFoundError:
            print("No smiles generated yet.")
        if len(smiles_cur) == 0:
            print("No valid smiles generated this round")
            return
        self.preds.append(sum(prediction_cur) / len(prediction_cur))
        logps = [MolLogP(Chem.MolFromSmiles(sm)) for sm in smiles_cur]
        tpsas = [CalcTPSA(Chem.MolFromSmiles(sm)) for sm in smiles_cur]
        qeds = list()
        for sm in smiles_cur:
            try:
                qeds.append(qed(Chem.MolFromSmiles(sm)))
            except:
                pass
        _, solvations, _ = self.solvation_predictor.predict(smiles_cur)

        smiles_to_save = list()
        for index, smile in enumerate(smiles_cur):
            smiles_to_save.append(smile + " " + str(prediction_cur[index]) + "\n")
        with open(self.smiles_file, "a+") as f:
            f.writelines(smiles_to_save)

        self.logp_iter.append(np.mean(logps))
        self.solvation_iter.append(np.mean(solvations))
        self.qed_iter.append(np.mean(qeds))
        self.tpsa_iter.append(np.mean(tpsas))
        print(f"BA: {self.preds[-1]}")
        print(f"LogP {self.logp_iter[-1]}")
        print(f"Hydration {self.solvation_iter[-1]}")
        print(f"TPSA {self.tpsa_iter[-1]}")
        print(f"QED {self.qed_iter[-1]}")
        self.rl_instance.generator.save_model(f"{self.MODEL_NAME}")

        if self.use_wandb:
            wandb.log({
                "loss": self.rewards[-1],
                "reward": self.rl_losses[-1],
                "predictions": self.preds[-1],
                "logP": sum(logps) / len(logps),
                "TPSA": sum(tpsas) / len(tpsas),
                "QED": sum(qeds) / len(qeds),
                "Solvation": sum(solvations) / len(solvations)
            })
            wandb.save(self.MODEL_NAME)
        np.savetxt(self.LOSS_FILE, self.rl_losses)
        np.savetxt(self.REWARD_FILE, self.rewards)
        np.savetxt(self.PRED_FILE, self.preds)

    def run_de_novo(self):
        if self.use_wandb:
            wandb.init(project=f"{self.reward_function}")
            # wandb.config.update(args)
        self.TRAJ_FILE = open(f"/workspace/MoleGuLAR/Optimizer/trajectories/traj_{self.reward_function}", "w")

        docking_active = False
        logp_active = True
        qed_active = False

        use_arr = np.array([False, False, True])
        for i in range(self.n_iterations):
            if self.switch:
                if self.use_logP and self.use_qed:
                    if i % self.switch_frequency == 0:
                        use_arr = np.roll(use_arr, 1)
                        docking_active, logp_active, qed_active = use_arr
                    self.get_reward = rwds.MultiReward(rwds.exponential, docking_active, logp_active, qed_active,
                                                       self.use_tpsa, self.use_solvation, **self.thresholds)
                if self.use_logP and not self.use_qed:
                    if i % self.switch_frequency == 0:
                        logp_active = not logp_active
                        docking_active = not docking_active
                    self.get_reward = rwds.MultiReward(rwds.exponential, docking_active, logp_active, qed_active,
                                                       self.use_tpsa, self.use_solvation, **self.thresholds)
            for _ in trange(self.n_policy, desc="Policy Gradient...."):
                if self.adaptive_reward:
                    cur_reward, cur_loss = self.rl_instance.policy_gradient(self.gen_data, self.get_reward,
                                                                            self.mol_index, multi=self.multi)
                else:
                    cur_reward, cur_loss = self.rl_instance.policy_gradient(self.gen_data, self.get_reward,
                                                                            multi=self.multi)
                self.rewards.append(self.simple_moving_average(self.rewards, cur_reward))
                self.rl_losses.append(self.simple_moving_average(self.rl_losses, cur_loss))

            smiles_cur, prediction_cur = self.estimate_and_update()
            self.evaluate_round(smiles_cur, prediction_cur)

        self.TRAJ_FILE.close()
        if self.use_wandb:
            wandb.save(self.MODEL_NAME)

        print("Finished running MoleGuLAR")
