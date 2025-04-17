"""
MIT License

Copyright (c) 2020 Reymond Research Group, University of Bern

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

------
Original file from RAscore
A. Thakkar et Al.: Retrosynthetic accessibility score (RAscore) – rapid machine learned synthesizability classification from AI driven retrosynthetic planning
Chem. Sci., 2021, 12, 3339—3349 

The present file has been modified for use in the DNS-HIT pipeline.
"""

import os
from zipfile import ZipFile
import numpy as np


from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.DataStructs import cDataStructs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


optimized = {
    "layer_1": 512, "activation_1": "linear", "num_layers": 10,
    "units_2": 512, "activation_2": "relu", "dropout_2": 0.45834579304621176,
    "units_3": 128, "activation_3": "linear", "dropout_3": 0.20214636121010582,
    "units_4": 512, "activation_4": "elu", "dropout_4": 0.13847113009081813,
    "units_5": 256, "activation_5": "linear", "dropout_5": 0.21312873496871235,
    "units_6": 128, "activation_6": "relu", "dropout_6": 0.33530504087548707,
    "units_7": 128, "activation_7": "linear", "dropout_7": 0.11559123444807062,
    "units_8": 128, "activation_8": "relu", "dropout_8": 0.2618908919792556,
    "units_9": 512, "activation_9": "relu", "dropout_9": 0.3587291059530903,
    "units_10": 512, "activation_10": "selu", "dropout_10": 0.43377277017943133,
    "learning_rate": 1.5691774834712003e-05
}


class PyTorchModel(nn.Module):
    def __init__(self, input_size=2048, best_params=optimized):
        super(PyTorch_Model, self).__init__()
        self.input_size = input_size
        self.num_layers = best_params['num_layers']
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, best_params['layer_1']))
        self.layers.append(self.get_activation(best_params['activation_1']))

        for i in range(2, self.num_layers + 1):
            if i == 2:
                self.layers.append(nn.Linear(best_params[f'layer_{l - 1}'], best_params[f'units_{l}']))
            else:
                self.layers.append(nn.Linear(best_params[f'units_{l - 1}'], best_params[f'units_{l}']))
            self.layers.append(self.get_activation(best_params[f'activation_{l}']))
            self.layers.append(nn.Dropout(p=best_params[f'dropout_{l}']))
        self.out = nn.Linear(best_params[f'units_{self.num_layers}'], 1)
        self.out_activation = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        x = self.out_activation(x)
        return x

    def get_activation(self, name):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'linear':
            return nn.Identity()
        elif name == 'elu':
            return nn.ELU()
        elif name == 'selu':
            return nn.SELU()
        else:
            raise NotImplementedError(f"Activation {name} is not implemented.")

    def load_weights(self, weights_dict):
        # Assuming weights_dict is a dictionary containing weights for each layer
        for name, param in self.named_parameters():
            if name in weights_dict:
                param.data = torch.tensor(weights_dict[name])

# from https://www.tensorflow.org/api_docs/python/tf/keras/activations
# tensorflow:pytorch
# linear: do nothing
# relu: F.relu
# elu: F.elu
# selu: F.selu
# sigmoid: torch.sigmoid


class PyTorchModel2(nn.Module):
    """
    The model is build after the tensorflow.summary().
    The dropout chances are just approximated.
    Weights are transferred at the end of the cell explicitly.
    Therefore take care if the model changes, weight transfer
    might break.
    Activation functions from cell before.
    Dropout layers in init for .eval() behaviour
    """
    def __init__(self):
        super(PyTorchModel2, self).__init__()
        self.dense1 = nn.Linear(2048, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dense3 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dense4 = nn.Linear(128, 512)
        self.dropout3 = nn.Dropout(p=0.5)
        self.dense5 = nn.Linear(512, 256)
        self.dropout4 = nn.Dropout(p=0.5)
        self.dense6 = nn.Linear(256, 128)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dense7 = nn.Linear(128, 128)
        self.dropout6 = nn.Dropout(p=0.5)
        self.dense8 = nn.Linear(128, 128)
        self.dropout7 = nn.Dropout(p=0.5)
        self.dense9 = nn.Linear(128, 512)
        self.dropout8 = nn.Dropout(p=0.5)
        self.dense10 = nn.Linear(512, 512)
        self.dropout9 = nn.Dropout(p=0.5)
        self.target = nn.Linear(512, 1)

    def forward(self, x):
        x = self.dense1(x)
        x = F.relu(self.dense2(x))
        x = self.dropout1(x)
        x = self.dense3(x)
        x = self.dropout2(x)
        x = F.elu(self.dense4(x))  # Using ELU activation as specified in TensorFlow model
        x = self.dropout3(x)
        x = self.dense5(x)
        x = self.dropout4(x)
        x = F.relu(self.dense6(x))
        x = self.dropout5(x)
        x = self.dense7(x)
        x = self.dropout6(x)
        x = F.relu(self.dense8(x))
        x = self.dropout7(x)
        x = F.relu(self.dense9(x))
        x = self.dropout8(x)
        x = torch.selu(self.dense10(x))  # Using SELU activation as specified in TensorFlow model
        x = self.dropout9(x)
        x = torch.sigmoid(self.target(x))  # Using sigmoid activation for the output layer
        return x


class RAScorerNN(PyTorchModel):
    def __init__(self, model_path=None):
        super(RAScorerNN, self).__init__()
        here = os.path.abspath(os.path.dirname(__file__))
        model = os.path.join(here, "models/synthi_weights.pth")
        if model_path is None:
            self.nn_model = torch.load(model)
        else:
            self.nn_model = torch.load(model_path)
        self.load_weights(self.nn_model)

    @staticmethod
    def ecfp_counts(smiles):
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
        size = 2048
        arr = np.zeros((size,), np.int32)
        for idx, v in fp.GetNonzeroElements().items():
            nidx = idx % size
            arr[nidx] += int(v)
        return arr

    def predict(self, smiles):
        try:
            arr = self.ecfp_counts(smiles)
        except ValueError:
            print("SMILES could not be converted to ECFP6 count vector")
            return float("NaN")
        try:
            proba = self.forward(torch.tensor(arr.reshape(1, -1), dtype=torch.float32))
            return proba[0][0]
        except:
            print("Prediction not possible")
            return float("NaN")


class Switch:
    def __init__(self, pytorch_model=None, model_state_path=None):
        self.pytorch_model = pytorch_model if pytorch_model is not None else PyTorchModel2()
        self.model_state_path = model_state_path
        if self.model_state_path is not None:
            self.pytorch_model.load_state_dict(torch.load(self.model_state_path))
            self.pytorch_model.eval()

    @staticmethod
    def ecfp_counts(smiles):
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
        size = 2048
        arr = np.zeros((size,), np.int32)
        for idx, v in fp.GetNonzeroElements().items():
            nidx = idx % size
            arr[nidx] += int(v)
        return arr

    def predict(self, smiles):
        try:
            num_arr = self.ecfp_counts(smiles)
        except ValueError:
            print("SMILES could not be converted to ECFP6 count vector")
            return float("NaN")
        # convert to tensor and fix dtype
        tensor_input = torch.tensor(num_arr, dtype=torch.float32)
        try:
            with torch.no_grad():
                output = self.pytorch_model(tensor_input)
            return output.item()  # Assuming the output is a single value
        except Exception as e:
            print("Prediction not possible")
            import warnings
            warnings.warn(f"Catched exception: {e}")
            return float("NaN")
