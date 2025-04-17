import os
import sys
from RAscore_NN import Switch


class SynthiInterface:
    def __init__(self):
        self.torch_switcher = Switch(pytorch_model=None,
                                     model_state_path="/workspace/RAscore/RAscore/models/synthi_weights.pth")
        self.torch_switcher.pytorch_model.eval()

    def predict_synthesizability(self, smiles):
        synthi_score = self.torch_switcher.predict(smiles)
        return round(synthi_score, 4)
