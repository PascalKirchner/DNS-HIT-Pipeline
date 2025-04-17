"""
This class implements simple policy gradient algorithm for
biasing the generation of molecules towards desired values of
properties aka Reinforcement Learning for Structural Evolution (ReLeaSE)
as described in 
Popova, M., Isayev, O., & Tropsha, A. (2018). 
Deep reinforcement learning for de novo drug design. 
Science advances, 4(7), eaap7885.

The present file has been modified by P. Kirchner for use in the DNS-HIT pipeline.
"""

import torch
import torch.nn.functional as F
import numpy as np
from rdkit import Chem


class Reinforcement(object):
    def __init__(self, generator, predictor, get_reward):
        """
        Constructor for the Reinforcement object.

        Parameters
        ----------
        generator: object of type StackAugmentedRNN
            generative model that produces string of characters (trajectories)

        predictor: object of any predictive model type
            predictor accepts a trajectory and returns a numerical
            prediction of desired property for the given trajectory

        get_reward: function
            custom reward function that accepts a trajectory, predictor and
            any number of positional arguments and returns a single value of
            the reward for the given trajectory
            Example:
            reward = get_reward(trajectory=my_traj, predictor=my_predictor,
                                custom_parameter=0.97)

        Returns
        -------
        object of type Reinforcement used for biasing the properties estimated
        by the predictor of trajectories produced by the generator to maximize
        the custom reward function get_reward.
        """

        super(Reinforcement, self).__init__()
        self.generator = generator
        self.predictor = predictor
        self.get_reward = get_reward

    def preselect_molecule(self, multi=1):
        valid = list()
        if multi < 1:
            multi = 1
        elif multi > 20:
            multi = 20
        while len(valid) < multi:
            trajectory = self.generator.evaluate(data)
            try:
                mol = Chem.MolFromSmiles(trajectory[1:-1])
                trajectory = '<' + Chem.MolToSmiles(mol) + '>'
                valid[trajectory[1:-1]] = 0
            except:
                pass

        for molecule in valid.keys():
            formula = dict()
            mol = Chem.MolFromSmiles(molecule)
            for atom in mol.GetAtoms():
                atom_type = atom.GetSymbol()
                if atom_type in formula.keys():
                    formula[atom_type] += 1
                else:
                    formula[atom_type] = 1
            for atom, number in formula.items():
                if atom in ("O", "N"):
                    valid[molecule] += number
                elif atom in ("F", "Cl", "Br", "I", "S"):
                    valid[molecule] += 2 * number
            for entry in ("#N", "N#", "OH", "HO", "=O", "O="):
                number = len(molecule.split(entry)) - 1
                valid[molecule] += number

        best_molecule = None
        best_score = 0
        for molecule, score in valid.items():
            if score > best_score:
                best_molecule = molecule
        return best_molecule

    def policy_gradient(self, data, reward_func, OVERALL_INDEX=0, n_batch=10, gamma=0.97,
                        std_smiles=False, grad_clipping=None, multi=1, **kwargs):
        """
        Implementation of the policy gradient algorithm.

        Parameters:
        -----------

        data: object of type GeneratorData
            stores information about the generator data format such alphabet, etc

        n_batch: int (default 10)
            number of trajectories to sample per batch. When training on GPU
            setting this parameter to to some relatively big numbers can result
            in out of memory error. If you encountered such an error, reduce
            n_batch.

        gamma: float (default 0.97)
            factor by which rewards will be discounted within one trajectory.
            Usually this number will be somewhat close to 1.0.


        std_smiles: bool (default False)
            boolean parameter defining whether the generated trajectories will
            be converted to standardized SMILES before running policy gradient.
            Leave this parameter to the default value if your trajectories are
            not SMILES.

        grad_clipping: float (default None)
            value of the maximum norm of the gradients. If not specified,
            the gradients will not be clipped.

        kwargs: any number of other positional arguments required by the
            get_reward function.

        Returns
        -------
        total_reward: float
            value of the reward averaged through n_batch sampled trajectories

        rl_loss: float
            value for the policy_gradient loss averaged through n_batch sampled
            trajectories

        """
        rl_loss = 0
        self.get_reward = reward_func
        self.generator.optimizer.zero_grad()
        total_reward = 0
        
        for _ in range(n_batch):

            # Sampling new trajectory
            reward = 0
            trajectory = '<>'
            while reward == 0:
                trajectory = self.generator.evaluate(data)
                if std_smiles:
                    try:
                        molecule = self.preselect_molecule(multi)
                        reward = self.get_reward(molecule,
                                                 self.predictor,
                                                 -5.0, OVERALL_INDEX)
                    except:
                        reward = 0
                else:
                    reward = self.get_reward(trajectory[1:-1],
                                             self.predictor, 
                                             -5.0, OVERALL_INDEX)

            # Converting string of characters into tensor
            trajectory_input = data.char_tensor(trajectory)
            discounted_reward = reward
            total_reward += reward

            # Initializing the generator's hidden state
            hidden = self.generator.init_hidden()
            if self.generator.has_cell:
                cell = self.generator.init_cell()
                hidden = (hidden, cell)
            if self.generator.has_stack:
                stack = self.generator.init_stack()
            else:
                stack = None

            # "Following" the trajectory and accumulating the loss
            for p in range(len(trajectory)-1):
                output, hidden, stack = self.generator(trajectory_input[p], 
                                                       hidden, 
                                                       stack)
                log_probs = F.log_softmax(output, dim=1)
                top_i = trajectory_input[p+1]
                rl_loss -= (log_probs[0, top_i]*discounted_reward)
                discounted_reward = discounted_reward * gamma

        # Doing backward pass and parameters update
        rl_loss = rl_loss / n_batch
        total_reward = total_reward / n_batch
        rl_loss.backward()
        if grad_clipping is not None:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 
                                           grad_clipping)

        self.generator.optimizer.step()
        
        return total_reward, rl_loss.item()
