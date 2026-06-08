"""Environments used for proof-of-concept thinking experiments."""
import gym
import numpy as np
import random
import torch
import torch.nn as nn


# BASE SETTING
# GRID_SIZE = 5
# MAX_STEPS = 20

GRID_SIZE = 5
# MAX_STEPS = (GRID_SIZE // 2) * 2 + 2 * GRID_SIZE + 6
MAX_STEPS = GRID_SIZE * 2 * 10


class GridWorldEnv(gym.Env):
    def __init__(self, n_goals=2, deterministic_start=False, n_thought_acts=3, seed=42):
        super(GridWorldEnv, self).__init__()
        self.rng = np.random.RandomState(seed)
        self.grid_size = GRID_SIZE
        self.max_steps = MAX_STEPS
        self.deterministic_start = deterministic_start
        self.letter_goals = {
            1: (self.grid_size, self.grid_size),  # 'A'
        }
        if n_goals > 1:
            self.letter_goals[2] = (1, 1)  # 'C'
        self.letters = list(self.letter_goals.keys())
        self.action_meanings = ["UP", "DOWN", "LEFT", "RIGHT"] + [f"THOUGHT_{ii}" for ii in range(n_thought_acts)]
        self.action_space = gym.spaces.Discrete(len(self.action_meanings))
        self.observation_space = gym.spaces.Dict(
            {
                "letter": gym.spaces.Discrete(len(self.letters)),  # 0 = A, 1 = B, 2 = C
                "position": gym.spaces.Box(
                    low=1, high=self.grid_size, shape=(2,), dtype=np.int32
                ),
            }
        )

    def reset(self):
        if self.deterministic_start:
            self.agent_pos = [self.grid_size // 2 + 1, self.grid_size // 2 + 1]
        else:
            self.agent_pos = np.array(
                [
                    self.rng.randint(1, self.grid_size),
                    self.rng.randint(1, self.grid_size),
                ]
            )
        self.letter = self.rng.choice(self.letters)
        self.goal = self.letter_goals[self.letter]
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        # return {"letter": self.letter, "position": self.agent_pos.copy()}
        return {"position": np.concatenate((self.agent_pos.copy(), [self.letter]))}

    def step(self, action):
        self.steps += 1
        if action < 5:  # Directional action
            delta = [(-1, 0), (1, 0), (0, -1), (0, 1)][action - 1]
            new_pos = self.agent_pos + np.array(delta)
            if np.all((1 <= new_pos) & (new_pos <= self.grid_size)):
                self.agent_pos = new_pos

        reward = 0.0
        done = False
        if tuple(self.agent_pos) == self.goal:
            reward = 1.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}

    def generate_expert_action(self):
        goal = np.array(self.goal)
        act = 5
        direction = goal - self.agent_pos
        if direction[0] < 0:
            act = 1
        elif direction[0] > 0:
            act = 2
        elif direction[1] < 0:
            act = 3
        elif direction[1] > 0:
            act = 4
        return act


class TwoStageGridWorldEnv(GridWorldEnv):
    def __init__(self, inexact=False):
        super(TwoStageGridWorldEnv, self).__init__()
        self.grid_size = GRID_SIZE
        self.max_steps = MAX_STEPS
        self.deterministic_start = True
        if inexact:
            self.letter_goals = {
                5: [(self.grid_size, self.grid_size - 1)],  # 'A'
                6: [(2, 1)],  # 'B'
            }
        else:
            self.letter_goals = {
                5: [(self.grid_size, self.grid_size)],  # 'A'
                6: [(1, 1)],  # 'B'
            }
        # Goal for "C" (7) is to do "A" and then "B"
        self.letter_goals[7] = [self.letter_goals[5][0], self.letter_goals[6][0]]
        self.letters = list(self.letter_goals.keys())
        self.action_meanings = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "C"]
        self.action_space = gym.spaces.Discrete(len(self.action_meanings))
        self.observation_space = gym.spaces.Dict(
            {
                "letter": gym.spaces.Discrete(len(self.letters)),
                "position": gym.spaces.Box(
                    low=1, high=self.grid_size, shape=(2,), dtype=np.int32
                ),
            }
        )

    def reset(self):
        if self.deterministic_start:
            self.agent_pos = np.array(
                [self.grid_size // 2 + 1, self.grid_size // 2 + 1]
            )
        else:
            self.agent_pos = np.array(
                [
                    np.random.randint(1, self.grid_size),
                    np.random.randint(1, self.grid_size),
                ]
            )
        self.letter = 7
        self.goal = self.letter_goals[self.letter]
        self.steps = 0
        self.goal_ind = 0
        return self._get_obs()

    def _get_obs(self):
        return {"letter": self.letter, "position": self.agent_pos.copy()}

    def step(self, action):
        self.steps += 1
        if action < 5 and action > 0:  # Directional action, action 0 is pad value
            delta = [(-1, 0), (1, 0), (0, -1), (0, 1)][action - 1]
            new_pos = self.agent_pos + np.array(delta)
            if np.all((1 <= new_pos) & (new_pos <= self.grid_size)):
                self.agent_pos = new_pos

        reward = 0.0
        done = False
        if tuple(self.agent_pos) == self.goal[self.goal_ind]:
            self.goal_ind += 1
            if len(self.goal) == self.goal_ind:
                reward = 1.0
                done = True

        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}

    def generate_expert_action(self):
        goal = np.array(self.goal[self.goal_ind])
        act = 5

        direction = goal - self.agent_pos
        if direction[0] < 0:
            act = 1
        elif direction[0] > 0:
            act = 2
        elif direction[1] < 0:
            act = 3
        elif direction[1] > 0:
            act = 4
        return act


class PlayGridWorldEnv(gym.Env):
    def __init__(self):
        super(PlayGridWorldEnv, self).__init__()
        self.grid_size = GRID_SIZE  # 3
        self.max_steps = MAX_STEPS  # 10
        self.deterministic_start = False
        self.letter_goals = {
            5: (self.grid_size, self.grid_size),  # 'A'
            6: (1, 1),  # 'B'
        }
        self.letters = list(self.letter_goals.keys())
        self.action_meanings = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "C"]
        self.action_space = gym.spaces.Discrete(len(self.action_meanings))
        self.observation_space = gym.spaces.Dict(
            {
                "letter": gym.spaces.Discrete(len(self.letters)),  # 0 = A, 1 = B, 2 = C
                "position": gym.spaces.Box(
                    low=1, high=self.grid_size, shape=(2,), dtype=np.int32
                ),
            }
        )

    def reset(self):
        if self.deterministic_start:
            self.agent_pos = [self.grid_size // 2 + 1, self.grid_size // 2 + 1]
        else:
            self.agent_pos = np.array(
                [
                    np.random.randint(1, self.grid_size),
                    np.random.randint(1, self.grid_size),
                ]
            )
        self.letter = random.choice(self.letters)
        self.goal = self.letter_goals[self.letter]
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        return {"letter": self.letter, "position": self.agent_pos.copy()}

    def step(self, action):
        self.steps += 1
        if action < 5:  # Directional action
            delta = [(-1, 0), (1, 0), (0, -1), (0, 1)][action - 1]
            new_pos = self.agent_pos + np.array(delta)
            if np.all((1 <= new_pos) & (new_pos <= self.grid_size)):
                self.agent_pos = new_pos

        if action >= 5:
            self.letter = action
            self.goal = self.letter_goals[self.letter]

        reward = 0.0
        done = False
        if tuple(self.agent_pos) == self.goal:
            reward = 1.0
            # done = True

        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}

    def generate_expert_action(self):
        goal = np.array(self.goal)
        act = 5
        direction = goal - self.agent_pos
        if direction[0] < 0:
            act = 1
        elif direction[0] > 0:
            act = 2
        elif direction[1] < 0:
            act = 3
        elif direction[1] > 0:
            act = 4
        else:
            act = np.random.choice([5, 6])
        # Re-sample goal with probability 0.2
        act = np.random.choice([act, 5, 6], p=[0.8, 0.1, 0.1])
        return act


class DebugEnv(GridWorldEnv):
    def __init__(self):
        self.letter = 5
        self.agent_pos = [1, 1]

    def reset(self):
        return self._get_obs()

    def step(self, action):
        if action == 5:
            reward = 1
        else:
            reward = 0

        return self._get_obs(), reward, True, {}



class TFAugmentedGridWorldEnv(gym.Env):
    def __init__(self, n_goals=2, deterministic_start=False, n_thought_acts=3, seed=42):
        super(TFAugmentedGridWorldEnv, self).__init__()
        self.rng = np.random.RandomState(seed)
        self.grid_size = GRID_SIZE
        self.max_steps = MAX_STEPS
        self.deterministic_start = deterministic_start
        self.letter_goals = {
            1: (self.grid_size, self.grid_size),  # 'A'
        }
        if n_goals > 1:
            self.letter_goals[2] = (1, 1)  # 'C'
        self.letters = list(self.letter_goals.keys())
        self.action_meanings = ["UP", "DOWN", "LEFT", "RIGHT"] + [f"THOUGHT_{ii}" for ii in range(n_thought_acts)]
        self.action_space = gym.spaces.Discrete(len(self.action_meanings))

        # Thought space
        self.d_model = 128
        self.x_embedding = nn.Embedding(6, self.d_model // 4, padding_idx=0)
        self.y_embedding = nn.Embedding(6, self.d_model // 4, padding_idx=0)
        self.goal_embedding = nn.Embedding(3, self.d_model // 2, padding_idx=0)
        self.action_embedding = nn.Embedding(5 + n_thought_acts, self.d_model, padding_idx=0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=3 * self.d_model, nhead=4, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)

        self.observation_space = gym.spaces.Dict(
            {
                "letter": gym.spaces.Discrete(len(self.letters)),  # 0 = A, 1 = B, 2 = C
                "position": gym.spaces.Box(
                    low=1, high=self.grid_size, shape=(2,), dtype=np.int32
                ),
                "thought": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.d_model,), dtype=np.float32
                )
            }
        )

    def reset(self):
        if self.deterministic_start:
            self.agent_pos = [self.grid_size // 2 + 1, self.grid_size // 2 + 1]
        else:
            self.agent_pos = np.array(
                [
                    self.rng.randint(1, self.grid_size),
                    self.rng.randint(1, self.grid_size),
                ]
            )
        self.letter = self.rng.choice(self.letters)
        self.goal = self.letter_goals[self.letter]
        self.thought = np.zeros(self.d_model, dtype=np.float32)
        self.tape = []
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        return {
            "letter": self.letter,
            "position": self.agent_pos.copy(),
            "thought": self.thought.copy(),
        }

    def step(self, action):
        self.steps += 1

        with torch.no_grad():
            self.tape.append(np.hstack((
                self.x_embedding(torch.tensor(self.agent_pos[0])).detach().numpy(),
                self.y_embedding(torch.tensor(self.agent_pos[1])).detach().numpy(),
                self.goal_embedding(torch.tensor(self.letter)).detach().numpy(),
                self.action_embedding(torch.tensor(action)).detach().numpy(),
                self.thought,
            ))[None, None])

        if action < 5:  # Directional action
            delta = [(-1, 0), (1, 0), (0, -1), (0, 1)][action - 1]
            new_pos = self.agent_pos + np.array(delta)
            if np.all((1 <= new_pos) & (new_pos <= self.grid_size)):
                self.agent_pos = new_pos
            self.thought = np.zeros(self.d_model)
        else: # Thought action
            with torch.no_grad():
                self.thought = self.transformer(torch.cat(self.tape, dim=1))[0, -1].detach().numpy()

        reward = 0.0
        done = False
        if tuple(self.agent_pos) == self.goal:
            reward = 1.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}

    def generate_expert_action(self):
        goal = np.array(self.goal)
        act = 5
        direction = goal - self.agent_pos
        if direction[0] < 0:
            act = 1
        elif direction[0] > 0:
            act = 2
        elif direction[1] < 0:
            act = 3
        elif direction[1] > 0:
            act = 4
        return act
