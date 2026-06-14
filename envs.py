"""Environments used for proof-of-concept thinking experiments."""
import gym
import numpy as np
import random
import torch
import torch.nn as nn

from policies import init_weights

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



class CachedEncoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, max_window=-1):
        super().__init__()

        self.max_window = max_window

        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, kv_cache=None):
        """
        x: [B, 1, D] (new token only)
        kv_cache: previous hidden states for this layer
                  [B, T, D]
        """

        if kv_cache is None:
            kv = x
        else:
            kv = torch.cat([kv_cache, x], dim=1)
            if self.max_window > 0 and kv.shape[1] > self.max_window:
                kv = kv[:, 1:]

        attn_out, _ = self.self_attn(
            query=x,   # current token
            key=kv,    # all previous + current
            value=kv,
            need_weights=False,
        )

        x = self.norm1(x + attn_out)

        ff = self.ffn(x)
        x = self.norm2(x + ff)

        return x, kv


class TwoLayerTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=8):
        super().__init__()

        self.layer1 = CachedEncoderLayer(d_model, n_heads)
        self.layer2 = CachedEncoderLayer(d_model, n_heads)

    def forward(self, x, caches):
        """
        x: [B,1,D]

        caches = {
            "l1": layer1_cache,
            "l2": layer2_cache,
        }
        """

        x, new_l1 = self.layer1(x, caches["l1"])
        x, new_l2 = self.layer2(x, caches["l2"])

        new_caches = {
            "l1": new_l1,
            "l2": new_l2,
        }

        return x, new_caches


class TFAugmentedGridWorldEnv(gym.Env):
    def __init__(self, n_goals=2, deterministic_start=False, n_thought_acts=3, d_model=128, seed=42):
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
        self.d_model = d_model
        self.x_embedding = nn.Embedding(6, self.d_model // 8, padding_idx=0)
        self.y_embedding = nn.Embedding(6, self.d_model // 8, padding_idx=0)
        self.goal_embedding = nn.Embedding(3, self.d_model // 4, padding_idx=0)
        self.action_embedding = nn.Embedding(5 + n_thought_acts, self.d_model // 2, padding_idx=0)

        self.transformer = TwoLayerTransformer(self.d_model, n_heads=4)
        self.transformer.apply(init_weights)
        self.transformer.eval()

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
        self.thought = torch.zeros(self.d_model, dtype=torch.float32)
        self.cache = {
            "l1": None,
            "l2": None,
        }
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        return {
            "letter": self.letter,
            "position": self.agent_pos.copy(),
            "thought": self.thought.clone(),
        }

    def step(self, action):
        self.steps += 1

        if action < 5:  # Directional action
            if action > 0:
                delta = [(-1, 0), (1, 0), (0, -1), (0, 1)][action - 1]
                new_pos = self.agent_pos + np.array(delta)
                if np.all((1 <= new_pos) & (new_pos <= self.grid_size)):
                    self.agent_pos = new_pos
            self.thought = torch.zeros(self.d_model, dtype=torch.float32)
            self.cache = {
                "l1": None,
                "l2": None,
            }
        else: # Thought action
            with torch.no_grad():
                self.thought, self.cache = self.transformer(
                    torch.hstack((
                        self.x_embedding(torch.tensor(self.agent_pos[0])),
                        self.y_embedding(torch.tensor(self.agent_pos[1])),
                        self.goal_embedding(torch.tensor(self.letter)),
                        self.action_embedding(torch.tensor(action)),
                    ))[None, None],
                    self.cache
                )
                self.thought = self.thought[0, 0].detach()
                # self.thought = self.thought / torch.norm(self.thought, p=2)

        reward = 0.0
        done = False
        if tuple(self.agent_pos) == self.goal:
            reward = 1.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}


class TFAugmentedGridWorldEnv2(gym.Env):
    def __init__(self, n_goals=2, deterministic_start=False, n_thought_acts=3, n_thought_states=10, d_model=128, seed=42):
        super(TFAugmentedGridWorldEnv2, self).__init__()
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
        self.d_model = d_model
        self.x_embedding = nn.Embedding(6, self.d_model // 4, padding_idx=0)
        self.y_embedding = nn.Embedding(6, self.d_model // 4, padding_idx=0)
        self.goal_embedding = nn.Embedding(3, self.d_model // 2, padding_idx=0)
        self.action_embedding = nn.Embedding(5 + n_thought_acts, self.d_model, padding_idx=0)
        self.thought_state_embedding = nn.Linear(self.d_model, n_thought_states)
        self.thought_state_embedding.apply(init_weights)
        self.lnorm = nn.LayerNorm(d_model)
        self.lnorm.apply(init_weights)

        self.transformer = TwoLayerTransformer(self.d_model * 3, n_heads=4)
        self.transformer.apply(init_weights)
        self.transformer.eval()

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
        self.thought = torch.zeros(self.d_model, dtype=torch.float32)
        self.cache = {
            "l1": None,
            "l2": None,
        }
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        return {
            "letter": self.letter,
            "position": self.agent_pos.copy(),
            "thought": self.thought.clone(),
        }

    def step(self, action):
        self.steps += 1

        if action < 5:  # Directional action
            if action > 0:
                delta = [(-1, 0), (1, 0), (0, -1), (0, 1)][action - 1]
                new_pos = self.agent_pos + np.array(delta)
                if np.all((1 <= new_pos) & (new_pos <= self.grid_size)):
                    self.agent_pos = new_pos
            self.thought = torch.zeros(self.d_model, dtype=torch.float32)
            self.cache = {
                "l1": None,
                "l2": None,
            }
        else: # Thought action
            with torch.no_grad():
                self.thought, self.cache = self.transformer(
                    torch.hstack((
                        self.x_embedding(torch.tensor(self.agent_pos[0])),
                        self.y_embedding(torch.tensor(self.agent_pos[1])),
                        self.goal_embedding(torch.tensor(self.letter)),
                        self.action_embedding(torch.tensor(action)),
                        self.thought,
                    ))[None, None],
                    self.cache
                )
                self.thought = self.thought_state_embedding(
                    self.lnorm(self.thought[..., -self.d_model:]
                ))
                # self.thought = self.thought / torch.norm(self.thought, p=2)
                self.thought = self.thought_state_embedding.weight[torch.argmax(self.thought[0, 0])].detach()

        reward = 0.0
        done = False
        if tuple(self.agent_pos) == self.goal:
            reward = 1.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}


class TFAugmentedGridWorldEnv3(gym.Env):
    def __init__(self, n_goals=2, deterministic_start=False, n_thought_acts=3, n_thought_states=10, d_model=128, seed=42):
        super(TFAugmentedGridWorldEnv3, self).__init__()
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
        self.d_model = d_model
        self.x_embedding = nn.Embedding(6, self.d_model // 4, padding_idx=0)
        self.y_embedding = nn.Embedding(6, self.d_model // 4, padding_idx=0)
        self.goal_embedding = nn.Embedding(3, self.d_model // 2, padding_idx=0)
        self.action_embedding = nn.Embedding(5 + n_thought_acts, self.d_model, padding_idx=0)
        self.thought_state_embedding = nn.Linear(self.d_model, n_thought_states)
        self.thought_state_embedding.apply(init_weights)
        self.lnorm = nn.LayerNorm(d_model)
        self.lnorm.apply(init_weights)

        self.transformer = TwoLayerTransformer(self.d_model, n_heads=4)
        self.transformer.apply(init_weights)
        self.transformer.eval()

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
        self.steps = 0
        self.thought = torch.zeros(self.d_model, dtype=torch.float32)
        self.cache = {
            "l1": None,
            "l2": None,
        }

        obs = self._get_obs()

        self.thought, self.cache = self.transformer(
            torch.hstack((
                self.x_embedding(torch.tensor(self.agent_pos[0])),
                self.y_embedding(torch.tensor(self.agent_pos[1])),
                self.goal_embedding(torch.tensor(self.letter)),
            ))[None, None],
            self.cache
        )
        self.thought = self.thought.detach()[0, 0]

        return obs

    def _get_obs(self):
        return {
            "letter": self.letter,
            "position": self.agent_pos.copy(),
            "thought": self.thought.clone(),
        }

    def step(self, action):
        self.steps += 1

        if action < 5:  # Directional action
            if action > 0:
                delta = [(-1, 0), (1, 0), (0, -1), (0, 1)][action - 1]
                new_pos = self.agent_pos + np.array(delta)
                if np.all((1 <= new_pos) & (new_pos <= self.grid_size)):
                    self.agent_pos = new_pos
            self.thought = torch.zeros(self.d_model, dtype=torch.float32)
            self.cache = {
                "l1": None,
                "l2": None,
            }

            obs = self._get_obs()

            self.thought, self.cache = self.transformer(
                torch.hstack((
                    self.x_embedding(torch.tensor(self.agent_pos[0])),
                    self.y_embedding(torch.tensor(self.agent_pos[1])),
                    self.goal_embedding(torch.tensor(self.letter)),
                ))[None, None],
                self.cache
            )
            self.thought = self.thought.detach()[0, 0]
        else: # Thought action
            with torch.no_grad():
                self.thought, self.cache = self.transformer(
                    self.action_embedding(torch.tensor(action))[None, None],
                    self.cache,
                )
                self.thought, self.cache = self.transformer(
                    self.thought,
                    self.cache,
                )
                self.thought = self.thought_state_embedding(
                    self.lnorm(self.thought)
                )
                self.thought = self.thought_state_embedding.weight[torch.argmax(self.thought[0, 0])].detach()
                # self.thought = self.thought.detach()[0, 0]
                # # self.thought = self.thought / torch.norm(self.thought, p=2)
            obs = self._get_obs()

        reward = 0.0
        done = False
        if tuple(self.agent_pos) == self.goal:
            reward = 1.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        return obs, reward, done, {}


class TFAugmentedGridWorldEnv4(gym.Env):
    def __init__(self, n_goals=2, deterministic_start=False, n_thought_acts=3, n_thought_states=10, d_model=128, seed=42):
        super(TFAugmentedGridWorldEnv4, self).__init__()
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
        self.d_model = d_model
        self.x_embedding = nn.Embedding(6, self.d_model // 4, padding_idx=0)
        self.y_embedding = nn.Embedding(6, self.d_model // 4, padding_idx=0)
        self.goal_embedding = nn.Embedding(3, self.d_model // 2, padding_idx=0)
        self.action_embedding = nn.Embedding(5 + n_thought_acts, self.d_model, padding_idx=0)
        self.thought_state_embedding = nn.Linear(self.d_model, n_thought_states)
        self.lnorm = nn.LayerNorm(d_model)
        self.lnorm.apply(init_weights)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dropout=0.0,
            dim_feedforward=self.d_model * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)
        self.transformer.apply(init_weights)
        self.transformer.eval()

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
        self.steps = 0
        self.thought = torch.zeros(self.d_model, dtype=torch.float32)

        return self._get_obs()

    def _get_obs(self):
        return {
            "letter": self.letter,
            "position": self.agent_pos.copy(),
            "thought": self.thought.clone(),
        }

    def step(self, action):
        self.steps += 1

        if action < 5:  # Directional action
            if action > 0:
                delta = [(-1, 0), (1, 0), (0, -1), (0, 1)][action - 1]
                new_pos = self.agent_pos + np.array(delta)
                if np.all((1 <= new_pos) & (new_pos <= self.grid_size)):
                    self.agent_pos = new_pos
            self.thought = torch.zeros(self.d_model, dtype=torch.float32)

        else: # Thought action
            with torch.no_grad():
                self.thought = self.transformer(
                    torch.cat((
                        torch.hstack((
                            self.x_embedding(torch.tensor(self.agent_pos[0])),
                            self.y_embedding(torch.tensor(self.agent_pos[1])),
                            self.goal_embedding(torch.tensor(self.letter)),
                        ))[None, None],
                        self.action_embedding(torch.tensor(action))[None, None],
                        self.thought[None, None],
                    ), dim=1)
                )
                self.thought = self.thought_state_embedding(
                    self.lnorm(self.thought)
                )
                self.thought = self.thought_state_embedding.weight[torch.argmax(self.thought[0, -1])].detach()

        reward = 0.0
        done = False
        if tuple(self.agent_pos) == self.goal:
            reward = 1.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}
