"""Environments used for proof-of-concept thinking experiments."""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from policies import init_weights


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


class TFAugmentedFrozenLakeEnv(gym.Env):
    def __init__(self, n_thought_acts=3, n_thought_states=10, d_model=128, max_steps=50, tabular=False):
        super(TFAugmentedFrozenLakeEnv, self).__init__()
        self.tabular = tabular
        self.d_model = d_model
        self.base_env = gym.make(
            'FrozenLake-v1',
            desc=None,
            map_name="4x4",
            is_slippery=False,
            success_rate=1.0/3.0,
            reward_schedule=(1, -1, 0),
            max_episode_steps=-1,
        )

        self.max_steps = max_steps
        self.observation_space = gym.spaces.Dict(
            {
                "env": gym.spaces.Box(low=0, high=1, shape=(16,), dtype=int) if tabular else self.base_env.observation_space,
                "thought": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.d_model,), dtype=np.float32
                ),
            }
        )
        self.obs_dim = 16 if tabular else 1
        self.n_acts = self.base_env.action_space.n
        self.action_space = gym.spaces.Discrete(self.base_env.action_space.n + n_thought_acts)

        # Thought space
        self.pos_embedding = nn.Embedding(self.base_env.observation_space.n, self.d_model)
        self.action_embedding = nn.Embedding(self.action_space.n + 1, self.d_model, padding_idx=0)
        self.thought_state_embedding = nn.Linear(self.d_model, n_thought_states)
        self.thought_state_embedding.apply(init_weights)
        self.lnorm = nn.LayerNorm(d_model)
        self.lnorm.apply(init_weights)

        self.transformer = TwoLayerTransformer(self.d_model, n_heads=4)
        self.transformer.apply(init_weights)
        self.transformer.eval()

    def reset(self, seed: int):
        self.steps = 0
        self.env_obs, _ = self.base_env.reset(seed=seed)
        self.thought = torch.zeros(self.d_model, dtype=torch.float32)
        self.cache = {
            "l1": None,
            "l2": None,
        }
        obs = self._get_obs()

        self.thought, self.cache = self.transformer(
            self.pos_embedding(torch.tensor(self.env_obs))[None, None],
            self.cache
        )
        self.thought = self.thought.detach()[0, 0]

        return obs, {}

    def _get_obs(self):
        return {
            "env": np.eye(16)[self.env_obs] if self.tabular else [self.env_obs],
            "thought": self.thought.clone(),
        }

    def step(self, action):
        self.steps += 1

        done = False
        reward = 0.0
        if action < self.n_acts + 1:  # Environment action
            if action > 0:
                self.env_obs, reward, terminated, truncated, _ = self.base_env.step(action - 1)
                done = terminated or truncated
            self.thought = torch.zeros(self.d_model, dtype=torch.float32)
            self.cache = {
                "l1": None,
                "l2": None,
            }
            obs = self._get_obs()

            self.thought, self.cache = self.transformer(
                self.pos_embedding(torch.tensor(self.env_obs))[None, None],
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

        if self.steps >= self.max_steps:
            done = True

        return obs, reward, done, False, {}
