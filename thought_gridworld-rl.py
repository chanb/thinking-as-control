"""RL training code for learning to think."""
import argparse
import copy
import numpy as np
import gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque
from matplotlib import pyplot as plt
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data_utils import rl_collate_fn, RLDataset, compute_returns_and_advantages
from envs import TFAugmentedGridWorldEnv, TFAugmentedGridWorldEnv2
from policies import ThoughtMLP, init_weights


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a pretrained PyTorch model with specified options."
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the pretrained PyTorch model file (.pt or .pth)",
    )

    parser.add_argument(
        "--model_save_path",
        type=str,
        default=None,
        help="Path to save the trained model file (.pt or .pth)",
    )

    parser.add_argument(
        "--n_thought_states",
        type=int,
        default=10,
        help='Number of thought states'
    )

    parser.add_argument(
        "--n_thought_acts",
        type=int,
        default=3,
        help='Number of thought actions'
    )

    parser.add_argument(
        "--d_model",
        type=int,
        default=8,
        help='Dimensionality of the thought state space'
    )

    parser.add_argument(
        "--env",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help='The environment to use'
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to write evaluation results (e.g., results.json or results.csv)",
    )

    return parser.parse_args()


# TODO: Add action masking for fixed thought steps.
def train_rl(
    env,
    output_file_base,
    seed,
    model_path,
    n_thought_acts=3,
    d_model=128,
    save_path=None,
):
    device = torch.device("cpu")

    output_file = f"{output_file_base}_{seed}"

    if model_path is not None:
        policy = torch.load(model_path, weights_only=False)
    else:
        policy = ThoughtMLP(
            n_thought_acts=n_thought_acts,
            d_model=d_model,
        ).to(device)
        policy.apply(init_weights)

    vf_and_policy_optimizer = optim.Adam(policy.parameters(), lr=1e-3, weight_decay=0.02)
    vf_optimizer = optim.Adam(policy.parameters(), lr=1e-3, weight_decay=0.0)

    USE_PPO = False
    num_episodes = 100
    num_iterations = 200
    vf_burn_in_iters = 1
    rewards = np.zeros(num_iterations)
    frac_thinking_actions = np.zeros(num_iterations)

    for itr in tqdm(range(num_iterations)):

        episodes = []
        total_reward = 0.0
        action_counts = np.zeros(5 + n_thought_acts)

        # 1. Collect data
        for episode in tqdm(range(num_episodes)):
            obs = env.reset()
            done = False

            state_seq = [torch.tensor(np.hstack((obs["position"], [obs["letter"]], obs["thought"])), device=device)]
            action_seq = [torch.tensor(0, device=device)]
            rew_seq = []
            log_probs = []
            values = []
            timestep = 0

            while not done:
                sseq = torch.stack(state_seq).unsqueeze(0)
                aseq = torch.stack(action_seq).unsqueeze(0)

                # import ipdb
                # ipdb.set_trace()
                # print('--')
                # print(sseq)
                # print(aseq)

                with torch.no_grad():
                    logits, value = policy(sseq[:, -1])
                    probs = F.softmax(logits, dim=-1)[0]
                    # print(probs)
                    dist = Categorical(probs)
                    action = dist.sample()
                    log_probs.append(dist.log_prob(action).item())
                    values.append(value[0].item())
                    action_counts[action.item()] += 1

                next_obs, reward, done, _ = env.step(action)

                total_reward += reward
                rew_seq.append(reward)
                obs = next_obs
                timestep += 1
                action_seq.append(action)
                state_seq.append(
                    torch.tensor(np.hstack((obs["position"], [obs["letter"]], obs["thought"])), device=device)
                )

            sseq = torch.stack(state_seq)
            aseq = torch.stack(action_seq)
            log_prob_seq = torch.tensor(log_probs)
            returns, advs = compute_returns_and_advantages(rew_seq, values)
            episodes.append((sseq, aseq, returns, advs, log_prob_seq))
            # assert episode < 2
        frac_thinking_actions[itr] = action_counts[5:].sum() / action_counts.sum()
        rewards[itr] = total_reward / num_episodes
        print(
            f"Iter {itr}: Avg Return = {rewards[itr]}, Frac Thinking = {frac_thinking_actions[itr]}"
        )

        dataset = RLDataset(episodes)

        # Policy Optimization
        loader = DataLoader(
            dataset, batch_size=num_episodes, shuffle=True, collate_fn=rl_collate_fn
        )

        if vf_burn_in_iters > 0 and itr == 0:
            num_epochs = vf_burn_in_iters
        else:
            num_epochs = 1

        for epoch in range(num_epochs):

            total_mse = 0.0
            adv_mean = 0.0
            value_mean = 0.0

            for states, acts, targets, returns, advs, old_logprobs, mask in loader:
                logits, values = policy(states)
                dist = Categorical(logits=logits)
                logprobs = dist.log_prob(targets)

                # Not using baseline
                advs = returns

                binary_mask = mask.to(
                    dtype=torch.uint8
                )  # Or torch.int, torch.long, etc.

                if USE_PPO:
                    ratio = torch.exp(logprobs - old_logprobs)
                    surr1 = ratio * advs
                    surr2 = torch.clamp(ratio, 0.8, 1.2) * advs
                    surr = torch.min(surr1, surr2) * binary_mask
                else:
                    surr = advs * logprobs * binary_mask
                masked_squared_error = (returns - values) ** 2 * binary_mask
                sum_masked_squared_error = torch.sum(masked_squared_error)
                num_non_masked_elements = binary_mask.sum()

                if num_non_masked_elements == 0:
                    mse_loss = torch.tensor(0.0)
                else:
                    mse_loss = sum_masked_squared_error / num_non_masked_elements

                if vf_burn_in_iters > 0 and itr == 0:
                    loss = mse_loss
                    optimizer = vf_optimizer
                    print("Value Loss", loss.item())
                else:
                    loss = -(surr.sum() / num_non_masked_elements) + mse_loss
                    optimizer = vf_and_policy_optimizer

                total_mse += mse_loss.item()
                value_mean += values.mean().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        np.save(f"{output_file}.npy", rewards)
        np.save(f"{output_file}-thinkactions.npy", frac_thinking_actions)
        if save_path is not None:
            torch.save(policy, save_path)

    return policy


def evaluate_agent(env, agent, n_thought_acts, d_model, seed):
    num_episodes = 100
    total_reward = 0.0
    total_steps = 0
    total_act_steps = 0

    for episode in range(num_episodes):
        obs = env.reset()
        done = False

        state_seq = [torch.tensor(np.hstack((obs["position"], [obs["letter"]], obs["thought"])))]
        action_seq = [torch.tensor(0)]
        while not done:
            sseq = torch.stack(state_seq).unsqueeze(0)

            with torch.no_grad():
                logits, _ = agent(sseq[:, -1])
                probs = F.softmax(logits, dim=-1)[0]
                # print(probs)
                dist = Categorical(probs)
                action = dist.sample()

            next_obs, reward, done, _ = env.step(action)

            if action > 0 and action < 5:
                total_act_steps += 1

            total_reward += reward
            total_steps += 1
            obs = next_obs
            action_seq.append(action)
            state_seq.append(
                torch.tensor(np.hstack((obs["position"], [obs["letter"]], obs["thought"])))
            )

    print("EVAL AVG REWARD: ", total_reward / num_episodes)
    print("EVAL AVG EP LEN: ", total_steps / num_episodes)
    print("EVAL AVG NUM ACTS: ", total_act_steps / num_episodes)


if __name__ == "__main__":

    args = parse_args()
    seed = args.seed
    model_path = args.model_path
    results_file = args.output_file
    save_path = args.model_save_path
    n_thought_states = args.n_thought_states
    n_thought_acts = args.n_thought_acts
    d_model = args.d_model
    env = args.env

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if env == "v1":
        env = TFAugmentedGridWorldEnv(
            n_thought_acts=n_thought_acts,
            n_goals=2,
            deterministic_start=True,
            d_model=d_model,
            seed=seed,
        )
    elif env == "v2":
        env = TFAugmentedGridWorldEnv2(
            n_thought_states=n_thought_states,
            n_thought_acts=n_thought_acts,
            n_goals=2,
            deterministic_start=True,
            d_model=d_model,
            seed=seed,
        )
    eval_env = copy.deepcopy(env)


    agent = train_rl(
        env,
        results_file,
        seed,
        model_path, 
        n_thought_acts=n_thought_acts,
        d_model=d_model,
        save_path=save_path,
    )
    evaluate_agent(eval_env, agent, n_thought_acts, d_model, seed + 1)
