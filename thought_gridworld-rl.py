"""RL training code for learning to think."""
import _pickle as pickle
import argparse
import logging
import numpy as np
import random
import timeit
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Categorical
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import (
    rl_collate_fn,
    RLDataset,
    compute_returns_and_advantages
)
from envs import (
    TFAugmentedFrozenLakeEnv
)
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
        "--tabular",
        action="store_true",
        help="Whether or not to use tabular representation",
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help="Maximum environment steps",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )

    parser.add_argument(
        "--use_ppo",
        action="store_true",
        help="Whether or not to use PPO",
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
        help="Path to write evaluation results",
    )

    parser.add_argument(
        "--log_file",
        type=str,
        required=True,
        help="Path to write logging",
    )

    return parser.parse_args()


# TODO: Add action masking for fixed thought steps.
def train_rl(
    env,
    output_file_base,
    log_file_base,
    seed,
    model_path,
    gamma,
    n_thought_acts,
    d_model,
    use_ppo,
    save_path=None,
):
    device = torch.device("cpu")

    output_file = f"{output_file_base}_{seed}"
    log_file = f"{log_file_base}_{seed}"
    logging.basicConfig(filename=f"{log_file}.log", level=logging.INFO)

    if model_path is not None:
        policy = torch.load(model_path, weights_only=False)
    else:
        policy = ThoughtMLP(
            obs_dim=env.obs_dim,
            n_acts=env.n_acts,
            n_thought_acts=n_thought_acts,
            d_model=d_model,
        ).to(device)
        policy.apply(init_weights)

    vf_and_policy_optimizer = optim.Adam(policy.parameters(), lr=1e-3, weight_decay=0.02)
    vf_optimizer = optim.Adam(policy.parameters(), lr=1e-3, weight_decay=0.0)

    num_updates = 3 if use_ppo else 1
    num_episodes = 100
    num_iterations = 200
    vf_burn_in_iters = 1
    rewards = np.zeros(num_iterations)
    frac_thinking_actions = np.zeros(num_iterations)
    rng = np.random.RandomState(seed)

    for itr in range(num_iterations):
        tic = timeit.default_timer()
        episodes = []
        reach_count = 0
        hole_count = 0
        total_reward = 0.0
        action_counts = np.zeros(5 + n_thought_acts)

        # 1. Collect data
        for episode in range(num_episodes):
            obs, _ = env.reset(rng.randint(0, 2 ** 10))
            done = False

            state_seq = [torch.tensor(np.hstack((obs["env"], obs["thought"])), device=device)]
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
                # logging.info('--')
                # logging.info(sseq)
                # logging.info(aseq)
                # logging.info(obs)

                with torch.no_grad():
                    logits, value = policy(sseq[:, -1])
                    probs = F.softmax(logits, dim=-1)[0]
                    # logging.info(sseq[:, -1])
                    # logging.info(probs)
                    # logging.info(probs)
                    dist = Categorical(probs)
                    action = dist.sample()
                    log_probs.append(dist.log_prob(action).item())
                    values.append(value[0].item())
                    action_counts[action.item()] += 1

                next_obs, reward, done, _, _ = env.step(action.item())

                total_reward += reward
                rew_seq.append(reward)
                obs = next_obs
                timestep += 1
                action_seq.append(action)
                state_seq.append(
                    torch.tensor(np.hstack((obs["env"], obs["thought"])), device=device)
                )

                if done:
                    if reward > 0:
                        reach_count += 1
                    elif reward < 0:
                        hole_count += 1

            sseq = torch.stack(state_seq)
            aseq = torch.stack(action_seq)
            log_prob_seq = torch.tensor(log_probs)
            returns, advs = compute_returns_and_advantages(rew_seq, values, gamma=gamma)
            episodes.append((sseq, aseq, returns, advs, log_prob_seq))
            # assert episode < 2
        frac_thinking_actions[itr] = action_counts[5:].sum() / action_counts.sum()
        rewards[itr] = total_reward / num_episodes
        toc = timeit.default_timer()
        logging.info(
            f"Iter {itr}: Rollout = {toc - tic}s, Avg Return = {rewards[itr]}, Frac Thinking = {frac_thinking_actions[itr]}, Reach = {reach_count / num_episodes}, Hole = {hole_count / num_episodes}"
        )

        tic = timeit.default_timer()
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

            for grad_update in range(num_updates):
                for states, acts, targets, returns, advs, old_logprobs, mask in loader:
                    logits, values = policy(states)
                    dist = Categorical(logits=logits)
                    logprobs = dist.log_prob(targets)

                    # Not using baseline
                    advs = returns

                    binary_mask = mask.to(
                        dtype=torch.uint8
                    )  # Or torch.int, torch.long, etc.

                    if use_ppo:
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
                        logging.info(f"Value Loss: {loss.item()}")
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
        toc = timeit.default_timer()
        logging.info(f"Update time: {toc - tic}s")

    return policy


def evaluate_agent(env, agent, seed):
    num_episodes = 100
    total_reward = 0.0
    total_steps = 0
    total_act_steps = 0
    rng = np.random.RandomState(seed)

    for episode in range(num_episodes):
        obs, _ = env.reset(rng.randint(0, 2 ** 10))
        done = False

        state_seq = [torch.tensor(np.hstack(([obs["env"]], obs["thought"])))]
        action_seq = [torch.tensor(0)]
        while not done:
            sseq = torch.stack(state_seq).unsqueeze(0)

            with torch.no_grad():
                logits, _ = agent(sseq[:, -1])
                probs = F.softmax(logits, dim=-1)[0]
                # logging.info(probs)
                dist = Categorical(probs)
                action = dist.sample()

            next_obs, reward, done, _, _ = env.step(action.item())

            if action > 0 and action < 5:
                total_act_steps += 1

            total_reward += reward
            total_steps += 1
            obs = next_obs
            action_seq.append(action)
            state_seq.append(
                torch.tensor(np.hstack(([obs["env"]], obs["thought"])))
            )

    logging.info("EVAL AVG REWARD: ", total_reward / num_episodes)
    logging.info("EVAL AVG EP LEN: ", total_steps / num_episodes)
    logging.info("EVAL AVG NUM ACTS: ", total_act_steps / num_episodes)


if __name__ == "__main__":
    args = parse_args()
    seed = args.seed
    model_path = args.model_path
    results_file = args.output_file
    log_file = args.log_file
    save_path = args.model_save_path
    n_thought_states = args.n_thought_states
    n_thought_acts = args.n_thought_acts
    d_model = args.d_model
    gamma = args.gamma
    use_ppo = args.use_ppo
    tabular = args.tabular
    max_steps = args.max_steps

    pickle.dump(
        args,
        open(f"{results_file}_{seed}-args.pkl", "wb")
    )

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    env = TFAugmentedFrozenLakeEnv(
        n_thought_states=n_thought_states,
        n_thought_acts=n_thought_acts,
        d_model=d_model,
        max_steps=max_steps,
        tabular=tabular,
    )

    agent = train_rl(
        env,
        results_file,
        log_file,
        seed,
        model_path, 
        gamma=gamma,
        n_thought_acts=n_thought_acts,
        d_model=d_model,
        use_ppo=use_ppo,
        save_path=save_path,
    )
    evaluate_agent(env, agent, seed + 1)
