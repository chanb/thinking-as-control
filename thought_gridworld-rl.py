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

from tabulate import tabulate
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
        default=30,
        help="Maximum environment steps",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )

    parser.add_argument(
        "--algo",
        type=str,
        choices=["reinforce", "ppo:clip", "ppo:reverse_kl", "ac"],
        help="Algorithm to use",
    )

    parser.add_argument(
        "--num_iterations",
        type=int,
        default=300,
        help="Number of iterations (default: 300)",
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
    seed,
    model_path,
    gamma,
    n_thought_acts,
    d_model,
    algo,
    num_iterations=300,
    save_path=None,
):
    device = torch.device("cpu")

    output_file = f"{output_file_base}_{seed}"

    if model_path is not None:
        policy = torch.load(model_path, weights_only=False)
    else:
        policy = ThoughtMLP(
            obs_dim=env.obs_dim,
            n_acts=env.n_acts,
            n_thought_acts=n_thought_acts,
            d_model=d_model,
        ).to(device)
        # policy.apply(init_weights)

    vf_and_policy_optimizer = optim.Adam(policy.parameters(), lr=1e-2, weight_decay=0.0)
    vf_optimizer = optim.Adam(policy.parameters(), lr=1e-2, weight_decay=0.0)

    # vf_and_policy_optimizer = optim.SGD(policy.parameters(), lr=1e-1, weight_decay=0.0)
    # vf_optimizer = optim.SGD(policy.parameters(), lr=1e-1, weight_decay=0.0)

    beta_coef = 1e-2 if algo == "ppo:reverse_kl" else 0.0
    num_episodes = 50
    vf_burn_in_iters = int(algo != "reinforce")
    lam = 0.95 if algo.startswith("ppo:") else 1.0
    rewards = np.zeros(num_iterations)
    frac_thinking_actions = np.zeros(num_iterations)
    rng = np.random.RandomState(seed)

    # state_map = dict()
    # state_id = 0
    for itr in range(num_iterations):
        if vf_burn_in_iters > 0 and itr == 0:
            num_updates = 1
        else:
            num_updates = 3 if algo.startswith("ppo:") else 1
        tic = timeit.default_timer()
        episodes = []
        reach_count = 0
        hole_count = 0
        ep_rewards = np.zeros(num_episodes)
        ep_lens = np.zeros(num_episodes, dtype=int)
        action_counts = np.zeros(env.n_acts + env.n_thought_acts + 1)
        # state_counts = np.zeros(env.n_states * env.n_thought_states)

        # 1. Collect data
        for ep_i in range(num_episodes):
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

                    # if tuple(sseq[0, -1].tolist()) not in state_map:
                    #     state_map[tuple(sseq[0, -1].tolist())] = state_id
                    #     state_id += 1

                    # # print(len(state_counts), state_id, env.n_thought_states)
                    # state_counts[state_map[tuple(sseq[0, -1].tolist())]] += 1

                next_obs, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated

                ep_rewards[ep_i] += reward
                ep_lens[ep_i] += 1
                rew_seq.append(reward)
                obs = next_obs
                timestep += 1

                action_seq.append(action)
                state_seq.append(torch.tensor(np.hstack((obs["env"], obs["thought"])), device=device))

                if done:
                    if reward >= 1.0:
                        reach_count += 1
                    elif reward <= 0 and timestep < max_steps:
                        hole_count += 1

                    if truncated:
                        with torch.no_grad():
                            sseq = torch.stack(state_seq).unsqueeze(0)
                            _, next_value = policy(sseq[:, -1])
                            next_value = next_value.detach().item()
                    else:
                        next_value = 0

            sseq = torch.stack(state_seq)
            aseq = torch.stack(action_seq)
            log_prob_seq = torch.tensor(log_probs)

            # # XXX: Hindsight cycle removal
            # match_state = torch.all(sseq[:-1] == sseq[1:], dim=-1)
            # match_thought_act = torch.roll(torch.logical_and(aseq[:-1] == aseq[1:], aseq[1:] >= env.n_acts), -1)

            # keep_mask = torch.cat((
            #     torch.ones(1, dtype=bool),
            #     torch.logical_not(torch.logical_and(match_state, match_thought_act))
            # ))


            # # print('====')
            # # print(torch.hstack((sseq[:, :4], aseq[:, None])))
            # # print(keep_mask)

            # sseq = sseq[keep_mask]
            # aseq = aseq[keep_mask]
            # log_prob_seq = log_prob_seq[keep_mask[:-1]]
            # rew_seq = torch.tensor(rew_seq)[keep_mask[:-1]]
            # values = torch.tensor(values)[keep_mask[:-1]]
            # # XXX: Hindsight cycle removal

            returns, advs = compute_returns_and_advantages(rew_seq, values, gamma=gamma, lam=lam, next_value=next_value)
            episodes.append((sseq, aseq, returns, advs, log_prob_seq))
            # assert episode < 2
        frac_thinking_actions[itr] = action_counts[5:].sum() / action_counts.sum()
        rewards[itr] = np.mean(ep_rewards)
        toc = timeit.default_timer()
        rollout_time = toc - tic

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

            adv_mean = 0.0
            value_mean = 0.0
            surr_mean = 0.0
            reverse_kl_mean = 0.0
            ent_mean = 0.0
            mse_mean = 0.0

            for grad_update in range(num_updates):
                for states, acts, targets, returns, advs, old_logprobs, mask in loader:
                    logits, values = policy(states)
                    dist = Categorical(logits=logits)
                    logprobs = dist.log_prob(targets)
                    entropy = dist.entropy()

                    binary_mask = mask.to(
                        dtype=torch.uint8
                    )  # Or torch.int, torch.long, etc.
                    num_non_masked_elements = binary_mask.sum()

                    if algo == "ppo:clip":
                        ratio = torch.exp(logprobs - old_logprobs)
                        surr1 = ratio * advs
                        surr2 = torch.clamp(ratio, 0.8, 1.2) * advs
                        surr = torch.min(surr1, surr2) * binary_mask
                    elif algo == "ppo:reverse_kl":
                        ratio = torch.exp(logprobs - old_logprobs)
                        ratio = torch.where(ratio.isinf(), 0.0, ratio)
                        surr = ratio * advs * binary_mask
                    elif algo == "reinforce":
                        surr = returns * logprobs * binary_mask
                    elif algo == "ac":
                        surr = advs * logprobs * binary_mask

                    log_ratios = (old_logprobs - logprobs) * binary_mask
                    reverse_kl = torch.exp(log_ratios) - 1 - log_ratios
                    # print(reverse_kl.max())
                    reverse_kl = torch.where(reverse_kl.isinf(), 0, reverse_kl) * binary_mask
                    # act_mask = (targets < env.n_acts).float()
                    # weights = act_mask + (1 - act_mask) * env.n_thought_acts / (env.n_thought_acts + env.n_acts)
                    # reverse_kl = reverse_kl * weights
                    reverse_kl = reverse_kl.sum() / num_non_masked_elements
                    # print(reverse_kl)

                    
                    surr = surr.sum() / num_non_masked_elements
                    # print(reverse_kl, surr)
                    ent = (entropy * binary_mask).sum() / num_non_masked_elements
                    masked_squared_error = (returns - values) ** 2 * binary_mask
                    sum_masked_squared_error = torch.sum(masked_squared_error)

                    if algo == "reinforce" or num_non_masked_elements == 0:
                        mse_loss = torch.tensor(0.0)
                    else:
                        mse_loss = sum_masked_squared_error / num_non_masked_elements

                    if vf_burn_in_iters > 0 and itr == 0:
                        loss = mse_loss
                        optimizer = vf_optimizer
                        # print("VF", loss)
                    else:
                        loss = -surr + beta_coef * reverse_kl + mse_loss
                        optimizer = vf_and_policy_optimizer
                        # print("VF & PI", loss)

                    surr_mean += surr.item() / num_updates
                    reverse_kl_mean += reverse_kl.item() / num_updates
                    ent_mean += ent.item() / num_updates
                    mse_mean += mse_loss.item() / num_updates
                    value_mean += ((values * binary_mask).sum() / num_non_masked_elements).item() / num_updates
                    adv_mean += ((advs * binary_mask).sum() / num_non_masked_elements).item() / num_updates
                    optimizer.zero_grad()
                    # import ipdb
                    # ipdb.set_trace()
                    loss.backward()
                    optimizer.step()

        np.save(f"{output_file}.npy", rewards)
        np.save(f"{output_file}-thinkactions.npy", frac_thinking_actions)
        if save_path is not None:
            torch.save(policy, save_path)
        toc = timeit.default_timer()
        update_time = toc - tic

        headers = [
            "Itr",
            "Roll(s)",
            "Ret μ",
            # "Ret min",
            # "Ret max",
            "Reach",
            "Hole",
            "Len μ",
            "Len min",
            "Len max",
            "Think",
            "Update(s)",
            "Value",
            "Adv",
            "Ent",
            "Pi Loss",
            "Reverse KL",
            "V Loss",
        ]

        table = [[
            itr,
            f"{rollout_time:.2f}",
            f"{ep_rewards.mean():.3f}",
            # f"{ep_rewards.min():.3f}",
            # f"{ep_rewards.max():.3f}",
            f"{reach_count / num_episodes:.4f}",
            f"{hole_count / num_episodes:.4f}",
            f"{ep_lens.mean():.1f}",
            f"{ep_lens.min()}",
            f"{ep_lens.max()}",
            f"{frac_thinking_actions[itr]:.4f}",
            f"{update_time:.2f}",
            f"{value_mean:.4f}",
            f"{adv_mean:.4f}",
            f"{ent_mean:.4f}",
            f"{surr_mean:.4f}",
            f"{reverse_kl_mean:.4f}",
            f"{mse_mean:.4f}",
        ]]

        logging.info(
            "\n" + tabulate(table, headers=headers, tablefmt="simple")
            + "\n" + tabulate(
                [[f"{p:.4f}" for i, p in enumerate(action_counts / action_counts.sum())]],
                headers=[f"Act: {act_i}" for act_i in range(len(action_counts))],
                tablefmt="grid",
            )
        )

    #     logging.info(
    #         "\n" + tabulate(table, headers=headers, tablefmt="simple")
    #         + "\n" + tabulate(
    #             [[f"{p:.4f}" for i, p in enumerate(state_counts / state_counts.sum())]],
    #             headers=[f"State: {state_i}" for state_i in range(len(state_counts))],
    #             tablefmt="grid",
    #         )
    #     )

    # logging.info(state_map)

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

        state_seq = [torch.tensor(np.hstack((obs["env"], obs["thought"])))]
        action_seq = [torch.tensor(0)]
        while not done:
            sseq = torch.stack(state_seq).unsqueeze(0)

            with torch.no_grad():
                logits, _ = agent(sseq[:, -1])
                # probs = F.softmax(logits, dim=-1)[0]
                # logging.info(probs)
                # dist = Categorical(probs)
                # action = dist.sample()
                action = torch.argmax(logits, dim=-1)

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            if action > 0 and action < 5:
                total_act_steps += 1

            total_reward += reward
            total_steps += 1
            obs = next_obs
            action_seq.append(action)
            state_seq.append(
                torch.tensor(np.hstack((obs["env"], obs["thought"])))
            )

    logging.info(f"EVAL AVG REWARD: {total_reward / num_episodes}")
    logging.info(f"EVAL AVG EP LEN: {total_steps / num_episodes}")
    logging.info(f"EVAL AVG NUM ACTS: {total_act_steps / num_episodes}")


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
    algo = args.algo
    tabular = args.tabular
    max_steps = args.max_steps
    num_iterations = args.num_iterations

    pickle.dump(
        args,
        open(f"{results_file}_{seed}-args.pkl", "wb")
    )

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    log_file = f"{log_file}_{seed}"
    logging.basicConfig(filename=f"{log_file}.log", level=logging.INFO)

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
        seed,
        model_path, 
        gamma=gamma,
        n_thought_acts=n_thought_acts,
        d_model=d_model,
        algo=algo,
        num_iterations=num_iterations,
        save_path=save_path,
    )
    evaluate_agent(env, agent, seed + 1)
