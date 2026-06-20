import torch
import torch.nn as nn


def init_weights(module):
    # Linear layers
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# def init_weights(module):
#     # Linear layers
#     if isinstance(module, nn.Linear):
#         nn.init.xavier_uniform_(module.weight)
#         if module.bias is not None:
#             nn.init.zeros_(module.bias)

#     # Multihead attention (Q, K, V projections + output proj)
#     elif isinstance(module, nn.MultiheadAttention):
#         nn.init.xavier_uniform_(module.in_proj_weight)
#         if module.in_proj_bias is not None:
#             nn.init.zeros_(module.in_proj_bias)

#         nn.init.xavier_uniform_(module.out_proj.weight)
#         if module.out_proj.bias is not None:
#             nn.init.zeros_(module.out_proj.bias)



class ThoughtMLP(nn.Module):
    def __init__(self, obs_dim, n_acts, n_thought_acts, d_model=128):
        super().__init__()
        self.d_model = d_model
        vocab_size = n_acts + 1 + n_thought_acts  # 1 padding action, environment and thought actions
        self.policy_head = nn.Linear(obs_dim + d_model, vocab_size, bias=False)
        self.value_head = nn.Linear(obs_dim + d_model, 1)
        self.temperature = torch.tensor(1.0)
        self.action_mask = ~torch.tensor([0] + [1] * (n_acts + n_thought_acts)).bool()

    def forward(
        self, state_seq, *args, **kwargs
    ):
        state_embed = state_seq.float()

        logits = self.policy_head(state_embed)
        logits = logits * self.temperature
        logits = logits.masked_fill(self.action_mask, -1e10)

        value = self.value_head(state_embed).squeeze(-1)
        return logits, value


class ThoughtNN(nn.Module):
    def __init__(self, obs_dim, n_acts, n_thought_acts, d_model=128):
        super().__init__()
        self.d_model = d_model
        vocab_size = n_acts + 1 + n_thought_acts  # 1 padding action, environment and thought actions
        self.policy_trunk = nn.Linear(obs_dim + d_model, 128)
        self.policy_head = nn.Linear(128, vocab_size, bias=False)
        self.value_head = nn.Linear(obs_dim + d_model, 1)
        self.temperature = torch.tensor(1.0)
        self.action_mask = ~torch.tensor([0] + [1] * (n_acts + n_thought_acts)).bool()

    def forward(
        self, state_seq, *args, **kwargs
    ):
        state_embed = state_seq.float()

        logits = self.policy_head(nn.functional.relu(self.policy_trunk(state_embed)))
        logits = logits * self.temperature
        logits = logits.masked_fill(self.action_mask, -1e10)

        value = self.value_head(state_embed).squeeze(-1)
        return logits, value
