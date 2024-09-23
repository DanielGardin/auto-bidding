from typing import Callable, Sequence, Optional

import torch
import torch.nn as nn

from .base import Actor

import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(Actor):
    def __init__(
            self, 
            state_dim             : int,
            act_dim               : int,
            max_ep_len            : int,
            K                     : Optional[int],
            hidden_size           : int,
            d_model               : int,
            transformer_num_layers: int,
            nhead                 : int,
            target_return         : float = 1.,
            dim_feedforward       : int = 1024,
            activation            : Callable[[torch.Tensor], torch.Tensor] | str = 'relu',
            dropout               : float = 0.1,
            norm_first            : bool = False,
            layer_norm_eps        : float = 1.e-5,
            bias                  : bool = True,
            action_tanh           : bool = False
        ):
        

        super(Transformer, self).__init__()

        ## Params for network architecture
        self.hidden_size = hidden_size
        self.max_ep_len = max_ep_len

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = K

        self.target_return = target_return


        ## NNs
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model         = d_model,
                nhead           = nhead,
                dim_feedforward = dim_feedforward,
                dropout         = dropout,
                activation      = activation,
                layer_norm_eps  = layer_norm_eps,
                batch_first     = True,
                norm_first      = norm_first,
                bias            = bias,
            ),
            num_layers = transformer_num_layers
        )

        self.register_buffer("attn_mask", torch.zeros(3*max_ep_len, 3*max_ep_len, dtype=torch.bool))

        self.attn_mask = torch.triu(torch.ones(3*max_ep_len, 3*max_ep_len, dtype=torch.bool), diagonal=1)

        self.register_buffer("eval_states", torch.empty((0, state_dim), dtype=torch.float32))
        self.register_buffer("eval_actions", torch.empty((0, act_dim), dtype=torch.float32))
        self.register_buffer("eval_rtg", torch.empty((0, 1), dtype=torch.float32))
        self.register_buffer("eval_timesteps", torch.zeros((1), dtype=torch.long))

        self.embed_timestep = nn.Embedding(self.max_ep_len, self.hidden_size)
        self.embed_return = torch.nn.Linear(1, self.hidden_size)
        self.embed_reward = torch.nn.Linear(1, self.hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, self.hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, self.hidden_size)

        self.embed_ln = nn.LayerNorm(self.hidden_size)
        self.predict_state = torch.nn.Linear(self.hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )

        self.predict_return = torch.nn.Linear(self.hidden_size, 1)


    def forward(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            returns_to_go: torch.Tensor,
            timesteps: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if mask is None:
            mask = torch.zeros((batch_size, seq_length), dtype=torch.bool, device=states.device)

        state_embeddings   = self.embed_state(states)
        action_embeddings  = self.embed_action(actions.view(batch_size, seq_length, -1))
        returns_embeddings = self.embed_return(returns_to_go.view(batch_size, seq_length, -1))
        time_embeddings    = self.embed_timestep(timesteps)

        state_embeddings   = state_embeddings   + time_embeddings
        action_embeddings  = action_embeddings  + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)

        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_mask = torch.repeat_interleave(mask, 3, dim=1)

        x = self.transformer(
            stacked_inputs,
            self.attn_mask[:(3*seq_length), :(3*seq_length)],
            src_key_padding_mask=stacked_mask,
            is_causal=True
        )

        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        return_preds = self.predict_return(x[:, 2])
        state_preds = self.predict_state(x[:, 2])
        action_preds = self.predict_action(x[:, 1])

        return state_preds, action_preds, return_preds


    def reset(self):
        self.eval_states    = torch.empty((0, self.state_dim), dtype=torch.float32, device=self.eval_states.device)
        self.eval_actions   = torch.empty((0, self.act_dim), dtype=torch.float32, device=self.eval_states.device)
        self.eval_rtg       = torch.empty((0, 1), dtype=torch.float32, device=self.eval_states.device)
        self.eval_timesteps = torch.zeros(1, dtype=torch.long, device=self.eval_timesteps.device)


    def callback(self, reward: float):
        if self.eval_rtg.size(0) == 0:
            rtg = self.target_return - reward

        else:
            rtg = self.eval_rtg[-1].item() - reward


        tensor_rtg = torch.zeros((1, 1), dtype=torch.float32, device=self.eval_rtg.device)
        tensor_rtg[0] = rtg

        self.eval_rtg = torch.cat([self.eval_rtg, tensor_rtg])


    def get_action(
            self,
            obs: torch.Tensor,
            action: torch.Tensor | None = None,
            deterministic: bool = False
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if obs.dim() == 3:
            raise ValueError("This actor only supports single inference")

        self.eval_states = torch.cat([self.eval_states, obs])

        self.eval_actions = torch.cat([
            self.eval_actions,
            torch.zeros((1, self.act_dim), dtype=torch.float32, device=self.eval_states.device)
        ])


        if self.max_length and self.eval_states.size(0) > self.max_length:
            self.eval_states    = self.eval_states[-self.max_length:]
            self.eval_actions   = self.eval_actions[-self.max_length:]
            self.eval_rtg       = self.eval_rtg[-self.max_length:]
            self.eval_timesteps = self.eval_timesteps[-self.max_length:]

        _, action_preds, _ = self(
            self.eval_states.unsqueeze(0),
            self.eval_actions.unsqueeze(0),
            self.eval_rtg.unsqueeze(0),
            self.eval_timesteps.unsqueeze(0)
        )


        last_action = action_preds.view(-1, self.act_dim)[-1]
        self.eval_actions[-1] = last_action

        self.eval_timesteps = torch.cat([self.eval_timesteps, torch.tensor([len(self.eval_timesteps)], device=self.eval_timesteps.device)])

        return last_action, torch.tensor(0.), torch.tensor(0.)
