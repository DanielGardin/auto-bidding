from typing import Callable, Sequence, Optional

import torch
import torch.nn as nn

from .base import Actor

import torch
import torch.nn as nn
import torch.nn.functional as F

class DecisionTransformer(nn.Module):

    def __init__(self, 
                 state_dim: int, 
                 act_dim: int, 
                 state_std: float, 
                 action_tanh: bool, 
                 max_ep_len: int,
                 length_times: int,
                 hidden_size: int,
                 decoder_d_model: int, 
                 decoder_nhead: int,
                 decoder_dim_feedforward: int, 
                 decoder_dropout: float, 
                 decoder_activation: Callable[[torch.Tensor], torch.Tensor] | str, 
                 decoder_layer_norm_eps: float, 
                 decoder_batch_first: bool, 
                 decoder_norm_first: bool, 
                 decoder_bias: bool, 
                 transformer_num_layers: int
                 ):
        

        super(DecisionTransformer, self).__init__()

        ## Params for network architecture
        self.length_times = self.length_times
        self.hidden_size = self.hidden_size
        self.state_std = state_std
        self.max_ep_len = max_ep_len


        self.state_dim = state_dim
        self.act_dim = act_dim


        ## Params for TransformerDecoder
        self.transformer_num_layers = transformer_num_layers
        self.decoder_d_model = decoder_d_model
        self.decoder_nhead = decoder_nhead
        self.decoder_dim_feedforward = decoder_dim_feedforward
        self.decoder_dropout = decoder_dropout
        self.decoder_activation = decoder_activation
        self.decoder_layer_norm_eps = decoder_layer_norm_eps
        self.decoder_batch_first = decoder_batch_first
        self.decoder_norm_first = decoder_norm_first
        self.decoder_bias = decoder_bias,



        ## NNs
        self.transformer = nn.TransformerDecoder(nn.TransformerDecoderLayer(self.d_model, 
                                                                            self.nhead,

                                                                            ), 
                                                 num_layers=self.num_transformer_layers,
                                                 )

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




        ## Memory
        


        ## Memory
        self.eval_states = None
        self.eval_actions = torch.zeros((0, self.act_dim), dtype=torch.float32)
        self.eval_rewards = torch.zeros(0, dtype=torch.float32)
        self.eval_target_return = None
        self.eval_timesteps = torch.tensor(0, dtype=torch.long).reshape(1, 1)
        self.eval_episode_return, self.eval_episode_length = 0, 0

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        rewards_embeddings = self.embed_reward(rewards)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        rewards_embeddings = rewards_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = torch.stack(
            ([attention_mask for _ in range(self.length_times)]), dim=1
        ).permute(0, 2, 1).reshape(batch_size, self.length_times * seq_length).to(stacked_inputs.dtype)

        x = stacked_inputs
        for block in self.transformer:
            x = block(x, stacked_attention_mask)

        x = x.reshape(batch_size, seq_length, self.length_times, self.hidden_size).permute(0, 2, 1, 3)

        return_preds = self.predict_return(x[:, 2])
        state_preds = self.predict_state(x[:, 2])
        action_preds = self.predict_action(x[:, 1])
        return state_preds, action_preds, return_preds, None
    
    def get_action(
            self,
            obs: torch.Tensor,
            action: torch.Tensor | None = None
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        self.eval()
        if self.eval_states is None:
            self.eval_states = torch.from_numpy(state).reshape(1, self.state_dim)
            ep_return = target_return if target_return is not None else self.target_return
            self.eval_target_return = torch.tensor(ep_return, dtype=torch.float32).reshape(1, 1)
        else:
            assert pre_reward is not None
            cur_state = torch.from_numpy(state).reshape(1, self.state_dim)
            self.eval_states = torch.cat([self.eval_states, cur_state], dim=0)
            self.eval_rewards[-1] = pre_reward
            pred_return = self.eval_target_return[0, -1] - (pre_reward / self.scale)
            self.eval_target_return = torch.cat([self.eval_target_return, pred_return.reshape(1, 1)], dim=1)
            self.eval_timesteps = torch.cat(
                [self.eval_timesteps, torch.ones((1, 1), dtype=torch.long) * self.eval_timesteps[:, -1] + 1], dim=1)
        self.eval_actions = torch.cat([self.eval_actions, torch.zeros(1, self.act_dim)], dim=0)
        self.eval_rewards = torch.cat([self.eval_rewards, torch.zeros(1)])

        action = self.get_action(
            (self.eval_states.to(dtype=torch.float32) - self.state_mean) / self.state_std,
            self.eval_actions.to(dtype=torch.float32),
            self.eval_rewards.to(dtype=torch.float32),
            self.eval_target_return.to(dtype=torch.float32),
            self.eval_timesteps.to(dtype=torch.long)
        )
        self.eval_actions[-1] = action
        action = action.detach().cpu().numpy()
        return action


        return torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
