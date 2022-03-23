# Copyright (c) 2020, Fabio Muratore, Honda Research Institute Europe GmbH, and
# Technical University of Darmstadt.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of Fabio Muratore, Honda Research Institute Europe GmbH,
#    or Technical University of Darmstadt, nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL FABIO MURATORE, HONDA RESEARCH INSTITUTE EUROPE GMBH,
# OR TECHNICAL UNIVERSITY OF DARMSTADT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import sys
from copy import deepcopy
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch as to
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.step_based.value_based import ValueBased
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.utils import typed_env
from pyrado.environments.base import Env
from pyrado.exploration.stochastic_action import SACExplStrat
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy, TwoHeadedPolicy
from pyrado.sampling.cvar_sampler import CVaRSampler
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler, ParallelRolloutSamplerTensor
from pyrado.utils.data_processing import standardize
from pyrado.utils.math import soft_update_
from pyrado.algorithms.utils import compute_action_statistics
import ipdb

class SAC(ValueBased):
    """
    Soft Actor-Critic (SAC) variant with stochastic policy and two Q-functions and two Q-targets (no V-function)

    .. seealso::
        [1] T. Haarnoja, A. Zhou, P. Abbeel, S. Levine, "Soft Actor-Critic: Off-Policy Maximum Entropy Deep
            Reinforcement Learning with a Stochastic Actor", ICML, 2018
        [2] This implementation was inspired by https://github.com/pranz24/pytorch-soft-actor-critic
            which is seems to be based on https://github.com/vitchyr/rlkit
        [3] This implementation also borrows (at least a bit) from
            https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/sac
        [4] https://github.com/MushroomRL/mushroom-rl/blob/dev/mushroom_rl/algorithms/actor_critic/deep_actor_critic/sac.py

    .. note::
        The update order of the policy, the Q-functions, (and the entropy coefficient) is different in almost every
        implementation out there. Here we follow the one from [3].
    """

    name: str = "sac"

    def __init__(
        self,
        save_dir: pyrado.PathLike,
        env: Env,
        policy: TwoHeadedPolicy,
        qfcn_1: Policy,
        qfcn_2: Policy,
        memory_size: int,
        gamma: float,
        max_iter: int,
        num_updates_per_step: Optional[int] = None,
        tau: float = 0.995,
        ent_coeff_init: float = 0.2,
        learn_ent_coeff: bool = True,
        target_update_intvl: int = 1,
        num_init_memory_steps: Optional[int] = None,
        standardize_rew: bool = True,
        rew_scale: Union[int, float] = 1.0,
        min_rollouts: Optional[int] = None,
        min_steps: Optional[int] = None,
        batch_size: Optional[int] = 256,
        eval_intvl: int = 100,
        max_grad_norm: float = 5.0,
        lr: float = 3e-4,
        lr_scheduler=None,
        lr_scheduler_hparam: Optional[dict] = None,
        num_workers: int = 4,
        logger: Optional[StepLogger] = None,
        use_trained_policy_for_refill: bool = False,
        env_sim = None,
        num_init_rollouts = 1,
    ):
        r"""
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param policy: policy to be updated
        :param qfcn_1: state-action value function $Q(s,a)$, the associated target Q-functions is created from a
                        re-initialized copies of this one
        :param qfcn_2: state-action value function $Q(s,a)$, the associated target Q-functions is created from a
                        re-initialized copies of this one
        :param memory_size: number of transitions in the replay memory buffer, e.g. 1000000
        :param gamma: temporal discount factor for the state values
        :param max_iter: maximum number of iterations (i.e. policy updates) that this algorithm runs
        :param num_updates_per_step: number of (batched) gradient updates per algorithm step
        :param tau: interpolation factor in averaging for target networks, update used for the soft update a.k.a. polyak
                    update, between 0 and 1
        :param ent_coeff_init: initial weighting factor of the entropy term in the loss function
        :param learn_ent_coeff: adapt the weighting factor of the entropy term
        :param target_update_intvl: number of iterations that pass before updating the target network
        :param num_init_memory_steps: number of samples used to initially fill the replay buffer with, pass `None` to
                                      fill the buffer completely
        :param standardize_rew:  if `True`, the rewards are standardized to be $~ N(0,1)$
        :param rew_scale: scaling factor for the rewards, defaults no scaling
        :param min_rollouts: minimum number of rollouts sampled per policy update batch
        :param min_steps: minimum number of state transitions sampled per policy update batch
        :param batch_size: number of samples per policy update batch
        :param eval_intvl: interval in which the evaluation rollouts are collected, also the interval in which the
                           logger prints the summary statistics
        :param max_grad_norm: maximum L2 norm of the gradients for clipping, set to `None` to disable gradient clipping
        :param lr: (initial) learning rate for the optimizer which can be by modified by the scheduler.
                   By default, the learning rate is constant.
        :param lr_scheduler: learning rate scheduler type for the policy and the Q-functions that does one step
                             per `update()` call
        :param lr_scheduler_hparam: hyper-parameters for the learning rate scheduler
        :param num_workers: number of environments for parallel sampling
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        :param use_trained_policy_for_refill: whether to use the trained policy instead of a dummy policy to refill the
                                              replay buffer after resets
        """
        if typed_env(env, ActNormWrapper) is None:
            raise pyrado.TypeErr(msg="SAC required an environment wrapped by an ActNormWrapper!")
        if not isinstance(qfcn_1, Policy):
            raise pyrado.TypeErr(given=qfcn_1, expected_type=Policy)
        if not isinstance(qfcn_2, Policy):
            raise pyrado.TypeErr(given=qfcn_2, expected_type=Policy)

        # Call ValueBased's constructor
        super().__init__(
            save_dir=save_dir,
            env=env,
            policy=policy,
            memory_size=memory_size,
            gamma=gamma,
            max_iter=max_iter,
            num_updates_per_step=num_updates_per_step,
            target_update_intvl=target_update_intvl,
            num_init_memory_steps=num_init_memory_steps,
            min_rollouts=min_rollouts,
            min_steps=min_steps,
            batch_size=batch_size,
            eval_intvl=eval_intvl,
            max_grad_norm=max_grad_norm,
            num_workers=num_workers,
            logger=logger,
            use_trained_policy_for_refill=use_trained_policy_for_refill,
            env_sim=env_sim,
        )

        self.qfcn_1 = qfcn_1
        self.qfcn_2 = qfcn_2
        self.qfcn_targ_1 = deepcopy(self.qfcn_1).eval()  # will not be trained using an optimizer
        self.qfcn_targ_2 = deepcopy(self.qfcn_2).eval()  # will not be trained using an optimizer
        self.tau = tau
        self.learn_ent_coeff = learn_ent_coeff
        self.standardize_rew = standardize_rew
        self.rew_scale = rew_scale
        self.num_updates = 0
        self.num_update_calls = 0
        self.grad_vi = False
        self.num_init_rollouts = num_init_rollouts
        self.batch_size_used = batch_size
        # self._env_sim = env_sim

        # Create sampler for exploration during training
        self._expl_strat = SACExplStrat(self._policy)
        self._sampler = ParallelRolloutSamplerTensor(
        # self._sampler = ParallelRolloutSampler(
            self._env,
            self._expl_strat,
            num_workers=num_workers if min_steps != 1 else 1,
            min_steps=min_steps,  # in [2] this would be 1
            min_rollouts=min_rollouts,  # in [2] this would be None
        )

        # Q-function optimizers
        self._optim_policy = to.optim.Adam([{"params": self._policy.parameters()}], lr=lr, eps=1e-5)
        self._optim_qfcns = to.optim.Adam(
            [{"params": self.qfcn_1.parameters()}, {"params": self.qfcn_2.parameters()}], lr=lr, eps=1e-5
        )

        # Automatic entropy tuning
        log_ent_coeff_init = to.log(to.tensor(ent_coeff_init, device=policy.device, dtype=to.get_default_dtype()))
        if learn_ent_coeff:
            self._log_ent_coeff = nn.Parameter(log_ent_coeff_init, requires_grad=True)
            self._ent_coeff_optim = to.optim.Adam([{"params": self._log_ent_coeff}], lr=lr, eps=1e-5)
            self.target_entropy = -to.prod(to.tensor(env.act_space.shape))
        else:
            self._log_ent_coeff = log_ent_coeff_init

        # Learning rate scheduler
        self._lr_scheduler_policy = lr_scheduler
        self._lr_scheduler_hparam = lr_scheduler_hparam
        if lr_scheduler is not None:
            self._lr_scheduler_policy = lr_scheduler(self._optim_policy, **lr_scheduler_hparam)
            self._lr_scheduler_qfcns = lr_scheduler(self._optim_qfcns, **lr_scheduler_hparam)
        else:
            self._lr_scheduler_policy = self._lr_scheduler_qfcns = None

    @property
    def sampler(self) -> ParallelRolloutSampler:
        return self._sampler

    @sampler.setter
    def sampler(self, sampler: ParallelRolloutSampler):
        if not isinstance(sampler, (ParallelRolloutSampler, CVaRSampler)):
            raise pyrado.TypeErr(given=sampler, expected_type=(ParallelRolloutSampler, CVaRSampler))
        self._sampler = sampler

    @property
    def ent_coeff(self) -> to.Tensor:
        """Get the detached entropy coefficient."""
        return to.exp(self._log_ent_coeff.detach())

    def update(self):
        """Update the policy's and Q-functions' parameters on transitions sampled from the replay memory."""
        # Containers for logging
        expl_strat_stds = to.zeros(self.num_batch_updates)
        qfcn_1_losses = to.zeros(self.num_batch_updates)
        qfcn_2_losses = to.zeros(self.num_batch_updates)
        qfcn_1_grad_norm = to.zeros(self.num_batch_updates)
        qfcn_2_grad_norm = to.zeros(self.num_batch_updates)
        policy_losses = to.zeros(self.num_batch_updates)
        policy_grad_norm = to.zeros(self.num_batch_updates)
        self.num_update_calls +=1
        for b in tqdm(
            range(self.num_batch_updates),
            total=self.num_batch_updates,
            desc="Updating",
            unit="batches",
            file=sys.stdout,
            leave=False,
        ):
            # Sample steps and the associated next step from the replay memory
            steps, next_steps = self._memory.sample(self.batch_size_used)
            try:
                next_steps.observations = steps.next_obs
            except:
                pass
            steps.torch(data_type=to.get_default_dtype())
            next_steps.torch(data_type=to.get_default_dtype())
            # ipdb.set_trace()

            # Standardize and optionally scale the rewards
            if self.standardize_rew:
                rewards = standardize(steps.rewards)
            else:
                rewards = steps.rewards
            rewards = rewards.to(self.policy.device)
            rewards *= self.rew_scale

            # Explore and compute the current log probs (later used for policy update)
            # ipdb.set_trace()
            if self.policy.is_recurrent:
                # next_act_expl1, _, _ = self._expl_strat(next_steps.observations, next_steps.hidden_states)
                act_expl, log_probs_expl, _ = self._expl_strat(steps.observations, steps.hidden_states)
            else:
                # next_act_expl1, _ = self._expl_strat(next_steps.observations)
                act_expl, log_probs_expl = self._expl_strat(steps.observations)
            # next_steps.observations[:,-1] = act_expl[:,0]
            expl_strat_stds[b] = to.mean(self._expl_strat.std.data)

            # Update the the entropy coefficient
            if self.learn_ent_coeff:
                # Compute entropy coefficient loss
                ent_coeff_loss = -to.mean(self._log_ent_coeff * (log_probs_expl.detach() + self.target_entropy))
                self._ent_coeff_optim.zero_grad()
                ent_coeff_loss.backward()
                self._ent_coeff_optim.step()
            if not self.grad_vi:
                with to.no_grad():
                    # Create masks for the non-final observations
                    not_done = to.from_numpy(1.0 - steps.done).to(device=self.policy.device, dtype=to.get_default_dtype()).view(-1, 1)

                    # Compute the (next)state-(next)action values Q(s',a') from the target networks
                    if self.policy.is_recurrent:
                        next_act_expl, next_log_probs, _ = self._expl_strat(
                            next_steps.observations, next_steps.hidden_states
                        )
                    else:
                        next_act_expl, next_log_probs = self._expl_strat(next_steps.observations)
                    next_obs_act = to.cat([next_steps.observations.to(self.policy.device), next_act_expl], dim=1)
                    next_q_val_target_1 = self.qfcn_targ_1(next_obs_act)
                    next_q_val_target_2 = self.qfcn_targ_2(next_obs_act)
                    next_q_val_target_min = to.min(next_q_val_target_1, next_q_val_target_2)
                    next_q_val_target_min -= self.ent_coeff * next_log_probs  # add entropy term
                    # TD error (including entropy term)
                    next_q_val = rewards.view(-1, 1) + not_done.view(-1, 1) * self.gamma * next_q_val_target_min
            else:
                                    # Create masks for the non-final observations
                not_done = to.from_numpy(1.0 - steps.done).to(device=self.policy.device, dtype=to.get_default_dtype()).view(-1, 1)
                next_obs = next_steps.observations.to(self.policy.device)
                next_obs.requires_grad_(True)
                # Compute the (next)state-(next)action values Q(s',a') from the target networks
                if self.policy.is_recurrent:
                    next_act_expl, next_log_probs, _ = self._expl_strat(
                        next_obs, next_steps.hidden_states
                    )
                else:
                    next_act_expl, next_log_probs = self._expl_strat(next_obs)
                next_obs_act = to.cat([next_obs, next_act_expl.detach().clone()], dim=1)
                next_q_val_target_1 = self.qfcn_targ_1(next_obs_act)
                next_q_val_target_2 = self.qfcn_targ_2(next_obs_act)
                next_q_val_target_min = to.min(next_q_val_target_1, next_q_val_target_2)
                next_q_val_target_min -= self.ent_coeff * next_log_probs#.detach()  # add entropy term
                # TD error (including entropy term)
                next_q_val = (rewards.view(-1, 1) + not_done.view(-1, 1) * self.gamma * next_q_val_target_min).detach().clone()
                
                grad_q = to.autograd.grad(next_q_val_target_min.sum(), next_obs)[0].unsqueeze(1)
                obs_grad = steps.env_infos['obs_grad'].to(self.policy.device)
                rew_grad = steps.env_infos['rew_grad'].to(self.policy.device)
                grad_transition = torch.clamp(obs_grad,-5,5)
                if (torch.abs(obs_grad)>5).any():# or self.num_update_calls==1:
                    ipdb.set_trace()
                grad_transition_bool = 1-torch.logical_or(obs_grad > 5 , obs_grad < -5).sum(dim=1).bool().float()
                next_q_grad = rew_grad + not_done.view(-1,1,1) * \
                                        self.gamma * to.bmm(grad_q, grad_transition)
                next_q_grad_uc = next_q_grad.squeeze(1).detach().clone()
                next_q_grad = next_q_grad_uc#, -5, 5)
                log_probs_old = compute_action_statistics(steps, self._expl_strat).log_probs.to(self.policy.device)
                prob_ratio = torch.exp(log_probs_old - log_probs_expl)
                prob_ratio = torch.exp(prob_ratio*5/prob_ratio.max())
                imp_wts = (prob_ratio/prob_ratio.max()).detach()
            # Compute the (current)state-(current)action values Q(s,a) from the two Q-networks
            # E_{(s_t, a_t) ~ D} [1/2 * (Q_i(s_t, a_t) - r_t - gamma * E_{s_{t+1} ~ p} [V(s_{t+1})] )^2]
            curr_obs_act = to.cat([steps.observations, steps.actions], dim=1).to(self.policy.device).requires_grad_(True)
            state_dim = steps.observations.shape[1]
            q_val_1 = self.qfcn_1(curr_obs_act)
            q_val_2 = self.qfcn_2(curr_obs_act)
            q_1_loss = nn.functional.mse_loss(q_val_1, next_q_val)
            q_2_loss = nn.functional.mse_loss(q_val_2, next_q_val)
            q_loss = (q_1_loss + q_2_loss) / 2.0  # averaging the Q-functions is taken from [3]
            if self.grad_vi:
                qf1_grad = to.cat(to.autograd.grad(q_val_1.sum(), curr_obs_act,retain_graph=True,create_graph=True), dim=-1)
                qf2_grad = to.cat(to.autograd.grad(q_val_2.sum(), curr_obs_act,retain_graph=True,create_graph=True), dim=-1)
                # ipdb.set_trace()
                qf_loss_action = (F.mse_loss(qf1_grad[:, state_dim:], next_q_grad[:, state_dim:]))#, reduction='none')*not_done).mean()#grad_transition_bool[:, state_dim:]).mean()
                qf_loss_action += (F.mse_loss(qf2_grad[:, state_dim:], next_q_grad[:, state_dim:]))#, reduction='none')*not_done).mean()#grad_transition_bool[:, state_dim:]).mean()
                qf_loss_state = (F.mse_loss(qf1_grad[:, :state_dim], next_q_grad[:, :state_dim]))#, reduction='none')*imp_wts).mean()#*not_done).mean()#grad_transition_bool[:, :state_dim]).mean()
                qf_loss_state += (F.mse_loss(qf2_grad[:, :state_dim], next_q_grad[:, :state_dim]))#, reduction='none')*imp_wts).mean()#not_done).mean()#grad_transition_bool[:, :state_dim]).mean()
                if self.num_updates % 500==0:
                    grad_q = to.autograd.grad(q_loss, self.qfcn_1.net.output_layer.weight, retain_graph=True)[0].norm()
                    grad_action = to.autograd.grad(qf_loss_action, self.qfcn_1.net.output_layer.weight, retain_graph=True)[0].norm()
                    grad_state = to.autograd.grad(qf_loss_state, self.qfcn_1.net.output_layer.weight, retain_graph=True)[0].norm()
                    print("grad_state, action, Q: ", grad_state, grad_action, grad_q)

                    if to.isnan(grad_state):
                        ipdb.set_trace()
                    if self.num_updates ==0:
                        self.act_coeff = grad_q/grad_action/(2)
                        self.state_coeff = grad_q/grad_state/(2)
                    else:
                        self.act_coeff = 0.3*self.act_coeff + 0.7*(grad_q/grad_action)/(2*(self.num_update_calls/30+1))
                        self.state_coeff = 0.3*self.act_coeff + 0.7*(grad_q/grad_state)/(2*(self.num_update_calls/30+1))
                    if self.num_updates % 500==0:
                        self.logger.add_value("grad2_state", grad_state)
                        self.logger.add_value("grad2_action", grad_action)
                        self.logger.add_value("grad_q", grad_q)


                # qf_loss += F.mse_loss(qf1_grad, next_q_grad)
                # qf_loss += F.mse_loss(qf2_grad, next_q_grad)
                q_loss += (qf_loss_action)*self.act_coeff #+ qf_loss_state*self.state_coeff 
                # q_loss += (qf_loss_state)*self.state_coeff
            # ipdb.set_trace()

            qfcn_1_losses[b] = q_1_loss.data
            qfcn_2_losses[b] = q_2_loss.data

            # Update the Q-fcns
            self._optim_qfcns.zero_grad()
            q_loss.backward()
            qfcn_1_grad_norm[b] = Algorithm.clip_grad(self.qfcn_1, None)
            qfcn_2_grad_norm[b] = Algorithm.clip_grad(self.qfcn_2, None)
            self._optim_qfcns.step()

            # Compute the policy loss
            # E_{s_t ~ D, eps_t ~ N} [log( pi( f(eps_t; s_t) ) ) - Q(s_t, f(eps_t; s_t))]
            curr_obs_act_expl = to.cat([steps.observations.to(self.policy.device), act_expl], dim=1)
            q_1_val_expl = self.qfcn_1(curr_obs_act_expl)
            q_2_val_expl = self.qfcn_2(curr_obs_act_expl)
            min_q_val_expl = to.min(q_1_val_expl, q_2_val_expl)
            # smooth_loss = to.abs(act_expl - next_act_expl1).mean()
            smooth_loss = (to.abs(act_expl)**2).mean()
            policy_loss = to.mean(self.ent_coeff * log_probs_expl - min_q_val_expl)# + 2*smooth_loss)  # self.ent_coeff is detached
            policy_losses[b] = policy_loss.data

            # Update the policy
            self._optim_policy.zero_grad()
            policy_loss.backward()
            policy_grad_norm[b] = Algorithm.clip_grad(self._expl_strat.policy, self.max_grad_norm)
            self._optim_policy.step()

            # Soft-update the target networks
            if (self._curr_iter * self.num_batch_updates + b) % self.target_update_intvl == 0:
                soft_update_(self.qfcn_targ_1, self.qfcn_1, self.tau)
                soft_update_(self.qfcn_targ_2, self.qfcn_2, self.tau)
            self.num_updates +=1
        # Update the learning rate if the schedulers have been specified
        if self._lr_scheduler_policy is not None:
            self._lr_scheduler_policy.step()
            self._lr_scheduler_qfcns.step()

        # Logging
        self.logger.add_value("Q1 loss", to.mean(qfcn_1_losses))
        self.logger.add_value("Q2 loss", to.mean(qfcn_2_losses))
        self.logger.add_value("policy loss", to.mean(policy_losses))
        self.logger.add_value("avg grad norm policy", to.mean(policy_grad_norm))
        self.logger.add_value("avg expl strat std", to.mean(expl_strat_stds))
        self.logger.add_value("ent_coeff", self.ent_coeff)
        if self.grad_vi:
            self.logger.add_value("grad_state", next_q_grad_uc[:,:-1].norm(dim=-1).median())
            self.logger.add_value("grad_action", next_q_grad_uc[:,-1:].norm(dim=-1).median())
            self.logger.add_value("grad_transition", steps.env_infos['obs_grad'].norm(dim=-1).norm(dim=-1).median())
            self.logger.add_value("grad_transition_max", steps.env_infos['obs_grad'].norm(dim=-1).norm(dim=-1).max())
            self.logger.add_value("grad_transition_high", (steps.env_infos['obs_grad'].norm(dim=-1).norm(dim=-1)>2).sum())
        # self.logger.add_value("grad_state_high", (steps.env_infos['obs_grad'].norm(dim=-1).norm(dim=-1)>2).sum())
        if self._lr_scheduler_policy is not None:
            self.logger.add_value("avg lr policy", np.mean(self._lr_scheduler_policy.get_last_lr()), 6)
            self.logger.add_value("avg lr critic", np.mean(self._lr_scheduler_qfcns.get_last_lr()), 6)

    # def update_real(self):
        # """Update the policy's and Q-functions' parameters on transitions sampled from the replay memory."""
        # # Containers for logging
        # expl_strat_stds = to.zeros(self.num_batch_updates)
        # qfcn_1_losses = to.zeros(self.num_batch_updates)
        # qfcn_2_losses = to.zeros(self.num_batch_updates)
        # qfcn_1_grad_norm = to.zeros(self.num_batch_updates)
        # qfcn_2_grad_norm = to.zeros(self.num_batch_updates)
        # policy_losses = to.zeros(self.num_batch_updates)
        # policy_grad_norm = to.zeros(self.num_batch_updates)
        # self.num_update_calls +=1
        # for b in tqdm(
        #     range(self.num_batch_updates),
        #     total=self.num_batch_updates,
        #     desc="Updating",
        #     unit="batches",
        #     file=sys.stdout,
        #     leave=False,
        # ):
        #     # Sample steps and the associated next step from the replay memory
        #     steps, next_steps = self._memory.sample(self.batch_size)
        #     steps.torch(data_type=to.get_default_dtype())
        #     next_steps.torch(data_type=to.get_default_dtype())
        #     # ipdb.set_trace()

        #     # Standardize and optionally scale the rewards
        #     if self.standardize_rew:
        #         rewards = standardize(steps.rewards)
        #     else:
        #         rewards = steps.rewards
        #     rewards = rewards.to(self.policy.device)
        #     rewards *= self.rew_scale

        #     # Explore and compute the current log probs (later used for policy update)
        #     # ipdb.set_trace()
        #     if self.policy.is_recurrent:
        #         # next_act_expl1, _, _ = self._expl_strat(next_steps.observations, next_steps.hidden_states)
        #         act_expl, log_probs_expl, _ = self._expl_strat(steps.observations, steps.hidden_states)
        #     else:
        #         # next_act_expl1, _ = self._expl_strat(next_steps.observations)
        #         act_expl, log_probs_expl = self._expl_strat(steps.observations)
        #     # next_steps.observations[:,-1] = act_expl[:,0]
        #     expl_strat_stds[b] = to.mean(self._expl_strat.std.data)

        #     # Update the the entropy coefficient
        #     if self.learn_ent_coeff:
        #         # Compute entropy coefficient loss
        #         ent_coeff_loss = -to.mean(self._log_ent_coeff * (log_probs_expl.detach() + self.target_entropy))
        #         self._ent_coeff_optim.zero_grad()
        #         ent_coeff_loss.backward()
        #         self._ent_coeff_optim.step()
        #     if not self.grad_vi:
        #         with to.no_grad():
        #             # Create masks for the non-final observations
        #             not_done = to.from_numpy(1.0 - steps.done).to(device=self.policy.device, dtype=to.get_default_dtype())

        #             # Compute the (next)state-(next)action values Q(s',a') from the target networks
        #             if self.policy.is_recurrent:
        #                 next_act_expl, next_log_probs, _ = self._expl_strat(
        #                     next_steps.observations, next_steps.hidden_states
        #                 )
        #             else:
        #                 next_act_expl, next_log_probs = self._expl_strat(next_steps.observations)
        #             next_obs_act = to.cat([next_steps.observations.to(self.policy.device), next_act_expl], dim=1)
        #             next_q_val_target_1 = self.qfcn_targ_1(next_obs_act)
        #             next_q_val_target_2 = self.qfcn_targ_2(next_obs_act)
        #             next_q_val_target_min = to.min(next_q_val_target_1, next_q_val_target_2)
        #             next_q_val_target_min -= self.ent_coeff * next_log_probs  # add entropy term
        #             # TD error (including entropy term)
        #             next_q_val = rewards.view(-1, 1) + not_done.view(-1, 1) * self.gamma * next_q_val_target_min
        #     else:
        #                             # Create masks for the non-final observations
        #         not_done = to.from_numpy(1.0 - steps.done).to(device=self.policy.device, dtype=to.get_default_dtype())
        #         next_obs = next_steps.observations.to(self.policy.device)
        #         next_obs.requires_grad_(True)
        #         # Compute the (next)state-(next)action values Q(s',a') from the target networks
        #         if self.policy.is_recurrent:
        #             next_act_expl, next_log_probs, _ = self._expl_strat(
        #                 next_obs, next_steps.hidden_states
        #             )
        #         else:
        #             next_act_expl, next_log_probs = self._expl_strat(next_obs)
        #         next_obs_act = to.cat([next_obs, next_act_expl.detach().clone()], dim=1)
        #         next_q_val_target_1 = self.qfcn_targ_1(next_obs_act)
        #         next_q_val_target_2 = self.qfcn_targ_2(next_obs_act)
        #         next_q_val_target_min = to.min(next_q_val_target_1, next_q_val_target_2)
        #         next_q_val_target_min -= self.ent_coeff * next_log_probs#.detach()  # add entropy term
        #         # TD error (including entropy term)
        #         next_q_val = (rewards.view(-1, 1) + not_done.view(-1, 1) * self.gamma * next_q_val_target_min).detach().clone()
        #         # ipdb.set_trace()
        #         grad_q = to.autograd.grad(next_q_val_target_min.sum(), next_obs)[0].unsqueeze(1)
        #         grad_transition = torch.clamp(steps.env_infos['obs_grad'], -5,5)
        #         next_q_grad = steps.env_infos['rew_grad'] + not_done.unsqueeze(1).unsqueeze(1) * \
        #                                 self.gamma * to.bmm(grad_q, grad_transition)
        #         next_q_grad_uc = next_q_grad.squeeze(1).detach().clone()
        #         next_q_grad = next_q_grad_uc#, -5, 5)

        #     # Compute the (current)state-(current)action values Q(s,a) from the two Q-networks
        #     # E_{(s_t, a_t) ~ D} [1/2 * (Q_i(s_t, a_t) - r_t - gamma * E_{s_{t+1} ~ p} [V(s_{t+1})] )^2]
        #     curr_obs_act = to.cat([steps.observations, steps.actions], dim=1).to(self.policy.device).requires_grad_(True)
        #     state_dim = steps.observations.shape[1]
        #     q_val_1 = self.qfcn_1(curr_obs_act)
        #     q_val_2 = self.qfcn_2(curr_obs_act)
        #     q_1_loss = nn.functional.mse_loss(q_val_1, next_q_val)
        #     q_2_loss = nn.functional.mse_loss(q_val_2, next_q_val)
        #     q_loss = (q_1_loss + q_2_loss) / 2.0  # averaging the Q-functions is taken from [3]
        #     if self.grad_vi:
        #         qf1_grad = to.cat(to.autograd.grad(q_val_1.sum(), curr_obs_act,retain_graph=True,create_graph=True), dim=-1)
        #         qf2_grad = to.cat(to.autograd.grad(q_val_2.sum(), curr_obs_act,retain_graph=True,create_graph=True), dim=-1)
        #         qf_loss_action = F.mse_loss(qf1_grad[:, state_dim:], next_q_grad[:, state_dim:])
        #         qf_loss_action += F.mse_loss(qf2_grad[:, state_dim:], next_q_grad[:, state_dim:])
        #         qf_loss_state = F.mse_loss(qf1_grad[:, :state_dim], next_q_grad[:, :state_dim])            
        #         qf_loss_state += F.mse_loss(qf2_grad[:, :state_dim], next_q_grad[:, :state_dim])
        #         if self.num_updates % 500==0:
        #             grad_q = to.autograd.grad(q_loss, self.qfcn_1.net.output_layer.weight, retain_graph=True)[0].norm()
        #             grad_action = to.autograd.grad(qf_loss_action, self.qfcn_1.net.output_layer.weight, retain_graph=True)[0].norm()
        #             grad_state = to.autograd.grad(qf_loss_state, self.qfcn_1.net.output_layer.weight, retain_graph=True)[0].norm()
        #             print("grad_state, action, Q: ", grad_state, grad_action, grad_q)

        #             if to.isnan(grad_state):
        #                 ipdb.set_trace()
        #             if self.num_updates ==0:
        #                 self.act_coeff = 1#grad_q/grad_action/2
        #                 self.state_coeff = 1#grad_q/grad_state/2
        #             else:
        #                 self.act_coeff = 0.3*self.act_coeff + 0.7*(grad_q/grad_action/(2*(self.num_update_calls/30+1)))
        #                 self.state_coeff = 0.3*self.act_coeff + 0.7*(grad_q/grad_state)/(2*(self.num_update_calls/30+1))

        #             self.logger.add_value("grad2_state", grad_state)
        #             self.logger.add_value("grad2_action", grad_action)
        #             self.logger.add_value("grad_q", grad_q)


        #         # qf_loss += F.mse_loss(qf1_grad, next_q_grad)
        #         # qf_loss += F.mse_loss(qf2_grad, next_q_grad)
        #         q_loss += (qf_loss_action)*self.act_coeff + qf_loss_state*self.state_coeff
        #     # ipdb.set_trace()

        #     qfcn_1_losses[b] = q_1_loss.data
        #     qfcn_2_losses[b] = q_2_loss.data

        #     # Update the Q-fcns
        #     self._optim_qfcns.zero_grad()
        #     q_loss.backward()
        #     qfcn_1_grad_norm[b] = Algorithm.clip_grad(self.qfcn_1, None)
        #     qfcn_2_grad_norm[b] = Algorithm.clip_grad(self.qfcn_2, None)
        #     self._optim_qfcns.step()

        #     # Compute the policy loss
        #     # E_{s_t ~ D, eps_t ~ N} [log( pi( f(eps_t; s_t) ) ) - Q(s_t, f(eps_t; s_t))]
        #     curr_obs_act_expl = to.cat([steps.observations.to(self.policy.device), act_expl], dim=1)
        #     q_1_val_expl = self.qfcn_1(curr_obs_act_expl)
        #     q_2_val_expl = self.qfcn_2(curr_obs_act_expl)
        #     min_q_val_expl = to.min(q_1_val_expl, q_2_val_expl)
        #     # smooth_loss = to.abs(act_expl - next_act_expl1).mean()
        #     smooth_loss = (to.abs(act_expl)**2).mean()
        #     policy_loss = to.mean(self.ent_coeff * log_probs_expl - min_q_val_expl + 2*smooth_loss)  # self.ent_coeff is detached
        #     policy_losses[b] = policy_loss.data

        #     # Update the policy
        #     self._optim_policy.zero_grad()
        #     policy_loss.backward()
        #     policy_grad_norm[b] = Algorithm.clip_grad(self._expl_strat.policy, self.max_grad_norm)
        #     self._optim_policy.step()

        #     # Soft-update the target networks
        #     if (self._curr_iter * self.num_batch_updates + b) % self.target_update_intvl == 0:
        #         soft_update_(self.qfcn_targ_1, self.qfcn_1, self.tau)
        #         soft_update_(self.qfcn_targ_2, self.qfcn_2, self.tau)
        #     self.num_updates +=1
        # # Update the learning rate if the schedulers have been specified
        # if self._lr_scheduler_policy is not None:
        #     self._lr_scheduler_policy.step()
        #     self._lr_scheduler_qfcns.step()

        # # Logging
        # self.logger.add_value("Q1 loss", to.mean(qfcn_1_losses))
        # self.logger.add_value("Q2 loss", to.mean(qfcn_2_losses))
        # self.logger.add_value("policy loss", to.mean(policy_losses))
        # self.logger.add_value("avg grad norm policy", to.mean(policy_grad_norm))
        # self.logger.add_value("avg expl strat std", to.mean(expl_strat_stds))
        # self.logger.add_value("ent_coeff", self.ent_coeff)
        # self.logger.add_value("grad_state", next_q_grad_uc[:,:-1].norm(dim=-1).median())
        # self.logger.add_value("grad_action", next_q_grad_uc[:,-1:].norm(dim=-1).median())
        # self.logger.add_value("grad_transition", steps.env_infos['obs_grad'].norm(dim=-1).norm(dim=-1).median())
        # self.logger.add_value("grad_transition_max", steps.env_infos['obs_grad'].norm(dim=-1).norm(dim=-1).max())
        # self.logger.add_value("grad_transition_high", (steps.env_infos['obs_grad'].norm(dim=-1).norm(dim=-1)>2).sum())
        # # self.logger.add_value("grad_state_high", (steps.env_infos['obs_grad'].norm(dim=-1).norm(dim=-1)>2).sum())
        # if self._lr_scheduler_policy is not None:
        #     self.logger.add_value("avg lr policy", np.mean(self._lr_scheduler_policy.get_last_lr()), 6)
        #     self.logger.add_value("avg lr critic", np.mean(self._lr_scheduler_qfcns.get_last_lr()), 6)

    def reset(self, seed: Optional[int] = None):
        # Reset samplers, replay memory, exploration strategy, internal variables and the random seeds
        super().reset(seed)

        # Reset the learning rate schedulers
        if self._lr_scheduler_policy is not None:
            self._lr_scheduler_policy.last_epoch = -1
        if self._lr_scheduler_qfcns is not None:
            self._lr_scheduler_qfcns.last_epoch = -1

    def init_modules(self, warmstart: bool, suffix: str = "", prefix: str = None, **kwargs):
        # Initialize the policy
        super().init_modules(warmstart, suffix, prefix, **kwargs)

        if prefix is None:
            prefix = f"iter_{self._curr_iter - 1}"

        t1pi = kwargs.get("target1_param_init", None)
        t2pi = kwargs.get("target2_param_init", None)

        if warmstart and None not in (t1pi, t2pi):
            self.qfcn_targ_1.init_param(t1pi)
            self.qfcn_targ_2.init_param(t2pi)
        elif warmstart and None in (t1pi, t2pi) and self._curr_iter > 0:
            self.qfcn_targ_1 = pyrado.load(
                "qfcn_target1.pt", self.save_dir, prefix=prefix, suffix=suffix, obj=self.qfcn_targ_1
            )
            self.qfcn_targ_2 = pyrado.load(
                "qfcn_target2.pt", self.save_dir, prefix=prefix, suffix=suffix, obj=self.qfcn_targ_2
            )
        else:
            # Reset the target Q-functions
            self.qfcn_targ_1.init_param()
            self.qfcn_targ_2.init_param()

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            pyrado.save(self.qfcn_targ_1, "qfcn_target1.pt", self.save_dir, use_state_dict=True)
            pyrado.save(self.qfcn_targ_2, "qfcn_target2.pt", self.save_dir, use_state_dict=True)
        else:
            # This algorithm instance is a subroutine of another algorithm
            prefix = meta_info.get("prefix", "")
            suffix = meta_info.get("suffix", "")
            pyrado.save(
                self.qfcn_targ_1, "qfcn_target1.pt", self.save_dir, prefix=prefix, suffix=suffix, use_state_dict=True
            )
            pyrado.save(
                self.qfcn_targ_2, "qfcn_target2.pt", self.save_dir, prefix=prefix, suffix=suffix, use_state_dict=True
            )

    def load_snapshot(self, parsed_args) -> Tuple[Env, Policy, dict]:
        env, policy, extra = super().load_snapshot(parsed_args)

        # Algorithm specific
        ex_dir = getattr(parsed_args, "dir", None) or self._save_dir
        extra["qfcn_target1"] = pyrado.load(parsed_args.prefix+"qfcn_target1.pt", ex_dir, obj=self.qfcn_targ_1, verbose=True)
        extra["qfcn_target2"] = pyrado.load(parsed_args.prefix+"qfcn_target2.pt", ex_dir, obj=self.qfcn_targ_2, verbose=True)
        self.qfcn_1 = deepcopy(self.qfcn_targ_1).train()
        self.qfcn_2 = deepcopy(self.qfcn_targ_2).train()
        self.qfcn_targ_1 = self.qfcn_targ_1.eval()  # will not be trained using an optimizer
        self.qfcn_targ_2 = self.qfcn_targ_2.eval()  # will not be trained using an optimizer

        return env, policy, extra
