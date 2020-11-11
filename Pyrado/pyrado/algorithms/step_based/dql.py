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

import numpy as np
import os.path as osp
import sys
import torch as to
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.utils import ReplayMemory
from pyrado.policies.special.dummy import RecurrentDummyPolicy, DummyPolicy
from pyrado.utils.saving_loading import save_prefix_suffix, load_prefix_suffix
from pyrado.environments.base import Env
from pyrado.exploration.stochastic_action import EpsGreedyExplStrat
from pyrado.logger.step import StepLogger, ConsolePrinter, CSVPrinter, TensorBoardPrinter
from pyrado.policies.feed_forward.fnn import DiscreteActQValPolicy
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler
from pyrado.utils.input_output import print_cbt, print_cbt_once


class DQL(Algorithm):
    """
    Deep Q-Learning (without bells and whistles)

    .. seealso::
        [1] V. Mnih et.al., "Human-level control through deep reinforcement learning", Nature, 2015
    """

    name: str = 'dql'

    def __init__(self,
                 save_dir: str,
                 env: Env,
                 policy: DiscreteActQValPolicy,
                 memory_size: int,
                 eps_init: float,
                 eps_schedule_gamma: float,
                 gamma: float,
                 max_iter: int,
                 num_batch_updates: int,
                 target_update_intvl: int = 5,
                 num_init_memory_steps: int = None,
                 min_rollouts: int = None,
                 min_steps: int = None,
                 batch_size: int = 256,
                 num_workers: int = 4,
                 max_grad_norm: float = 0.5,
                 lr: float = 5e-4,
                 lr_scheduler=None,
                 lr_scheduler_hparam: [dict, None] = None,
                 logger: StepLogger = None):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param policy: (current) Q-network updated by this algorithm
        :param memory_size: number of transitions in the replay memory buffer
        :param eps_init: initial value for the probability of taking a random action, constant if `eps_schedule_gamma=1`
        :param eps_schedule_gamma: temporal discount factor for the exponential decay of epsilon
        :param gamma: temporal discount factor for the state values
        :param max_iter: number of iterations (policy updates)
        :param num_batch_updates: number of batch updates per algorithm steps
        :param target_update_intvl: number of iterations that pass before updating the q_targ network
        :param num_init_memory_steps: number of samples used to initially fill the replay buffer with, pass `None` to
                                      fill the buffer completely
        :param min_rollouts: minimum number of rollouts sampled per policy update batch
        :param min_steps: minimum number of state transitions sampled per policy update batch
        :param batch_size: number of samples per policy update batch
        :param num_workers: number of environments for parallel sampling
        :param max_grad_norm: maximum L2 norm of the gradients for clipping, set to `None` to disable gradient clipping
        :param lr: (initial) learning rate for the optimizer which can be by modified by the scheduler.
                   By default, the learning rate is constant.
        :param lr_scheduler: learning rate scheduler that does one step per epoch (pass through the whole data set)
        :param lr_scheduler_hparam: hyper-parameters for the learning rate scheduler
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        if not isinstance(env, Env):
            raise pyrado.TypeErr(given=env, expected_type=Env)
        if not isinstance(policy, DiscreteActQValPolicy):
            raise pyrado.TypeErr(given=policy, expected_type=DiscreteActQValPolicy)

        if logger is None:
            # Create logger that only logs every 100 steps of the algorithm
            logger = StepLogger(print_intvl=100)
            logger.printers.append(ConsolePrinter())
            logger.printers.append(CSVPrinter(osp.join(save_dir, 'progress.csv')))
            logger.printers.append(TensorBoardPrinter(osp.join(save_dir, 'tb')))

        # Call Algorithm's constructor
        super().__init__(save_dir, max_iter, policy, logger)

        self._env = env
        self.q_targ = deepcopy(self._policy)
        self.q_targ.eval()  # will not be trained using the optimizer
        self._memory = ReplayMemory(memory_size)
        self.eps = eps_init
        self.gamma = gamma
        self.target_update_intvl = target_update_intvl
        self.num_batch_updates = num_batch_updates
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        if num_init_memory_steps is None:
            self.num_init_memory_steps = memory_size
        else:
            self.num_init_memory_steps = min(num_init_memory_steps, memory_size)

        # Create sampler for initial filling of the replay memory, exploration during training, and evaluation
        if policy.is_recurrent:
            init_expl_policy = RecurrentDummyPolicy(env.spec, policy.hidden_size)
        else:
            init_expl_policy = DummyPolicy(env.spec)
        self.sampler_init = ParallelRolloutSampler(
            self._env, init_expl_policy,
            num_workers=num_workers,
            min_steps=self.num_init_memory_steps,
        )
        self._expl_strat = EpsGreedyExplStrat(self._policy, eps_init, eps_schedule_gamma)
        self.sampler_trn = ParallelRolloutSampler(
            self._env, self._expl_strat,
            num_workers=num_workers if min_steps != 1 else 1,
            min_steps=min_steps,
            min_rollouts=min_rollouts
        )
        self.sampler_eval = ParallelRolloutSampler(
            self._env, self._policy,
            num_workers=num_workers,
            min_steps=10*env.max_steps,
            min_rollouts=None
        )

        self.optim = to.optim.RMSprop([{'params': self._policy.parameters()}], lr=lr)
        self._lr_scheduler = lr_scheduler
        self._lr_scheduler_hparam = lr_scheduler_hparam
        if lr_scheduler is not None:
            self._lr_scheduler = lr_scheduler(self.optim, **lr_scheduler_hparam)

    @property
    def expl_strat(self) -> EpsGreedyExplStrat:
        return self._expl_strat

    @property
    def memory(self) -> ReplayMemory:
        """ Get the replay memory. """
        return self._memory

    def step(self, snapshot_mode: str, meta_info: dict = None):
        if self._memory.isempty:
            # Warm-up phase
            print_cbt_once('Collecting samples until replay memory if full.', 'w')
            # Sample steps and store them in the replay memory
            ros = self.sampler_init.sample()
            self._memory.push(ros)
        else:
            # Sample steps and store them in the replay memory
            ros = self.sampler_trn.sample()
            self._memory.push(ros)
        self._cnt_samples += sum([ro.length for ro in ros])  # don't count the evaluation samples

        # Log return-based metrics
        if self._curr_iter%self.logger.print_intvl == 0:
            ros = self.sampler_eval.sample()
            rets = [ro.undiscounted_return() for ro in ros]
            ret_max = np.max(rets)
            ret_med = np.median(rets)
            ret_avg = np.mean(rets)
            ret_min = np.min(rets)
            ret_std = np.std(rets)
        else:
            ret_max, ret_med, ret_avg, ret_min, ret_std = 5*[-pyrado.inf]  # dummy values
        self.logger.add_value('max return', np.round(ret_max, 4))
        self.logger.add_value('median return', np.round(ret_med, 4))
        self.logger.add_value('avg return', np.round(ret_avg, 4))
        self.logger.add_value('min return', np.round(ret_min, 4))
        self.logger.add_value('std return', np.round(ret_std, 4))
        self.logger.add_value('avg memory reward', np.round(self._memory.avg_reward(), 4))
        self.logger.add_value('avg rollout length', np.round(np.mean([ro.length for ro in ros]), 2))
        self.logger.add_value('num total samples', self._cnt_samples)

        # Use data in the memory to update the policy and the q_targ Q-function
        self.update()

        # Save snapshot data
        self.make_snapshot(snapshot_mode, float(ret_avg), meta_info)

    def loss_fcn(self, q_vals: to.Tensor, expected_q_vals: to.Tensor) -> to.Tensor:
        r"""
        The Huber loss function on the one-step TD error $\delta = Q(s,a) - (r + \gamma \max_a Q(s^\prime, a))$.

        :param q_vals: state-action values $Q(s,a)$, from policy network
        :param expected_q_vals: expected state-action values $r + \gamma \max_a Q(s^\prime, a)$, from target network
        :return: loss value
        """
        return nn.functional.smooth_l1_loss(q_vals, expected_q_vals)

    def update(self):
        """ Update the policy's and q_targ Q-function's parameters on transitions sampled from the replay memory. """
        losses = to.zeros(self.num_batch_updates)
        policy_grad_norm = to.zeros(self.num_batch_updates)

        for b in tqdm(range(self.num_batch_updates), total=self.num_batch_updates,
                      desc=f'Updating', unit='batches', file=sys.stdout, leave=False):

            # Sample steps and the associated next step from the replay memory
            steps, next_steps = self._memory.sample(self.batch_size)
            steps.torch(data_type=to.get_default_dtype())
            next_steps.torch(data_type=to.get_default_dtype())

            # Create masks for the non-final observations
            not_done = to.tensor(1. - steps.done, dtype=to.get_default_dtype())

            # Compute the state-action values Q(s,a) using the current DQN policy
            q_vals = self.expl_strat.policy.q_values_argmax(steps.observations)

            # Compute the second term of TD-error
            with to.no_grad():
                next_v_vals = self.q_targ.q_values_argmax(next_steps.observations)
                expected_q_val = steps.rewards + not_done*self.gamma*next_v_vals

            # Compute the loss, clip the gradients if desired, and do one optimization step
            loss = self.loss_fcn(q_vals, expected_q_val)
            losses[b] = loss.data
            self.optim.zero_grad()
            loss.backward()
            policy_grad_norm[b] = self.clip_grad(self.expl_strat.policy, self.max_grad_norm)
            self.optim.step()

            # Update the q_targ network by copying all weights and biases from the DQN policy
            if (self._curr_iter*self.num_batch_updates + b)%self.target_update_intvl == 0:
                self.q_targ.load_state_dict(self.expl_strat.policy.state_dict())

        # Schedule the exploration parameter epsilon
        self.expl_strat.schedule_eps(self._curr_iter)

        # Update the learning rate if a scheduler has been specified
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

        # Logging
        with to.no_grad():
            self.logger.add_value('loss after', to.mean(losses), 4)
        self.logger.add_value('expl strat eps', self.expl_strat.eps, 4)
        self.logger.add_value('avg policy grad norm', to.mean(policy_grad_norm), 4)
        if self._lr_scheduler is not None:
            self.logger.add_value('avg learning rate', np.mean(self._lr_scheduler.get_lr()), 6)

    def reset(self, seed: int = None):
        # Reset the exploration strategy, internal variables and the random seeds
        super().reset(seed)

        # Re-initialize sampler in case env or policy changed
        self.sampler_trn.reinit(self._env, self._expl_strat)

        # Reset the replay memory
        self._memory.reset()

        # Reset the learning rate scheduler
        if self._lr_scheduler is not None:
            self._lr_scheduler.last_epoch = -1

    def init_modules(self, warmstart: bool, suffix: str = '', prefix: str = None, **kwargs):
        if prefix is None:
            prefix = f'iter_{self._curr_iter - 1}'

        ppi = kwargs.get('policy_param_init', None)
        tpi = kwargs.get('target_param_init', None)

        if warmstart and ppi is not None and tpi is not None:
            self._policy.init_param(ppi)
            self.q_targ.init_param(tpi)
            print_cbt('Learning given an fixed parameter initialization.', 'w')

        elif warmstart and ppi is None and self._curr_iter > 0:
            self._policy = load_prefix_suffix(
                self._policy, 'policy', 'pt', self.save_dir,
                meta_info=dict(prefix=prefix, suffix=suffix)
            )
            self.q_targ = load_prefix_suffix(
                self.q_targ, 'target', 'pt', self.save_dir,
                meta_info=dict(prefix=prefix, suffix=suffix)
            )
            print_cbt(f'Learning given the results from iteration {self._curr_iter - 1}', 'w')

        else:
            # Reset the policy
            self._policy.init_param()
            self.q_targ.init_param()
            print_cbt('Learning from scratch.', 'w')

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        save_prefix_suffix(self._expl_strat.policy, 'policy', 'pt', self.save_dir, meta_info)
        save_prefix_suffix(self.q_targ, 'target', 'pt', self.save_dir, meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            save_prefix_suffix(self._env, 'env', 'pkl', self.save_dir, meta_info)
