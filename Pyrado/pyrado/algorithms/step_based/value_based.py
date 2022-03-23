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

import os.path as osp
from abc import ABC, abstractmethod
from math import ceil
from typing import Optional, Union
import os
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import psutil

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.utils import ReplayMemory
from pyrado.environments.base import Env
from pyrado.exploration.stochastic_action import EpsGreedyExplStrat, SACExplStrat
from pyrado.logger.step import ConsolePrinter, CSVPrinter, StepLogger, TensorBoardPrinter
from pyrado.policies.base import Policy, TwoHeadedPolicy
from pyrado.policies.feed_forward.dummy import DummyPolicy, RecurrentDummyPolicy
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler, ParallelRolloutSamplerTensor
from pyrado.utils.input_output import print_cbt_once
from pyrado.sampling.step_sequence import StepSequence
import torch
import gc

class ValueBased(Algorithm, ABC):
    """Base class of all value-based algorithms"""

    def __init__(
        self,
        save_dir: pyrado.PathLike,
        env: Env,
        policy: Union[Policy, TwoHeadedPolicy],
        memory_size: int,
        gamma: float,
        max_iter: int,
        num_updates_per_step: int,
        target_update_intvl: int,
        num_init_memory_steps: int,
        min_rollouts: int,
        min_steps: int,
        batch_size: int,
        eval_intvl: int,
        max_grad_norm: float,
        num_workers: int,
        logger: StepLogger,
        use_trained_policy_for_refill: bool = False,
        env_sim = None,
    ):
        r"""
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param policy: policy to be updated
        :param memory_size: number of transitions in the replay memory buffer, e.g. 1000000
        :param gamma: temporal discount factor for the state values
        :param max_iter: maximum number of iterations (i.e. policy updates) that this algorithm runs
        :param num_updates_per_step: number of (batched) gradient updates per algorithm step
        :param target_update_intvl: number of iterations that pass before updating the target network
        :param num_init_memory_steps: number of samples used to initially fill the replay buffer with, pass `None` to
                                      fill the buffer completely
        :param min_rollouts: minimum number of rollouts sampled per policy update batch
        :param min_steps: minimum number of state transitions sampled per policy update batch
        :param batch_size: number of samples per policy update batch
        :param eval_intvl: interval in which the evaluation rollouts are collected, also the interval in which the
                           logger prints the summary statistics
        :param max_grad_norm: maximum L2 norm of the gradients for clipping, set to `None` to disable gradient clipping
        :param num_workers: number of environments for parallel sampling
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        :param use_trained_policy_for_refill: whether to use the trained policy instead of a dummy policy to refill the
                                              replay buffer after resets
        """
        if not isinstance(env, Env):
            raise pyrado.TypeErr(given=env, expected_type=Env)
        if not isinstance(memory_size, int):
            raise pyrado.TypeErr(given=memory_size, expected_type=int)
        if not (num_init_memory_steps is None or isinstance(num_init_memory_steps, int)):
            raise pyrado.TypeErr(given=num_init_memory_steps, expected_type=int)

        if logger is None:
            # Create logger that only logs every logger_print_intvl steps of the algorithm
            logger = StepLogger(print_intvl=eval_intvl)
            logger.printers.append(ConsolePrinter())
            logger.printers.append(CSVPrinter(osp.join(save_dir, "progress.csv")))
            logger.printers.append(TensorBoardPrinter(osp.join(save_dir, "tb")))

        # Call Algorithm's constructor
        super().__init__(save_dir, max_iter, policy, logger)

        self._env = env
        self._env_sim = env_sim
        self._memory = ReplayMemory(memory_size)
        self.gamma = gamma
        self.target_update_intvl = target_update_intvl
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        if num_init_memory_steps is None:
            self.num_init_memory_steps = memory_size
        else:
            self.num_init_memory_steps = max(min(num_init_memory_steps, memory_size), batch_size)

        # Heuristic for number of gradient updates per step
        if num_updates_per_step is None:
            self.num_batch_updates = ceil(min_steps / env.max_steps) if min_steps is not None else min_rollouts
        else:
            self.num_batch_updates = num_updates_per_step

        # Create sampler for initial filling of the replay memory
        if policy.is_recurrent:
            self.init_expl_policy = RecurrentDummyPolicy(env.spec, policy.hidden_size)
        else:
            self.init_expl_policy = DummyPolicy(env.spec)
        self.sampler_init = ParallelRolloutSamplerTensor(#ParallelRolloutSampler(
            self._env,
            self.init_expl_policy,
            num_workers=num_workers,
            min_steps=self.num_init_memory_steps,
        )

        # Create sampler for initial filling of the replay memory and evaluation
        self.sampler_eval = ParallelRolloutSamplerTensor(# ParallelRolloutSampler(
            self._env,
            self._policy,
            num_workers=num_workers,
            min_steps=None,
            min_rollouts=5,
            show_progress_bar=True,
        )

        if self._env_sim is None:
            self._env_sim = self._env
        self.sampler_transfer = ParallelRolloutSamplerTensor(# ParallelRolloutSampler(
            self._env_sim,
            self._policy,
            num_workers=num_workers,
            min_steps=None,
            min_rollouts=1,
            show_progress_bar=True,
        )        

        self._expl_strat = None  # must be implemented by subclass
        self._sampler = None  # must be implemented by subclass

        self._fill_with_init_sampler = True  # use the init sampler with the dummy policy on first run
        self._use_trained_policy_for_refill = use_trained_policy_for_refill

    @property
    def expl_strat(self) -> Union[SACExplStrat, EpsGreedyExplStrat]:
        return self._expl_strat

    @property
    def memory(self) -> ReplayMemory:
        """Get the replay memory."""
        return self._memory

    def get_obs(self, traj):
        acts = np.insert(traj['data_Vclip'][:,1]/6, 0, 0)
        obs = np.stack([traj['data_pos'][:,2], traj['data_theta'][:,2], traj['data_vel'][:,2], traj['data_thetadot'][:,2], acts], axis=1)
        return obs

    def compute_th_ddot(self, traj):
        thetadot = traj['data_thetadot'][:,2]
        theta_ddot = thetadot[1:] - thetadot[:-1]
        theta_ddot = theta_ddot/0.05
        theta_ddot = np.insert(theta_ddot, 0, 0)
        return theta_ddot
        
    def load_memory(self, traj_path='episodes/15_rand_actdiff2_long_maxa6_T300_rlclip05/'):
        from os import listdir
        import scipy.io as io
        from os.path import isfile, join
        traj_files = [join(traj_path, f) for f in listdir(traj_path) if isfile(join(traj_path, f))]
        # trajs = [ ]
        rollouts = []
        for file in traj_files:
            traj = io.loadmat(file)
            obs_hist_np = self.get_obs(traj)
            act_hist_np = traj['data_action'][:,1:]

            ## step through env
            if self.grad_vi:
                th_ddot_hist = self.compute_th_ddot(traj)
                obs = torch.tensor(obs_hist_np).requires_grad_(True)
                act = torch.tensor(act_hist_np).requires_grad_(True)
                th_ddot = torch.tensor(th_ddot_hist)
                next_obs, rew, dones, info = self._env_sim.step_diff_state(obs, act, th_ddot)
                obs_grad = torch.stack([torch.cat(torch.autograd.grad(next_obs[:,i].sum(), [obs, act], retain_graph=True), dim=-1) for i in range(next_obs.shape[1])], dim=1).detach().numpy()
                rew_grad = torch.cat(torch.autograd.grad(rew.sum(), [obs, act]), dim=-1).detach().numpy()
                rew_np = rew.detach().numpy()
                obs_hist = []
                act_hist = []
                rew_hist = []
                env_info_hist = []
                for i in range(obs.shape[0]):
                    obs_hist.append(obs_hist_np[i])
                    act_hist.append(act_hist_np[i])
                    rew_hist.append(rew_np[i])
                    env_info = {}
                    env_info['obs_grad'] = obs_grad[i]
                    env_info['rew_grad'] = rew_grad[i]
                    env_info_hist.append(env_info)
            else:
                obs = torch.tensor(obs_hist_np).requires_grad_(True)
                act = torch.tensor(act_hist_np).requires_grad_(True)
                next_obs, rew, dones, info = self._env_sim.step_diff_state(obs, act)
                rew_np = rew.detach().numpy()
                obs_hist = []
                act_hist = []
                rew_hist = []
                env_info_hist = []
                for i in range(obs.shape[0]):
                    obs_hist.append(obs_hist_np[i])
                    act_hist.append(act_hist_np[i])
                    rew_hist.append(rew_np[i])
                    env_info = {np.zeros(obs_hist_np[i].shape)}
                    env_info_hist.append(env_info)
            res = StepSequence(
            observations=obs_hist,
            actions=act_hist,
            rewards=rew_hist,
            env_infos=env_info_hist,
            complete=True,  # the rollout function always returns complete paths
            continuous=False,
            )
            rollouts.append(res)
        self._memory.push(rollouts)

    def tranform_rollouts(self, ros):
        from os import listdir
        from os.path import isfile, join
        # trajs = [ ]
        # ros = self.memory._memory
        rollouts = []
        # for i in range(size):
        # steps = ros[self.batch_size]
        # traj = io.loadmat(file)
        for ro in ros:
            obs_hist_np = ro.observations
            act_hist_np = ro.actions
            th_ddot_np = ro.th_ddot

            ## step through env
            obs = torch.tensor(obs_hist_np).requires_grad_(True)[:-1].float()
            act = torch.tensor(act_hist_np).requires_grad_(True).float()
            th_ddot = torch.tensor(th_ddot_np)[:-1].float()
            # ipdb.set_trace()
            next_obs, rew, dones, info = self._env_sim.step_diff_state(obs, act, th_ddot)
            obs_grad = torch.stack([torch.cat(torch.autograd.grad(next_obs[:,i].sum(), [obs, act], retain_graph=True), dim=-1) for i in range(next_obs.shape[1])], dim=1).detach().numpy()
            rew_grad = torch.cat(torch.autograd.grad(rew.sum(), [obs, act]), dim=-1).detach().unsqueeze(1).numpy()
            rew_np = rew.detach().numpy()
            obs_hist = []
            act_hist = []
            rew_hist = []
            th_ddot_hist = []
            env_info_hist = []
            for i in range(obs.shape[0]):
                obs_hist.append(obs_hist_np[i])
                act_hist.append(act_hist_np[i])
                rew_hist.append(rew_np[i])
                th_ddot_hist.append(th_ddot_np[i])
                env_info = {}
                env_info['obs_grad'] = obs_grad[i]
                env_info['rew_grad'] = rew_grad[i]
                env_info_hist.append(env_info)
            obs_hist.append(obs_hist_np[obs.shape[0]])
            th_ddot_hist.append(th_ddot_np[obs.shape[0]])
            # ipdb.set_trace()
            res = StepSequence(
            observations=obs_hist,
            actions=act_hist,
            rewards=rew_hist,
            env_infos=env_info_hist,
            th_ddot=th_ddot_hist,
            complete=True,  # the rollout function always returns complete paths
            )
            rollouts.append(res)
            # ipdb.set_trace()
        return rollouts

    def step(self, snapshot_mode: str, meta_info: dict = None):
        if self._memory.isempty:
            # Warm-up phase
            print_cbt_once(f"Empty replay memory, collecting {self.num_init_memory_steps} samples.", "w")
            # Sample steps and store them in the replay memory
            if self._fill_with_init_sampler:
                ros = self.sampler_init.sample()
                # self.tranform_rollouts(ros)
                self._fill_with_init_sampler = not self._use_trained_policy_for_refill
            else:
                # Save old bounds from the sampler
                min_rollouts = self.sampler.min_rollouts
                min_steps = self.sampler.min_steps
                # Set and sample with the init sampler settings
                self.sampler.set_min_count(min_steps=self.num_init_memory_steps)
                ros = self.sampler.sample()
                # Revert back to initial parameters
                self.sampler.set_min_count(min_rollouts=min_rollouts, min_steps=min_steps)
            self._memory.push(ros)
        else:
            # Sample steps and store them in the replay memory
            ros = self.sampler.sample()
            self._memory.push(ros)
        self._cnt_samples += sum([ro.length for ro in ros])  # don't count the evaluation samples

        # Log metrics computed from the old policy (before the update)
        if self._curr_iter % self.logger.print_intvl == 0:
            print('before test: ', psutil.Process().memory_info().rss / (1024 * 1024))
            ros = self.sampler_eval.sample()
            print('after test: ', psutil.Process().memory_info().rss / (1024 * 1024))
            rets = [ro.undiscounted_return() for ro in ros]
            ret_max = np.max(rets)
            ret_med = np.median(rets)
            ret_avg = np.mean(rets)
            ret_min = np.min(rets)
            ret_std = np.std(rets)
        else:
            ret_max, ret_med, ret_avg, ret_min, ret_std = 5 * [-pyrado.inf]  # dummy values
        self.logger.add_value("max return", ret_max, 4)
        self.logger.add_value("median return", ret_med, 4)
        self.logger.add_value("avg return", ret_avg, 4)
        self.logger.add_value("min return", ret_min, 4)
        self.logger.add_value("std return", ret_std, 4)
        self.logger.add_value("avg memory reward", self._memory.avg_reward(), 4)
        self.logger.add_value("avg rollout length", np.mean([ro.length for ro in ros]), 4)
        self.logger.add_value("num total samples", self._cnt_samples)

        # Save snapshot data
        self.make_snapshot(snapshot_mode, float(ret_avg), meta_info)

        # Use data in the memory to update the policy and the Q-functions
        self.update()
        if self.num_update_calls % 2==0:
            self.plot_vfunc_state_space(str(self.num_update_calls))

    def step_sim2sim(self, snapshot_mode: str, meta_info: dict = None):
        if self._memory.isempty:
            # Warm-up phase
            print_cbt_once(f"Empty replay memory, collecting {self.num_init_memory_steps} samples.", "w")
            # Sample steps and store them in the replay memory
            # Save old bounds from the sampler
            rollouts = []
            for n in range(self.num_init_rollouts):
                ros = self.sampler_transfer.sample()
                rollouts += ros
            # Revert back to initial parameters
            ros = rollouts
        else:
            # Sample steps and store them in the replay memory
            ros = self.sampler_transfer.sample()
        ros_with_jac = self.tranform_rollouts(ros)

        self._memory.push(ros_with_jac)
        cnt_samples_step = sum([ro.length for ro in ros])
        self._cnt_samples += cnt_samples_step  # don't count the evaluation samples

        # Log metrics computed from the old policy (before the update)
        if self._curr_iter % self.logger.print_intvl == 0:
            ros = self.sampler_eval.sample()
            rets = [ro.undiscounted_return() for ro in ros]
            ret_max = np.max(rets)
            ret_med = np.median(rets)
            ret_avg = np.mean(rets)
            ret_min = np.min(rets)
            ret_std = np.std(rets)
        else:
            ret_max, ret_med, ret_avg, ret_min, ret_std = 5 * [-pyrado.inf]  # dummy values
        self.logger.add_value("max return", ret_max, 4)
        self.logger.add_value("median return", ret_med, 4)
        self.logger.add_value("avg return", ret_avg, 4)
        self.logger.add_value("min return", ret_min, 4)
        self.logger.add_value("std return", ret_std, 4)
        self.logger.add_value("avg memory reward", self._memory.avg_reward(), 4)
        self.logger.add_value("avg rollout length", np.mean([ro.length for ro in ros]), 4)
        self.logger.add_value("num total samples", self._cnt_samples)

        # self.plot_trajectories(ros)
        # self.plot_trajectories_actions(ros)
        # Save snapshot data
        self.make_snapshot(snapshot_mode, float(ret_avg), meta_info)

        # Use data in the memory to update the policy and the Q-functions
        self.num_batch_updates = cnt_samples_step
        self.batch_size_used = min(len(self._memory)-2, self.batch_size)
        self.update()

    def plot_vfunc_state_space(self, num_update_calls = None):
        num_samples = 600*600
        sample_states = torch.rand((num_samples, 2))*2-1
        sample_states[:,0] *= np.pi
        sample_states[:,1] *= 3.5*np.pi
        if self._env.obs_space.shape[0]== 5:
            sample_states = torch.stack([sample_states[:,0]*0, sample_states[:,0], sample_states[:,1]*0, sample_states[:,1], sample_states[:,1]*0], dim=1)
        else:
            sample_states = torch.stack([sample_states[:,0]*0, sample_states[:,0], sample_states[:,1]*0, sample_states[:,1]], dim=1)
        V = []
        states = []
        # model = model.cpu()
        # ipdb.set_trace()
        with torch.no_grad():
            # if True:
            #     for i in range(num_samples//self.batch_size):
            #         state = sample_states[i*batch_size: (i+1)*batch_size]

            #         # state = torch.stack(state1)
            #         # for buffer_i in buffer[i*self.batch_size: (i+1)*self.batch_size]:
            #             # for j in range(len(buffer_i)):
            #             # state1[j].append(buffer_i[j])
            #         # for j in range(len(state1)):
            #         #     try:
            #         #         torch.stack(state1[j])
            #         #     except:
            #         #         ipdb.set_trace()
            #         # try:
            #         #     state, _, _, _, _ = map(torch.stack, buffer[i*self.batch_size: (i+1)*self.batch_size])
            #         # except:
            #         #     ipdb.set_trace()

            #         states.append(state)
            #         V.append(model(state).squeeze().cpu())
            #         # if i%20==0:
            #         #     print(i, ) 
            # V = torch.cat(V, dim=0).cpu()*0 + 1
            action, _ = self.policy(sample_states)
            sample_state_action = torch.cat([sample_states, action], dim=1)
            V = torch.clamp(self.qfcn_1(sample_state_action).squeeze().cpu(), 0, 20000)
        # states = torch.cat(states, dim=0).cpu().detach().requires_grad_(False)
        states = sample_states
        # print("buffer size", states.shape[0])
        # ipdb.set_trace()
        plt.clf()
        plt.scatter(states[:,1], states[:,3], c=V, s=2, cmap='gray')
        # plt.scatter(samples[:,0], samples[:,1], c='r', s=2)
        # plt.scatter(new_samples[:,0], new_samples[:,1], c='b', s=2)
        # for i in range(samples.shape[0]):
        #     plt.plot([samples[i,0], new_samples[i,0]], [samples[i,1], new_samples[i,1]], linewidth=1)
        plt.ylim([-3.5*np.pi, 3.5*np.pi])
        plt.xlim([-np.pi, np.pi])
            
        # path = env_name + 'explore_backward/rrt_lqr_act_cost005_rand200_top20_step13_5200switchonpolicy'
        if num_update_calls is not None:
            path = osp.join(self.save_dir, 'qfn_plots/')
            # path = self.save_dir + 
            os.makedirs(path, exist_ok=True)
            plt.savefig(osp.join(path, f'val_states_{num_update_calls}.png'))
            return None

        path = 'saved_plots/qfn_plots/'
        os.makedirs(path, exist_ok=True)
        # plt.savefig(path + 'val_states_rlvgi_euler405_clamped.png')
        # plt.savefig(path + 'val_states_dt05_rlvgi_rk4_parrol_MAXA6_fixed4_unclamped.png')
        plt.savefig(path + 'val_states_sac_dt01_rl_rk4_parrol_MAXA6_fixed4_nosmooth_hidden256_clamped.png')
        # plt.savefig(path + 'val_states_dt05_rl_euler_MAXA6_unclamped.png')
        # del V
        # del states
        # del sample_states
        # plt.clf()
        # plt.close()
        # gc.collect()
        ipdb.set_trace()


    def plot_trajectories(self, ros):
        rets = [ro.undiscounted_return() for ro in ros]
        for ro in ros:
            if ro.undiscounted_return() == np.min(rets):
                ro_min = ro
        ipdb.set_trace()
        states = ro_min.observations[:-1]
        actions = ro_min.actions
        import matplotlib.pyplot as plt
        dt=0.05
        plt.plot(np.arange(len(states))*dt, states[:,0], label='x')
        plt.plot(np.arange(len(states))*dt, states[:,1], label='th')
        plt.plot(np.arange(len(states))*dt, states[:,2], label='xdot')
        plt.plot(np.arange(len(states))*dt, states[:,3], label='thdot')
        plt.plot(np.arange(len(states))*dt, states[:,4] + actions[:,0], label='act')
        plt.legend()
        plt.savefig('saved_plots/states_rl_rk405_wildmod_debug_min.png')

        for i in range(10):
            plt.clf()
            ro_min = ros[i]
            states = ro_min.observations[:-1]
            actions = ro_min.actions
            plt.plot(np.arange(len(states))*dt, states[:,0], label='x')
            plt.plot(np.arange(len(states))*dt, states[:,1], label='th')
            plt.plot(np.arange(len(states))*dt, states[:,2], label='xdot')
            plt.plot(np.arange(len(states))*dt, states[:,3], label='thdot')
            plt.plot(np.arange(len(states))*dt, states[:,4] + actions[:,0], label='act')
            plt.legend()
            plt.savefig(f'saved_plots/states_rl_rk405_wildmod_debug{i}.png')
        ipdb.set_trace()
    def plot_trajectories_actions(self, ros):
        rets = [ro.undiscounted_return() for ro in ros]
        for ro in ros:
            if ro.undiscounted_return() == np.min(rets):
                ro_min = ro
        states = ro_min.observations[:-1]
        actions = ro_min.actions
        import matplotlib.pyplot as plt
        dt=0.05
        T = 200
        plt.plot(np.arange(len(states[:T]))*dt, np.clip(states[:T,4] + np.clip(actions[:T,0], -0.5, 0.5), -1, 1), label='act_min')
        for i in range(5):
            plt.plot(np.arange(len(ros[i].observations[:-1][:T]))*dt, np.clip(ros[i].observations[:-1][:T,4] + np.clip(ros[i].actions[:T,0], -0.5, 0.5), -1, 1), label=f'act_{i}')
        plt.legend()
        plt.savefig('saved_plots/policy_actions_rlclip05_grad_vi_wild_false_again.png')
        ipdb.set_trace()

    def step_sim2real(self, snapshot_mode: str, meta_info: dict = None):
        self.load_memory()
        self._cnt_samples = self._memory._memory.length  # don't count the evaluation samples

        # Log metrics computed from the old policy (before the update)
        # if self._curr_iter % self.logger.print_intvl == 0:
        #     ros = self.sampler_eval.sample()
        #     rets = [ro.undiscounted_return() for ro in ros]
        #     ret_max = np.max(rets)
        #     ret_med = np.median(rets)
        #     ret_avg = np.mean(rets)
        #     ret_min = np.min(rets)
        #     ret_std = np.std(rets)
        # else:
        #     ret_max, ret_med, ret_avg, ret_min, ret_std = 5 * [-pyrado.inf]  # dummy values
        # self.logger.add_value("max return", ret_max, 4)
        # self.logger.add_value("median return", ret_med, 4)
        # self.logger.add_value("avg return", ret_avg, 4)
        # self.logger.add_value("min return", ret_min, 4)
        # self.logger.add_value("std return", ret_std, 4)
        # self.logger.add_value("avg memory reward", self._memory.avg_reward(), 4)
        # self.logger.add_value("avg rollout length", np.mean([ro.length for ro in ros]), 4)
        # self.logger.add_value("num total samples", self._cnt_samples)
        self.num_batch_updates = 300#cnt_samples_step
        self.update()

        # Save snapshot data
        self.make_snapshot(snapshot_mode, 300, meta_info)
        # Use data in the memory to update the policy and the Q-functions


    @abstractmethod
    def update(self):
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None):
        # Reset the exploration strategy, internal variables and the random seeds
        super().reset(seed)

        # Re-initialize samplers in case env or policy changed
        self.sampler_init.reinit(self._env, self.init_expl_policy)
        self.sampler.reinit(self._env, self._expl_strat)
        self.sampler_eval.reinit(self._env, self._policy)

        # Reset the replay memory
        self._memory.reset()

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            pyrado.save(self._env, "env.pkl", self.save_dir)
            pyrado.save(self._expl_strat.policy, "policy.pt", self.save_dir, use_state_dict=True)
        else:
            pyrado.save(
                self._expl_strat.policy,
                "policy.pt",
                self.save_dir,
                prefix=meta_info.get("prefix", ""),
                suffix=meta_info.get("suffix", ""),
                use_state_dict=True,
            )
