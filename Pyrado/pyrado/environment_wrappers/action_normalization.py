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
import torch
from pyrado.environment_wrappers.base import EnvWrapperAct, EnvWrapper
from pyrado.spaces.box import BoxSpace
from torch.cuda.amp import custom_bwd, custom_fwd

class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, min, max):
        ctx.save_for_backward(input, min, max)
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input,min,max = ctx.saved_tensors
        # mask_neg = (1-(input>max).float()*(grad_output>0).float())
        # mask_pos = (1-(input<min).float()*(grad_output<0).float())
        # grad_output = grad_output*mask_pos*mask_neg
        return grad_output.clone(), None, None


def dclamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)

class ActNormWrapper(EnvWrapperAct):
    """Environment wrapper which normalizes the action space, such that all action values are in range [-1, 1]."""

    def _process_act(self, act: np.ndarray) -> np.ndarray:
        # Get the bounds of the inner action space
        lb, ub = self.wrapped_env.act_space.bounds

        # Denormalize action
        act_denorm = lb + (act + 1) * (ub - lb) / 2
        return act_denorm  # can be out of action space, but this has to be checked by the environment

    def _process_act_tensor(self, act):
        lb, ub = self.wrapped_env.act_space.bounds
        lb, ub = torch.tensor(lb).to(act).unsqueeze(0), torch.tensor(ub).to(act).unsqueeze(0)

        act_denorm = lb + (act + 1) * (ub - lb) / 2
        return act_denorm

    def _process_act_space(self, space: BoxSpace) -> BoxSpace:
        if not isinstance(space, BoxSpace):
            raise NotImplementedError("Only implemented ActNormWrapper._process_act_space() for BoxSpace!")

        # Return space with same shape but bounds from -1 to 1
        return BoxSpace(-np.ones(space.shape), np.ones(space.shape), labels=space.labels)

class ObsActCatWrapper(EnvWrapper):
    """Environment wrapper which normalizes the action space, such that all action values are in range [-1, 1]."""

    def _process_obs(self, obs: np.ndarray, act):
        """
        Return the modified observation vector to be returned from this environment.

        :param obs: observation from the inner environment
        :return: changed observation vector
        """
        return np.append(obs, act)
        # return obs

    def _process_obs_tensor(self, obs, act):
        return torch.cat([obs, act], dim=-1)
        # return obs

    def _process_obs_space(self, space: BoxSpace) -> BoxSpace:
        """
        Return the modified observation space.
        Override if the operation defined in _process_obs affects shape or bounds of the observation vector.
        :param space: inner env observation space
        :return: action space to report for this env
        """
        # return space
        if not isinstance(space, BoxSpace):
            raise NotImplementedError("Only implemented ActNormWrapper._process_act_space() for BoxSpace!")
        if 'act' not in space.labels:
            new_space = BoxSpace(np.append(space.bound_lo, [-1]), np.append(space.bound_up, [1]), labels=np.append(space.labels, ['act']))
        else:
            new_space = space

        return new_space

    # def _process_act_space(self, space: BoxSpace) -> BoxSpace:
    #     if not isinstance(space, BoxSpace):
    #         raise NotImplementedError("Only implemented ActNormWrapper._process_act_space() for BoxSpace!")

    #     # Return space with same shape but bounds from -1 to 1
    #     return BoxSpace(-np.ones(space.shape), np.ones(space.shape), labels=space.labels)

    @property
    def obs_space(self) -> BoxSpace:
        # Process space
        # By not using _wrapped_env directly, we can mix this class with EnvWrapperAct
        # return self._process_obs_space(super().obs_space)
        return self._process_obs_space(self._wrapped_env._wrapped_env._state_space)

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        # Forward to EnvWrapper, which delegates to self._wrapped_env
        init_obs = super().reset(init_state=init_state, domain_param=domain_param)
        # init_obs = self._wrapped_env._wrapped_env.state
        self.act_prev = np.array([0.0,])
        # Return processed observation
        return self._process_obs(init_obs, self.act_prev)

    def step(self, act: np.ndarray) -> tuple:
        # Step inner environment
        # By not using _wrapped_env directly, we can mix this class with EnvWrapperAct
        
        obs, rew, done, info = super().step(act + self.act_prev)
        self.act_prev = np.maximum(np.minimum(act + self.act_prev, 1),-1)
        # Return processed observation
        state = self._wrapped_env._wrapped_env.state
        return self._process_obs(state, self.act_prev), rew, done, info

    def step_diff(self, obs, act, th_ddot=None):
        obs_state = obs[:, :-1]
        obsn, rew, done, info = super().step_diff(obs_state, act + obs[:, -1:], th_ddot)
        # obsn, rew, done, info = super().step_diff(obs_state, act, th_ddot)
        act_prev = dclamp(act + obs[:, -1:], torch.tensor(-1), torch.tensor(1))
        return self._process_obs_tensor(obsn, act_prev), rew, done, info

    def step_diff_state(self, obs, act, th_ddot=None):
        obs_state = obs[:, :-1]
        obsn, rew, done, info = super().step_diff_state(obs_state, obs[:, -1:], act, th_ddot)
        # obsn, rew, done, info = super().step_diff_state(obs_state, act, th_ddot)
        act = dclamp(act, torch.tensor(-0.5), torch.tensor(0.5))
        act_prev = dclamp(act + obs[:, -1:], torch.tensor(-1), torch.tensor(1))
        return self._process_obs_tensor(obsn, act_prev), rew, done, info

    def step11(self, act):
        obs_state = torch.tensor(self._wrapped_env._wrapped_env.state).unsqueeze(0).requires_grad_(True).float()
        act_full = torch.tensor(self.act_prev + act).unsqueeze(0).requires_grad_(True).float()
        obsn, rew, done, info = super().step_diff_state(obs_state, act_full)
        # print("Obsn Shape: ", obsn.shape)
        # obs_grad = torch.cat([torch.cat(torch.autograd.grad(obsn[0,i], [obs_state, act_full], retain_graph=True),dim=1) for i in range(obsn.shape[1])], dim=0)
        # rew_grad = torch.cat(torch.autograd.grad(rew.sum(), [obs_state, act_full]), dim=1)
        # act_prev = dclamp(act + obs[:, -1:], torch.tensor(-1), torch.tensor(1))
        self.act_prev = torch.clamp(act_full, -1, 1).detach().numpy()[0]
        # info['rew_grad'] = rew_grad
        # info['obs_grad'] = obs_grad
        # del obs_state
        # del act_full
        return self._process_obs(obsn.detach().numpy()[0], self.act_prev), rew.item(), done.item(), info

    def _process_act_space(self, space: BoxSpace):
        """
        Return the modified action space. Override if the operation defined in _process_action affects
        shape or bounds of the action vector.
        :param space: inner env action space
        :return: action space to report for this env
        """
        return BoxSpace(space.bound_lo*0.5, space.bound_up*0.5, labels=space.labels)

    @property
    def act_space(self) -> BoxSpace:
        # Process space
        # By not using _wrapped_env directly, we can mix this class with EnvWrapperObs
        return self._process_act_space(super().act_space)

