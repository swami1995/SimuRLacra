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

from pyrado.environment_wrappers.base import EnvWrapperAct, EnvWrapper
from pyrado.spaces.box import BoxSpace


class ActNormWrapper(EnvWrapperAct):
    """Environment wrapper which normalizes the action space, such that all action values are in range [-1, 1]."""

    def _process_act(self, act: np.ndarray) -> np.ndarray:
        # Get the bounds of the inner action space
        lb, ub = self.wrapped_env.act_space.bounds

        # Denormalize action
        act_denorm = lb + (act + 1) * (ub - lb) / 2
        return act_denorm  # can be out of action space, but this has to be checked by the environment

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

    def _process_obs_space(self, space: BoxSpace) -> BoxSpace:
        """
        Return the modified observation space.
        Override if the operation defined in _process_obs affects shape or bounds of the observation vector.
        :param space: inner env observation space
        :return: action space to report for this env
        """
        if not isinstance(space, BoxSpace):
            raise NotImplementedError("Only implemented ActNormWrapper._process_act_space() for BoxSpace!")
        if 'act' not in space.labels:
            new_space = BoxSpace(np.append(space.bound_lo, [-1]), np.append(space.bound_up, [1]), labels=np.append(space.labels, ['act']))
        else:
            new_space = space

        return new_space

    def _process_act_space(self, space: BoxSpace) -> BoxSpace:
        if not isinstance(space, BoxSpace):
            raise NotImplementedError("Only implemented ActNormWrapper._process_act_space() for BoxSpace!")

        # Return space with same shape but bounds from -1 to 1
        return BoxSpace(-np.ones(space.shape), np.ones(space.shape), labels=space.labels)

    @property
    def obs_space(self) -> BoxSpace:
        # Process space
        # By not using _wrapped_env directly, we can mix this class with EnvWrapperAct
        return self._process_obs_space(super().obs_space)

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        # Forward to EnvWrapper, which delegates to self._wrapped_env
        init_obs = super().reset(init_state=init_state, domain_param=domain_param)
        self.act_prev = np.array([0.0,])
        # Return processed observation
        return self._process_obs(init_obs, self.act_prev)

    def step(self, act: np.ndarray) -> tuple:
        # Step inner environment
        # By not using _wrapped_env directly, we can mix this class with EnvWrapperAct
        
        obs, rew, done, info = super().step(act + self.act_prev)
        self.act_prev = np.maximum(np.minimum(act + self.act_prev, 1),-1)
        # Return processed observation
        return self._process_obs(obs, self.act_prev), rew, done, info

    def _process_act_space(self, space: BoxSpace):
        """
        Return the modified action space. Override if the operation defined in _process_action affects
        shape or bounds of the action vector.
        :param space: inner env action space
        :return: action space to report for this env
        """
        return BoxSpace(space.bound_lo*2, space.bound_up*2, labels=space.labels)

    @property
    def act_space(self) -> BoxSpace:
        # Process space
        # By not using _wrapped_env directly, we can mix this class with EnvWrapperObs
        return self._process_act_space(super().act_space)

