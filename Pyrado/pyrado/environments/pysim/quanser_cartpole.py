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

from abc import abstractmethod
from typing import Optional

import numpy as np
from init_args_serializer.serializable import Serializable

import pyrado
from pyrado.environments.pysim.base import SimPyEnv
from pyrado.environments.quanser import MAX_ACT_QCP
from pyrado.spaces.box import BoxSpace
from pyrado.tasks.base import Task
from pyrado.tasks.desired_state import RadiallySymmDesStateTask
from pyrado.tasks.final_reward import FinalRewMode, FinalRewTask
from pyrado.tasks.reward_functions import ExpQuadrErrRewFcn, QuadrErrRewFcn
import torch
import ipdb

class QCartPoleSim(SimPyEnv, Serializable):
    """Base Environment for the Quanser Cart-Pole swing-up and stabilization task"""

    def __init__(
        self,
        dt: float,
        max_steps: int,
        task_args: Optional[dict],
        long: bool,
        simple_dynamics: bool,
        wild_init: str,
        mass=None,
    ):
        r"""
        Constructor

        :param dt: simulation step size [s]
        :param max_steps: maximum number of simulation steps
        :param task_args: arguments for the task construction
        :param long: set to `True` if using the long pole, else `False`
        :param simple_dynamics: if `True, use the simpler dynamics model from Quanser. If `False`, use a dynamics model
                                which includes friction
        :param wild_init: if `True` the init state space is increased drastically, e.g. the initial pendulum angle
                          can be in $[-\pi, +\pi]$. Only applicable to `QCartPoleSwingUpSim`.
        """
        Serializable._init(self, locals())

        self._simple_dynamics = simple_dynamics
        self._th_ddot = None  # internal memory necessary for computing the friction force
        self._th_ddot_tensor = None 
        self._obs_space = None
        self._long = long
        self._wild_init = wild_init
        self._x_buffer = 0.15  # [m]
        self._integrator = 'rk4'

        # Call SimPyEnv's constructor
        super().__init__(dt, max_steps, task_args)

        # Update the class-specific domain parameters
        self.domain_param = self.get_nominal_domain_param(long=long, mass=mass)

    def _create_spaces(self):
        l_rail = self.domain_param["rail_length"]
        max_obs = np.array([l_rail / 2.0, 1.0, 1.0, np.inf, np.inf])

        self._state_space = None
        self._obs_space = BoxSpace(-max_obs, max_obs, labels=["x", "sin_theta", "cos_theta", "x_dot", "theta_dot"])
        self._init_space = None
        self._act_space = BoxSpace(-MAX_ACT_QCP, MAX_ACT_QCP, labels=["V"])

    @abstractmethod
    def _create_task(self, task_args: dict) -> Task:
        raise NotImplementedError

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        # Set the initial angular acceleration to zero
        self._th_ddot = 0.0
        self._th_ddot_tensor = torch.zeros(1)
        obs = super().reset(init_state, domain_param)
        return self.state

    def observe(self, state) -> np.ndarray:
        return np.array([state[0], np.sin(state[1]), np.cos(state[1]), state[2], state[3]])

    @classmethod
    def get_nominal_domain_param(cls, long: bool = False, mass = None) -> dict:
        if long:
            m_pole = 0.23
            l_pole = 0.641 / 2
        else:
            m_pole = 0.127
            l_pole = 0.3365 / 2

        if mass is not None:
            m_pole = mass

        return dict(
            gravity_const=9.81,  # gravity constant [m/s**2]
            cart_mass=0.58,  # mass of the cart [kg]
            rail_length=0.814,  # length of the rail the cart is running on [m]
            motor_efficiency=0.9,  # motor efficiency [-], default 1.
            gear_efficiency=0.9,  # planetary gearbox efficiency [-], default 1.
            gear_ratio=3.71,  # planetary gearbox gear ratio [-]
            motor_inertia=3.9e-7,  # rotor inertia [kg*m**2]
            pinion_radius=6.35e-3,  # motor pinion radius [m]
            motor_resistance=2.6,  # motor armature resistance [Ohm]
            motor_back_emf=7.67e-3,  # motor torque constant [N*m/A] = back-EMF constant [V*s/rad]
            pole_damping=0.0024,  # viscous coefficient at the pole bearing [N*s]
            combined_damping=5.4,  # equivalent Viscous damping coefficient between cart and track [N*s/m]
            # B_p is an approximation, since the friction torque should actually depend on the normal force between
            # the cart and the pole. However, one could use one approx. equivalent mass for that force.
            pole_mass=m_pole,  # mass of the pole [kg]
            pole_length=l_pole,  # half pole length [m]
            cart_friction_coeff=0.02,  # Coulomb friction coefficient cart-rail [-]
            voltage_thold_neg=0,  # min. voltage required to move the servo in negative the direction [V]
            voltage_thold_pos=0,  # min. voltage required to move the servo in positive the direction [V]
        )

    def _calc_constants(self):
        l_pole = self.domain_param["pole_length"]
        m_pole = self.domain_param["pole_mass"]
        m_cart = self.domain_param["cart_mass"]
        eta_g = self.domain_param["gear_efficiency"]
        K_g = self.domain_param["gear_ratio"]
        J_m = self.domain_param["motor_inertia"]
        r_mp = self.domain_param["pinion_radius"]

        self.J_pole = l_pole ** 2 * m_pole / 3.0  # pole inertia [kg*m**2]
        self.J_eq = m_cart + (eta_g * K_g ** 2 * J_m) / r_mp ** 2  # equiv. inertia [kg]

    def _step_dynamics(self, u: np.ndarray):
        s_augmented = np.array([self.state[0], self.state[1], self.state[2], self.state[3], u[0]])
        if self._integrator=='rk4':
            next_state, th_ddot = rk4(self._dynamics, s_augmented, [0, self._dt], self._th_ddot)
        else:
            next_state, th_ddot = euler(self._dynamics, s_augmented, [0, self._dt], self._th_ddot)
        self._th_ddot = th_ddot
        self.state = np.asarray(next_state[:-1], dtype=np.float)

    def _dynamics(self, s_augmented: np.ndarray, dt, _th_ddot):
        gravity_const = self.domain_param["gravity_const"]
        l_p = self.domain_param["pole_length"]
        m_p = self.domain_param["pole_mass"]
        m_c = self.domain_param["cart_mass"]
        eta_m = self.domain_param["motor_efficiency"]
        eta_g = self.domain_param["gear_efficiency"]
        K_g = self.domain_param["gear_ratio"]
        R_m = self.domain_param["motor_resistance"]
        k_m = self.domain_param["motor_back_emf"]
        r_mp = self.domain_param["pinion_radius"]
        B_eq = self.domain_param["combined_damping"]
        B_p = self.domain_param["pole_damping"]
        mu_c = self.domain_param["cart_friction_coeff"]

        x, th, x_dot, th_dot, u = s_augmented
        sin_th = np.sin(th)
        cos_th = np.cos(th)
        m_tot = m_c + m_p

        # Apply a voltage dead zone, i.e. below a certain amplitude the system is will not move.
        # This is a very simple model of static friction.
        if (
            not self._simple_dynamics
            and self.domain_param["voltage_thold_neg"] <= u <= self.domain_param["voltage_thold_pos"]
        ):
            u = 0.0

        # Actuation force coming from the carts motor torque
        f_act = (eta_g * K_g * eta_m * k_m) / (R_m * r_mp) * (eta_m * float(u) - K_g * k_m * x_dot / r_mp)

        if self._simple_dynamics:
            f_tot = float(f_act)

        else:
            # Force normal to the rail causing the Coulomb friction
            f_normal = m_tot * gravity_const - m_p * l_p / 2 * (sin_th * _th_ddot + cos_th * th_dot ** 2)
            if f_normal < 0:
                # The normal force on the cart is negative, i.e. it is lifted up. This can be cause by a very high
                # angular momentum of the pole. Here we neglect this effect.
                f_c = 0.0
            else:
                f_c = mu_c * f_normal * np.sign(x_dot)
            f_tot = f_act - f_c

        M = np.array(
            [
                [m_p + self.J_eq, m_p * l_p * cos_th],
                [m_p * l_p * cos_th, self.J_pole + m_p * l_p ** 2],
            ]
        )
        rhs = np.array(
            [
                f_tot - B_eq * x_dot - m_p * l_p * sin_th * th_dot ** 2,
                -B_p * th_dot - m_p * l_p * gravity_const * sin_th,
            ]
        )
        # Compute acceleration from linear system of equations: M * x_ddot = rhs
        x_ddot, th_ddot = np.linalg.solve(M, rhs)

        # Integration step (symplectic Euler)
        th_dot = th_dot + float(th_ddot) * self._dt  # next velocity
        x_dot = x_dot + float(x_ddot) * self._dt
        # self.state[:2] += self.state[2:] * self._dt  # next position
        return np.array([x_dot, th_dot, x_ddot, th_ddot, u*0]), th_ddot


    def step_diff(self, obs, act, th_ddot=None):
        # Current reward depending on the state (before step) and the (unlimited) action
        remaining_steps = self._max_steps - (self._curr_step + 1) if self._max_steps is not pyrado.inf else 0

        # Apply actuator limits
        act_unclamped = act
        act = self.limit_act_clamp(act)
        curr_act = act  # just for the render function

        # Apply the action and simulate the resulting dynamics
        next_obs, rew, done, info = self._step_dynamics_diff(obs, act, act_unclamped, th_ddot)
        # self._curr_step += 1

        # Check if the task or the environment is done
        
        # if self._curr_step >= self._max_steps:
        #     done = True

        # if done:
        #     # Add final reward if done
        #     curr_rew += self._task.final_rew(self.state, remaining_steps) #0.0

        return next_obs, rew, done, info

    def step_diff_state(self, state, act, act_diff, th_ddot=None):
        # Current reward depending on the state (before step) and the (unlimited) action
        remaining_steps = self._max_steps - (self._curr_step + 1) if self._max_steps is not pyrado.inf else 0

        # Apply actuator limits
        act_unclamped = act + act_diff
        act_diff = self.limit_act_diff_clamp(act_diff)
        act = self.limit_act_clamp(act + act_diff)
        curr_act = act  # just for the render function

        # Apply the action and simulate the resulting dynamics
        next_state, rew, done, info = self._step_dynamics_diff(state, act, act_unclamped, th_ddot, state_based=True)
        # self._curr_step += 1

        # Check if the task or the environment is done
        
        # if self._curr_step >= self._max_steps:
        #     done = True

        # if done:
        #     # Add final reward if done
        #     curr_rew += self._task.final_rew(self.state, remaining_steps) #0.0

        return next_state, rew, done, info

    def limit_act_clamp(self, act):
        ## NOTE: only for 1D act spaces. Might be problematic if blindly copied to higher dimensional action spaces
        return torch.clamp(act, self.act_space.bound_lo.item(), self.act_space.bound_up.item())

    def limit_act_diff_clamp(self, act_diff):
        ## NOTE: only for 1D act spaces. Might be problematic if blindly copied to higher dimensional action spaces
        # return torch.clamp(act_diff, self.act_space.bound_lo.item()*0.5, self.act_space.bound_up.item()*0.5)
        return act_diff

    def is_done(self, state):
        done = (1-((self._state_space_bound_lo_tensor < state).float()*(self._state_space_bound_up_tensor > state).float()).prod(dim=-1)).bool()
        return done

    def rew_fn(self, state, act):
        # self.task._rew_fcn.Q, self.task._rew_fcn.R
        # Modulate the state error
        err_state = self.state_des - state
        theta = err_state[:, 1] + (-(err_state[:, 1]//(2*np.pi))*(2*np.pi) + (err_state[:,1]<0).float()*(2*np.pi)).detach()
        # torch.remainder(err_state[self.idcs], 2*pi)  # by default map to [-2pi, 2pi]
        # if (torch.any(theta>2*np.pi) or torch.any(theta<0)):
        #     ipdb.set_trace()

        # Look at the shortest path to the desired state i.e. desired angle
        mask = (theta > np.pi).float()
        theta = (1-mask)*theta + mask*(theta - 2*np.pi)
        # theta = 2 * np.pi - err_state[err_state > np.pi]  # e.g. 360 - (210) = 150
        # err_state[err_state < -np.pi] = -2 * np.pi - err_state[err_state < -np.pi]  # e.g. -360 - (-210) = -150
        # ipdb.set_trace()
        err_state = torch.stack([err_state[:, 0], theta, err_state[:,2], err_state[:,3]], dim=-1)
        # print(err_state)#[:, 0].shape, theta.shape, err_state[:,2].shape, err_state[:,3].shape)
        # ipdb.set_trace()
        Q = self.Q
        R = self.R.squeeze()
        if state.shape[0]>1:
            # ipdb.set_trace()
            Q = torch.cat([self.Q]*state.shape[0], dim=0)
        reward = torch.bmm(torch.bmm(err_state.unsqueeze(-2), Q), err_state.unsqueeze(-1)).squeeze()
        reward += (-act * R * (-act)).squeeze()
        reward = torch.exp(-reward)
        return reward

    def _step_dynamics_diff(self, obs, u, act_unclamped, _th_ddot=None, state_based=False, integrator='rk4'):
        if _th_ddot is not None:
            _th_ddot_tensor = _th_ddot
        else:
            _th_ddot_tensor = self._th_ddot_tensor

        if state_based:
            x, th, x_dot, th_dot = obs[:,0], obs[:,1], obs[:,2], obs[:,3]
            # cos_th = torch.cos(th)
            # sin_th = torch.sin(th)
        else:
            x, x_dot, th_dot = obs[:,0], obs[:,3], obs[:,4]
            cos_th = obs[:, 2]
            sin_th = obs[:, 1]
            mask_cos = (cos_th>0).float()
            mask_sin = (sin_th>0).float()
            # th = (sin_th + cos_th)/2#
            th = ((mask_sin*2-1)*torch.acos(cos_th) + (mask_cos*2-1)*torch.asin(sin_th) + np.pi * (2*mask_sin-1) * (1-mask_cos))/2
        u = u.squeeze(-1)
        state = torch.stack([x, th, x_dot, th_dot], dim=-1)
        rew = self.rew_fn(state, act_unclamped)
        s_augmented = torch.stack([x, th, x_dot, th_dot, u], dim=-1) #_th_ddot_tensor
        if self._integrator=='rk4':
            next_state, _th_ddot_tensor = rk4(self._dynamics_diff, s_augmented, [0, self._dt], _th_ddot_tensor)
        else:
            next_state, _th_ddot_tensor = euler(self._dynamics_diff, s_augmented, [0, self._dt], _th_ddot_tensor)
        # _th_ddot_tensor = next_state[:, -1]
        next_state = next_state[:, :-1]
        obs = torch.stack([next_state[:,0], torch.sin(next_state[:, 1]), torch.cos(next_state[:, 1]), next_state[:, 2], next_state[:, 3]], dim=-1)
        done = self.is_done(next_state)
        if _th_ddot is None:
            self._th_ddot_tensor = _th_ddot_tensor.detach().clone()
        if state_based:
            return next_state, rew, done, {'th_ddot': _th_ddot_tensor}
        else:
            return obs, rew, done, {'th_ddot': _th_ddot_tensor}

    def _dynamics_diff(self, s_augmented, dt, _th_ddot_tensor):
        gravity_const = self.domain_param["gravity_const"]
        l_p = self.domain_param["pole_length"]
        m_p = self.domain_param["pole_mass"]
        m_c = self.domain_param["cart_mass"]
        eta_m = self.domain_param["motor_efficiency"]
        eta_g = self.domain_param["gear_efficiency"]
        K_g = self.domain_param["gear_ratio"]
        R_m = self.domain_param["motor_resistance"]
        k_m = self.domain_param["motor_back_emf"]
        r_mp = self.domain_param["pinion_radius"]
        B_eq = self.domain_param["combined_damping"]
        B_p = self.domain_param["pole_damping"]
        mu_c = self.domain_param["cart_friction_coeff"]

        m_tot = m_c + m_p

        x, th, x_dot, th_dot, u = s_augmented[:,0], s_augmented[:,1], s_augmented[:,2], s_augmented[:,3], s_augmented[:, 4]#, s_augmented[:, 5]
        sin_th, cos_th = torch.sin(th), torch.cos(th)
        # Apply a voltage dead zone, i.e. below a certain amplitude the system is will not move.
        # This is a very simple model of static friction.
        # if (
        #     not self._simple_dynamics
        #     and self.domain_param["voltage_thold_neg"] <= u <= self.domain_param["voltage_thold_pos"]
        # ):
        #     u = 0.0
        mask = torch.ge(u,self.domain_param["voltage_thold_neg"]).float()*torch.le(u, self.domain_param["voltage_thold_pos"]).float()*float(not self._simple_dynamics)
        u = (1-mask)*u
        # Actuation force coming from the carts motor torque
        f_act = (eta_g * K_g * eta_m * k_m) / (R_m * r_mp) * (eta_m * u - K_g * k_m * x_dot / r_mp)

        if self._simple_dynamics:
            f_tot = f_act

        else:
            # Force normal to the rail causing the Coulomb friction
            f_normal = m_tot * gravity_const - m_p * l_p / 2 * (sin_th * _th_ddot_tensor + cos_th * th_dot ** 2)
            # if f_normal < 0:
            #     # The normal force on the cart is negative, i.e. it is lifted up. This can be cause by a very high
            #     # angular momentum of the pole. Here we neglect this effect.
            #     f_c = 0.0
            # else:
            #     f_c = mu_c * f_normal * torch.sign(x_dot)
            f_n_mask = (f_normal < 0).float()
            f_c = (1-f_n_mask) * (mu_c * f_normal * torch.sign(x_dot))
            f_tot = f_act - f_c

        M = torch.stack([
                            torch.stack([(m_p + self.J_eq)*torch.ones_like(cos_th), m_p * l_p * cos_th], dim=-1),
                            torch.stack([(m_p * l_p * cos_th), (self.J_pole + m_p * l_p ** 2)*torch.ones_like(cos_th)], dim=-1)
                        ],dim=-2)
        # if len(th_dot.shape)>1 and len(x_dot.shape)>1 and len(sin_th.shape)>1:
        #     ipdb.set_trace()
        try:
            rhs = torch.stack(
                            [
                                f_tot - B_eq * x_dot - m_p * l_p * sin_th * th_dot ** 2,
                                -B_p * th_dot - m_p * l_p * gravity_const * sin_th,
                            ],dim=-1).unsqueeze(-1)
        except:
            ipdb.set_trace()
        # Compute acceleration from linear system of equations: M * x_ddot = rhs
        inv, _ = torch.solve(rhs, M)
        inv = inv.squeeze(-1)
        x_ddot, th_ddot = inv[:, 0], inv[:, 1]

        # Integration step (symplectic Euler)
        vel = torch.stack([x_dot, th_dot], dim=-1) + inv*self._dt# np.array([x_ddot, self._th_ddot_tensor]) * self._dt  # next velocity
        pos = torch.stack([x, th], dim=-1) + vel * self._dt  # next position
        # derivs = torch.stack([x_dot, th_dot, inv[:,0], inv[:,1], u*0, (th_ddot-_th_ddot_tensor)/dt], dim=-1)
        derivs = torch.stack([vel[:,0], vel[:,1], inv[:,0], inv[:,1], u*0], dim=-1)
        return derivs, th_ddot

    def _init_anim(self):
        # Import PandaVis Class
        from pyrado.environments.pysim.pandavis import QCartPoleVis

        # Create instance of PandaVis
        self._visualization = QCartPoleVis(self, self._rendering)


class QCartPoleStabSim(QCartPoleSim, Serializable):
    """
    Environment in which a pole has to be stabilized in the upright position (inverted pendulum) by moving a cart on
    a rail. The pole can rotate around an axis perpendicular to direction in which the cart moves.
    """

    name: str = "qcp-st"

    def __init__(
        self,
        dt: float,
        max_steps: int = pyrado.inf,
        task_args: Optional[dict] = None,
        long: bool = True,
        simple_dynamics: bool = True,
    ):
        """
        Constructor

        :param dt: simulation step size [s]
        :param max_steps: maximum number of simulation steps
        :param task_args: arguments for the task construction
        :param long: set to `True` if using the long pole, else `False`
        :param simple_dynamics: if `True, use the simpler dynamics model from Quanser. If `False`, use a dynamics model
                                which includes friction
        """
        Serializable._init(self, locals())

        self.stab_thold = 15 / 180.0 * np.pi  # threshold angle for the stabilization task to be a failure [rad]
        self.max_init_th_offset = 8 / 180.0 * np.pi  # [rad]

        super().__init__(dt, max_steps, task_args, long, simple_dynamics, wild_init='False')

    def _create_spaces(self):
        super()._create_spaces()
        l_rail = self.domain_param["rail_length"]

        min_state = np.array(
            [-l_rail / 2.0 + self._x_buffer, np.pi - self.stab_thold, -l_rail, -2 * np.pi]
        )  # [m, rad, m/s, rad/s]
        max_state = np.array(
            [+l_rail / 2.0 - self._x_buffer, np.pi + self.stab_thold, +l_rail, +2 * np.pi]
        )  # [m, rad, m/s, rad/s]

        max_init_state = np.array(
            [+0.02, np.pi + self.max_init_th_offset, +0.02, +5 / 180 * np.pi]
        )  # [m, rad, m/s, rad/s]
        min_init_state = np.array(
            [-0.02, np.pi - self.max_init_th_offset, -0.02, -5 / 180 * np.pi]
        )  # [m, rad, m/s, rad/s]

        self._state_space = BoxSpace(min_state, max_state, labels=["x", "theta", "x_dot", "theta_dot"])
        self._init_space = BoxSpace(min_init_state, max_init_state, labels=["x", "theta", "x_dot", "theta_dot"])

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get("state_des", np.array([0.0, np.pi, 0.0, 0.0]))
        Q = task_args.get("Q", np.diag([5e-0, 1e1, 1e-2, 1e-2]))
        R = task_args.get("R", np.diag([1e-3]))

        return FinalRewTask(
            RadiallySymmDesStateTask(self.spec, state_des, QuadrErrRewFcn(Q, R), idcs=[1]),
            mode=FinalRewMode(state_dependent=True, time_dependent=True),
        )


class QCartPoleSwingUpSim(QCartPoleSim, Serializable):
    """
    Environment in which a pole has to be swung up and stabilized in the upright position (inverted pendulum) by
    moving a cart on a rail. The pole can rotate around an axis perpendicular to direction in which the cart moves.
    """

    name: str = "qcp-su"

    def __init__(
        self,
        dt: float,
        max_steps: int = pyrado.inf,
        task_args: Optional[dict] = None,
        long: bool = False,
        simple_dynamics: bool = False,
        wild_init: str = 'True',
        mass = None,
    ):
        r"""
        Constructor

        :param dt: simulation step size [s]
        :param max_steps: maximum number of simulation steps
        :param task_args: arguments for the task construction
        :param long: set to `True` if using the long pole, else `False`
        :param simple_dynamics: if `True`, use the simpler dynamics model from Quanser. If `False`, use a dynamics model
                                which includes friction
        :param wild_init: if `True` the init state space is increased drastically, e.g. the initial pendulum angle
                          can be in $[-\pi, +\pi]$
        """
        Serializable._init(self, locals())

        super().__init__(dt, max_steps, task_args, long, simple_dynamics, wild_init, mass=mass)

    def _create_spaces(self):
        super()._create_spaces()

        # Define the spaces
        l_rail = self.domain_param["rail_length"]
        max_state = np.array(
            [+l_rail / 2.0 - self._x_buffer, +4 * np.pi, 1 * l_rail, 20 * np.pi]
        )  # [m, rad, m/s, rad/s]
        min_state = np.array(
            [-l_rail / 2.0 + self._x_buffer, -4 * np.pi, -1 * l_rail, -20 * np.pi]
        )  # [m, rad, m/s, rad/s]
        if self._wild_init=='True':
            max_init_state = np.array([0.25, np.pi, 0.8, np.pi])  # [m, rad, m/s, rad/s]
        elif self._wild_init=='False':
            max_init_state = np.array([0.02, 2 / 180.0 * np.pi, 0.0, 1 / 180.0 * np.pi])  # [m, rad, m/s, rad/s]
        else:
            max_init_state = np.array([0.02, np.pi, 0.0, 1 / 180.0 * np.pi])  # [m, rad, m/s, rad/s]

        self._state_space = BoxSpace(min_state, max_state, labels=["x", "theta", "x_dot", "theta_dot"])
        self._init_space = BoxSpace(-max_init_state, max_init_state, labels=["x", "theta", "x_dot", "theta_dot"])
        # self._init_space = BoxSpace(-max_init_state, max_init_state, labels=["x", "theta", "x_dot", "theta_dot"])
        self._state_space_bound_lo_tensor = torch.Tensor(self._state_space.bound_lo).to(self._th_ddot_tensor)
        self._state_space_bound_up_tensor = torch.Tensor(self._state_space.bound_up).to(self._th_ddot_tensor)

    def random_states(self, bsz):
        states = np.random.rand(bsz, 4)*2 - 1
        states[:, 0] *= self.domain_param["rail_length"] / 2.0 - self._x_buffer
        states[:, 1] *= np.pi
        states[:, 2] *= self.domain_param["rail_length"]
        states[:, 3] *= 11
        return states
        

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get("state_des", np.array([0.0, np.pi, 0.0, 0.0]))

        Q = task_args.get("Q", np.diag([3e-1, 5e-1, 5e-3, 1e-3]))
        # Q = task_args.get("Q", np.diag([1e-1, 1.0, 5e-3, 1e-3]))
        R = task_args.get("R", np.diag([1e-3]))

        self.state_des = torch.Tensor(state_des).to(self._th_ddot_tensor).unsqueeze(0)
        self.Q = torch.Tensor(Q).to(self._th_ddot_tensor).unsqueeze(0)
        self.R = torch.Tensor(R).to(self._th_ddot_tensor).unsqueeze(0)
        self._state_space_bound_lo_tensor = self._state_space_bound_lo_tensor.to(self._th_ddot_tensor)
        self._state_space_bound_up_tensor = self._state_space_bound_up_tensor.to(self._th_ddot_tensor)
        return RadiallySymmDesStateTask(self.spec, state_des, ExpQuadrErrRewFcn(Q, R), idcs=[1])



def rk4(derivs, y0, t, th_ddot0):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.

    Args:
        derivs: the derivative of the system and has the signature ``dy = derivs(yi)``
        y0: initial state vector
        t: sample times
        args: additional arguments passed to the derivative function
        kwargs: additional keyword arguments passed to the derivative function

    Example 1 ::
        ## 2D system
        def derivs(x):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)

    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    This would then require re-adding the time variable to the signature of derivs.

    Returns:
        yout: Runge-Kutta approximation of the ODE
    """

    # try:
    #     Ny = len(y0)
    # except TypeError:
    #     yout = np.zeros((len(t),), np.float_)
    # else:
    #     yout = np.zeros((len(t), Ny), np.float_)

    # yout = torch.zeros((len(t),y0.shape[0], y0.shape[1])).to(y0)
    yout = []
    th_ddots = []
    yout.append(y0)
    th_ddots.append(th_ddot0)

    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]
        th_ddot0 = th_ddots[i]
        # y0.requires_grad_(True)
        k1, th_ddot1 = derivs(y0, dt, th_ddot0)
        # ipdb.set_trace()
        k2, th_ddot2 = derivs(y0 + dt2 * k1, dt, th_ddot1)
        k3, th_ddot3 = derivs(y0 + dt2 * k2, dt, th_ddot2)
        k4, th_ddot4 = derivs(y0 + dt * k3, dt, th_ddot3)
        yout.append(y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4))
        # th_ddots.append((th_ddot1 + 2*th_ddot2 + 2*th_ddot3 + th_ddot4)/6)
        th_ddots.append((th_ddot1 + th_ddot2 + th_ddot3 + th_ddot4)/4)
    # We only care about the final timestep and we cleave off action value which will be zero
    # ipdb.set_trace()
    return yout[-1], th_ddots[-1]

def euler(derivs, y0, t, th_ddot0):

    yout = []
    thddots = []
    yout.append(y0)
    thddots.append(th_ddot0)

    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        y0 = yout[i]
        th_ddot0 = th_ddots[i]
        k1, th_ddot = derivs(y0, dt, th_ddot0)
        yout.append(y0 + dt * k1)
        th_ddots.append(th_ddot)
    # We only care about the final timestep and we cleave off action value which will be zero
    # ipdb.set_trace()
    return yout[-1], th_ddots[-1]
