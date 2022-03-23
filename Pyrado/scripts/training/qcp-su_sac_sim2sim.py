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

"""
Train an agent to solve the Quanser CartPole swing-up task using Proximal Policy Optimization.
"""
import torch
import torch as to
from torch.optim import lr_scheduler

import pyrado
from pyrado.algorithms.step_based.sac import SAC
from pyrado.environment_wrappers.action_normalization import ActNormWrapper, ObsActCatWrapper
from pyrado.environments.pysim.quanser_cartpole import QCartPoleSwingUpSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.feed_back.fnn import FNNPolicy
from pyrado.policies.feed_back.two_headed_fnn import TwoHeadedFNNPolicy
from pyrado.spaces import ValueFunctionSpace
from pyrado.spaces.box import BoxSpace
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import EnvSpec
import ipdb
from IPython import embed

if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    seed_str = f"seed-{args.seed}" if args.seed is not None else None

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QCartPoleSwingUpSim.name + "_long_maxa6_T300_sim2sim", f"{SAC.name}_rl2init_qcp-su_long_maxa6_T1200_sac_dt01_rlvgiact_rk4_parrol_MAXA6_fixed4_nosmooth_hidden128_2022-03-21_17-56-56_blue", seed_str)

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environment
    env_hparams = dict(
        dt=1 / 100.0,
        max_steps=1200,
        long=True,
        simple_dynamics=False,
        wild_init='False',#'mod',#'False',
        mass=0.48,
    )
    env_sim_hparams = dict(
        dt=1 / 100.0,
        max_steps=1200,
        long=True,
        simple_dynamics=False,
        wild_init='False',#'mod',#'False',
        # mass=0.48,
        # mass=0.38,
    )
    hidden_size = 128
    env = QCartPoleSwingUpSim(**env_hparams)
    env_sim = QCartPoleSwingUpSim(**env_sim_hparams)
    # env = ObsVelFiltWrapper(env, idcs_pos=["theta", "alpha"], idcs_vel=["theta_dot", "alpha_dot"])
    env = ActNormWrapper(env)
    env = ObsActCatWrapper(env)
    env_sim = ActNormWrapper(env_sim)
    env_sim = ObsActCatWrapper(env_sim)

    # Policy
    policy_hparam = dict(shared_hidden_sizes=[hidden_size, hidden_size], shared_hidden_nonlin=to.tanh)
    policy = TwoHeadedFNNPolicy(spec=env.spec, **policy_hparam)
    # obs = torch.tensor(env.reset()).unsqueeze(0).float().requires_grad_(True)
    # act = (torch.ones(1,1)*0.1).requires_grad_(True)
    # nobs, rew, done, info = env.step_diff(obs, act)
    # embed()

    # Critic
    qfnc_param = dict(hidden_sizes=[hidden_size, hidden_size], hidden_nonlin=to.tanh)  # FNN
    # ipdb.set_trace()
    combined_space = BoxSpace.cat([env.obs_space, env.act_space])
    q1 = FNNPolicy(spec=EnvSpec(combined_space, ValueFunctionSpace), **qfnc_param)
    q2 = FNNPolicy(spec=EnvSpec(combined_space, ValueFunctionSpace), **qfnc_param)

    # Algorithm
    algo_hparam = dict(
        gamma=0.9844224855479998,
        memory_size=1000000,
        max_iter=300,
        num_updates_per_step=1000,
        # num_updates_ratio_transfer=1,
        tau=0.99,
        ent_coeff_init=0.3,
        learn_ent_coeff=True,
        target_update_intvl=1,
        num_init_rollouts=2,
        num_init_memory_steps=120 * env.max_steps,
        standardize_rew=False,
        min_steps=30 * env.max_steps,
        batch_size=512,
        lr=5e-4,
        max_grad_norm=1.5,
        num_workers=1,
        eval_intvl=1,
        env_sim=env_sim,
    )
    algo = SAC(ex_dir, env, policy, q1, q2, **algo_hparam)
    algo.load_snapshot(args)
    # algo.plot_vfunc_state_space()

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparams, seed=args.seed),
        dict(policy=policy_hparam),
        dict(qfcn=qfnc_param),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train_sim2sim(snapshot_mode="latest_and_best", seed=args.seed)
