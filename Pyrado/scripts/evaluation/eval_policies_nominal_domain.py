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
Script to evaluate multiple policies in one environment using the nominal domain parameters.
"""
import os.path as osp

import pandas as pd
from prettyprinter import pprint
from abc import ABC, abstractmethod

import pyrado
from pyrado.domain_randomization.utils import wrap_like_other_env
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environments.pysim.quanser_cartpole import QCartPoleStabSim, QCartPoleSwingUpSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.sampling.parallel_evaluation import eval_nominal_domain
from pyrado.sampling.sampler_pool import SamplerPool
from pyrado.utils.argparser import get_argparser
from pyrado.utils.checks import check_all_lengths_equal
from pyrado.utils.data_types import dict_arraylike_to_float
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt
import numpy as np
from torch import nn
import torch
import ipdb

if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    if args.max_steps == pyrado.inf:
        args.max_steps = 2500
        print_cbt(f"Set maximum number of time steps to {args.max_steps}", "y")
    # ipdb.set_trace()
    pyrado.EXP_DIR = pyrado.EVAL_DIR = pyrado.TEMP_DIR
    if args.env_name == QBallBalancerSim.name:
        # Create the environment for evaluating
        env = QBallBalancerSim(dt=args.dt, max_steps=args.max_steps)

        # Get the experiments' directories to load from
        prefixes = [
            osp.join(pyrado.EXP_DIR, "qcp-su", "ppo_fnn"),
        ]
        ex_names = [
            "2022-02-18_00-59-42",
        ]
        ex_labels = [
            "eval_qcp_ppo_15_rand",
        ]

    elif args.env_name in [QCartPoleStabSim.name, QCartPoleSwingUpSim.name]:
        # Create the environment for evaluating
        if args.env_name == QCartPoleSwingUpSim.name:
            env = QCartPoleSwingUpSim(dt=args.dt, max_steps=args.max_steps)
        else:
            env = QCartPoleStabSim(dt=args.dt, max_steps=args.max_steps)

        # Get the experiments' directories to load from
        prefixes = [
            osp.join(pyrado.EXP_DIR, "qcp-su", "sac"),
        ]
        ex_names = [
            #"2022-02-20_18-46-44",
            "2022-02-20_21-15-30",
        ]
        ex_labels = [
            "eval_qcp_sac_15_rand",
        ]

    else:
        raise pyrado.ValueErr(
            given=args.env_name,
            eq_constraint=f"{QBallBalancerSim.name}, {QCartPoleStabSim.name}," f"or {QCartPoleSwingUpSim.name}",
        )

    if not check_all_lengths_equal([prefixes, ex_names, ex_labels]):
        raise pyrado.ShapeErr(
            msg=f"The lengths of prefixes, ex_names, and ex_labels must be equal, "
            f"but they are {len(prefixes)}, {len(ex_names)}, and {len(ex_labels)}!"
        )

    # Loading the policies
    ex_dirs = [osp.join(p, e) for p, e in zip(prefixes, ex_names)]
    env_sim_list = []
    policy_list = []
    for ex_dir in ex_dirs:
        env_, policy, _ = load_experiment(ex_dir, args)
        policy_list.append(policy)
        env_sim_list.append(env_)
    #ipdb.set_trace()
    # Fix initial state (set to None if it should not be fixed)
    # init_state_list = [(j, None) for j in range(args.num_rollouts_per_config)]
    init_state_list = [(j, np.array([0.0, 0.0, 0.0, 0.0])) for j in range(args.num_rollouts_per_config)]
    class Policy_periodic(nn.Module):
        def __init__(self, T=1, dt=0.01, itype='square'):
            super().__init__()
            self.T = T
            self.dt = dt
            self.type = itype
            self.count = 0
            self.output = -2/8
            self.total_count = 0
            self.ff = nn.Linear(1, 1)
        def init_param(self, init_values: torch.Tensor = None, **kwargs):
            self.param_values = init_values
        def forward(self, obs):
            self.count +=1
            self.total_count +=1
            if self.count > self.T/(self.dt*2):
                self.output*= -1
                self.count = 0
            return self.output*torch.ones_like(obs)[:1]
    policy_test = Policy_periodic()
    #from IPython import embed; embed()
    # Crate empty data frame
    df = pd.DataFrame(columns=["policy", "ret", "len"])
    from pyrado.sampling.rollout import rollout
    # Evaluate all policies
    for i, (env_sim, policy) in enumerate(zip(env_sim_list, policy_list)):
        # Create a new sampler pool for every policy to synchronize the random seeds i.e. init states
        pool = SamplerPool(args.num_workers)

        # Seed the sampler
        if args.seed is not None:
            pool.set_seed(args.seed)
            print_cbt(f"Set the random number generators' seed to {args.seed}.", "w")
        else:
            print_cbt("No seed was set", "y")

        # Add the same wrappers as during training
        env = wrap_like_other_env(env, env_sim)

        # Sample rollouts
        # ros = [rollout(env, policy_test, eval=True, seed=0, sub_seed=0, sub_sub_seed=0, reset_kwargs=dict(init_state=init_state_list[0][1]))]
        ros = eval_nominal_domain(pool, env, policy, init_state_list, args.seed, i)

        # Compute results metrics
        rets = [ro.undiscounted_return() for ro in ros]
        lengths = [float(ro.length) for ro in ros]  # int values are not numeric in pandas
        df = df.append(pd.DataFrame(dict(policy=ex_labels[i], ret=rets, len=lengths)), ignore_index=True)
    import matplotlib.pyplot as plt
    ipdb.set_trace()
    plt.plot(np.arange(0, (lengths[0]+1)*0.01, 0.01), ros[0].observations[:, 0]*1000)
    plt.plot(np.arange(0, (lengths[0]+1)*0.01, 0.01), ros[0].observations[:, 3])
    metrics = dict(
        avg_len=df.groupby("policy").mean()["len"].to_dict(),
        avg_ret=df.groupby("policy").mean()["ret"].to_dict(),
        median_ret=df.groupby("policy").median()["ret"].to_dict(),
        min_ret=df.groupby("policy").min()["ret"].to_dict(),
        max_ret=df.groupby("policy").max()["ret"].to_dict(),
        std_ret=df.groupby("policy").std()["ret"].to_dict(),
    )
    pprint(metrics, indent=4)

    # Create sub-folder and save
    save_dir = setup_experiment("multiple_policies", args.env_name, "nominal", base_dir=pyrado.EVAL_DIR)

    save_dicts_to_yaml(
        {"ex_dirs": ex_dirs},
        {"num_rpp": args.num_rollouts_per_config, "seed": args.seed},
        {"metrics": dict_arraylike_to_float(metrics)},
        save_dir=save_dir,
        file_name="summary",
    )
    df.to_pickle(osp.join(save_dir, "df_nom_mp.pkl"))
