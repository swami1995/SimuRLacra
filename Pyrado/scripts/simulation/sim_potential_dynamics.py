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
Script to simulate 1-dimensional dynamical systems used in Activation Dynamic Networks (ADN) for different hyper-parameters
"""
import numpy as np
import torch as to
import os.path as osp
from matplotlib import pyplot as plt

import pyrado
from pyrado.logger.experiment import ask_for_experiment
from pyrado.policies.adn import ADNPolicy
from pyrado.policies.neural_fields import NFPolicy
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment()

    # Load the environment and the policy
    env, policy, _ = load_experiment(ex_dir, args)
    if not isinstance(policy, (ADNPolicy, NFPolicy)):
        raise pyrado.TypeErr(given=policy, expected_type=[ADNPolicy, NFPolicy])

    # Define the parameters for evaluation
    num_steps, dt_eval = 1000, env.dt/2
    policy._dt = dt_eval
    p_init_min, p_init_max, num_p_init = -6., 6., 11
    print_cbt(f'Evaluating an {policy.name} for {num_steps} steps ad {1/dt_eval} Hz with initial potentials ranging '
              f'from {p_init_min} to {p_init_max}.', 'c')

    time = to.linspace(0., num_steps*dt_eval, num_steps)  # endpoint included
    p_init = to.linspace(p_init_min, p_init_max, num_p_init)  # endpoint included
    num_p = len(policy.potentials)

    # For mode = standalone they are currently all the same because all neuron potential-based obey the same dynamics.
    # However, this does not necessarily have to be that way. Thus we plot the same way as for mode = policy.
    for idx_p in range(num_p):
        # Create the figure
        fig, ax = plt.subplots(1, figsize=(12, 10), subplot_kw={'projection': '3d'})
        fig.canvas.set_window_title(f'Potential dynamics for the {idx_p}-th dimension for initial values')
        ax.set_xlabel('$t$ [s]')
        ax.set_ylabel('$p_0$')
        ax.set_zlabel('$p(t)$')

        final_values = to.zeros(num_p_init)

        for idx_p_init, p_0 in enumerate(p_init):
            p = to.zeros(num_steps, num_p)
            s = to.zeros(num_steps, num_p)

            potentials_init = p_0*to.ones_like(policy.potentials)
            if isinstance(policy, ADNPolicy):
                hidden = to.cat([to.zeros(policy.env_spec.act_space.shape), potentials_init], dim=-1)  # pack hidden
            elif isinstance(policy, NFPolicy):
                hidden = potentials_init

            for i in range(num_steps):
                # Use the loaded ADNPolicy's forward method and pass zero-observations
                _, hidden = policy(to.zeros(policy.env_spec.obs_space.shape), hidden)  # previous action is in hidden
                p[i, :] = policy.potentials.clone()

            # Extract final value
            final_values[idx_p_init] = p[-1, idx_p]

            # Plot
            plt.plot(time.numpy(), p_0.repeat(num_steps).numpy(), p[:, idx_p].detach().numpy())
        plt.title(f'Final values for the different initial potentials\n'
                  f'{final_values.detach().numpy().round(3)}', y=1.05)

    # Save
    if args.save_figures:
        for fmt in ['pdf', 'pgf']:
            fig.savefig(osp.join(ex_dir, f'potdyn-kappa.{fmt}'), dpi=500)

    plt.show()
