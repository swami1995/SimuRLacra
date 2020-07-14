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

import torch as to

from pyrado.domain_randomization.domain_parameter import DomainParam, BernoulliDomainParam, UniformDomainParam, \
    NormalDomainParam, MultivariateNormalDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer


DomainParam(name='a', mean=1)

BernoulliDomainParam(name='b', val_0=2, val_1=5, prob_1=0.8)

DomainRandomizer(
    NormalDomainParam(name='mass', mean=1.2, std=0.1, clip_lo=10, clip_up=100)
)

DomainRandomizer(
    NormalDomainParam(name='mass', mean=1.2, std=0.1, clip_lo=10, clip_up=100),
    UniformDomainParam(name='special', mean=0, halfspan=42, clip_lo=-7.4, roundint=True),
    NormalDomainParam(name='length', mean=4, std=0.6, clip_up=50.1),
    UniformDomainParam(name='time_delay', mean=13, halfspan=6, clip_up=17, roundint=True),
    MultivariateNormalDomainParam(name='multidim', mean=10*to.ones((2,)), cov=2*to.eye(2), clip_up=11)
)
