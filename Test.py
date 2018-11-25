#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:20:48 2018

@author: bgautijonsson
"""

from GroupJ import AgentGroupJ
agent = AgentGroupJ(read_file = False, learning_rate=3e-4, gamma = 1, entropy = 1e-1)
agent.SelfPlay(n_envs = 5, n_games = 10000, test_each = 5, test_games = 1)

#tensorboard --logdir=Tboard
#localholst:6006

#agent._s.run(agent._meanwinrate)