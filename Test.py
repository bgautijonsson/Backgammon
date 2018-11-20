#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:20:48 2018

@author: bgautijonsson
"""

from GroupJ import AgentGroupJ
agent = AgentGroupJ(read_file = False, learning_rate=1e-4, gamma = 0.99)
agent.SelfPlay(n_envs = 100, n_games = 10000, test_each = 10, test_games = 1)



#agent._s.run(agent._meanwinrate)