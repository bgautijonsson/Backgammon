#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:20:48 2018

@author: bgautijonsson
"""

from GroupJ import AgentGroupJ

agent = AgentGroupJ(read_file = False)

agent.SelfPlay(n_envs = 100, n_games = 1000, test_each = 100, test_games = 20)