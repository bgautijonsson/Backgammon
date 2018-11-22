#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 09:39:56 2018

@author: bgautijonsson
"""

def actionj(board_copy, dice, player, i):
    from GroupJ import AgentGroupJ
    
    AgentJ = AgentGroupJ()
    
    possible_moves, possible_boards = AgentJ.legal_moves(board_copy, dice, player)
    
    return AgentJ.sample_action(possible_boards)