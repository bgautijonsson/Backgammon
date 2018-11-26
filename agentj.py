#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 09:39:56 2018

@author: bgautijonsson
"""

import GroupJ
import flipped_agent as FA
import Backgammon as B
import numpy as np
import tensorflow as tf
tf.reset_default_graph()
AgentJ = GroupJ.AgentGroupJ()

def action(board_copy, dice, player, i):
    
    if player == -1:
        board_copy = FA.flip_board(np.copy(board_copy))
    possible_moves, possible_boards = B.legal_moves(board_copy, dice, 1)
    
    if len(possible_moves) == 0:
        return []
    
    action = AgentJ.sample_action(np.vstack(possible_boards))
    move = possible_moves[action]
    if player == -1:
        move = FA.flip_move(move)
    return move


