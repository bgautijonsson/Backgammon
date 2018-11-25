#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The intelligent agent
see flipped_agent for an example of how to flip the board in order to always
perceive the board as player 1
"""
import numpy as np
import Backgammon as B
import flipped_agent as FA
import tensorflow as tf
import os.path
import copy
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import l2_regularizer
import pubeval

class backgammon:
    def __init__(self):
        self.board = B.init_board()
            
    def reset(self):
        self.board = B.init_board()
        self.done = False
    
    def legal_moves(self, dice, player):
        moves, boards = B.legal_moves(board = self.board, dice = dice, player = player)
        if len(boards) == 0:
            return [], []
        boards = np.vstack(boards)
        return moves, boards
    
    def swap_player(self):
        self.board = FA.flip_board(board_copy=np.copy(self.board))
    
    # oppents random move
    def make_move(self, dice):
        moves, _ = self.legal_moves(dice, -1)
        if len(moves) == 0:
            return self.step([], -1)
        move = moves[np.random.randint(len(moves))]
        return self.step(move, -1)
    
    def step(self, move, player = 1):
        old_board = np.copy(self.board)
        if len(move) != 0:
            for m in move:
                self.board = B.update_board(board = self.board, move = m, player = player)
        reward = 0
        self.done = False
        if self.iswin():
            reward = player
            self.done = True
        return old_board, np.copy(self.board), reward, self.done
        
    def iswin(self):
        return B.game_over(self.board)
        
    def render(self):
        B.pretty_print(self.board)

def network(inputs):
    with tf.variable_scope('Shared', reuse=tf.AUTO_REUSE):
        net = tf.layers.dense(inputs, 32, activation=tf.nn.leaky_relu,
                              kernel_initializer=xavier_initializer(),
                              kernel_regularizer=l2_regularizer(0.01),
                              name="hidden_1")
        net = tf.layers.dense(net, 64, activation=tf.nn.leaky_relu,
                              kernel_initializer=xavier_initializer(),
                              kernel_regularizer=l2_regularizer(0.01),
                              name="hidden_2")
        net = tf.layers.dense(net, 32, activation=tf.nn.leaky_relu,
                              kernel_initializer=xavier_initializer(),
                              kernel_regularizer=l2_regularizer(0.01),
                              name="hidden_3")
        
    return net

def critic(inputs):
    with tf.variable_scope('Shared', reuse=tf.AUTO_REUSE):
        critic = tf.layers.dense(inputs, 16, activation=tf.nn.leaky_relu,
                              kernel_initializer=xavier_initializer(),
                              kernel_regularizer=l2_regularizer(0.01),
                              name="critic_hidden_1")
        critic = tf.layers.dense(critic, 1, name="critic_out")
        
    return critic

def actor(inputs):
    with tf.variable_scope('Shared', reuse=tf.AUTO_REUSE):
        actor = tf.layers.dense(inputs, 16, activation=tf.nn.leaky_relu,
                              kernel_initializer=xavier_initializer(),
                              kernel_regularizer=l2_regularizer(0.01),
                              name="actor_hidden_1")
        actor = tf.layers.dense(actor, 1, name="actor_out")
        
    return actor


class AgentGroupJ:
    
    def __init__(self, gamma = 0.99, learning_rate = 0.001, entropy = 0.1, 
                 read_file = True, save_path = "/AgentData/AC_Agent"):
        
        self._gamma = gamma
        self._iters = tf.Variable(0, dtype = tf.float32, trainable = False)
        self._path = save_path
        
        self._currstate = tf.placeholder("float32", (None, 29), name = "CurrentStates")
        self._possible_states = tf.placeholder("float32", (None, 29), name = "PossibleStates")
        self._afterstate = tf.placeholder("float32", (None, 29), name = "AfterStates")
        self._is_terminal = tf.placeholder("float32", (), name = "IsTerminal")
        self._reward = tf.placeholder("float32", (), name = "Rewards")
        self._action = tf.placeholder("float32", (None, ), name = "Action")
        
        # Network
        self._s = tf.Session()
        self._network = network
        self._actor = actor
        self._critic = critic

        # Predictions
        ## Critic
        self._current_state_value = tf.nn.tanh(self._critic(self._network(self._currstate)))
        self._afterstate_value = tf.nn.tanh(self._critic(self._network(self._afterstate))) * (1 - self._is_terminal)

        self._target_state_value = self._reward
        self._target_state_value += self._gamma * self._afterstate_value * (1 - self._is_terminal)

        self._advantage = self._target_state_value - self._current_state_value

        ## Actor
        self._actor_logits = self._actor(self._network(self._possible_states))
        self._actor_policy = tf.nn.softmax(self._actor_logits, axis = 0)
        self._actor_log_policy = tf.nn.log_softmax(self._actor_logits, axis = 0)
        self._actor_entropy = -tf.reduce_mean(self._actor_policy * self._actor_log_policy)
        
        # Losses
        self._critic_loss = tf.reduce_mean(tf.square((tf.stop_gradient(self._target_state_value) - self._current_state_value)))
        self._actor_loss = -tf.reduce_mean(tf.stop_gradient(self._advantage) * self._action * self._actor_log_policy)

        self._optimizer = tf.train.AdamOptimizer(learning_rate)
        self._update = self._optimizer.minimize(self._actor_loss + 0.7 * self._critic_loss - entropy * entropy * self._actor_entropy, 
                                                global_step = self._iters)
        
        
        self._winrate_lookbehind = tf.Variable(100, dtype = tf.float32, trainable = False)
        
        self._meanwinrate_random = tf.Variable(0, dtype = tf.float32, trainable = False)
        self._iswin_random = tf.placeholder(dtype = tf.float32, shape = (), name = "IsWin_random")
        self._winrate_random = self._meanwinrate_random + (self._iswin_random - self._meanwinrate_random) / self._winrate_lookbehind
        self._track_random = tf.summary.scalar('Win_rate_Random', self._winrate_random)
        
        self._meanwinrate_pubeval = tf.Variable(0, dtype = tf.float32, trainable = False)
        self._iswin_pubeval = tf.placeholder(dtype = tf.float32, shape = (), name = "IsWin_pubeval")
        self._winrate_pubeval = self._meanwinrate_pubeval + (self._iswin_pubeval - self._meanwinrate_pubeval) / self._winrate_lookbehind
        self._track_pubeval = tf.summary.scalar('Win_rate_Pubeval', self._winrate_pubeval)
        
        self._c_loss = tf.summary.scalar('Critic_error', self._critic_loss)
        self._a_loss = tf.summary.scalar('Actor_error', self._actor_loss)
        self._a_entr = tf.summary.scalar('Entropy', self._actor_entropy)
        
        self._summary_losses = tf.summary.merge([self._c_loss, self._a_loss, self._a_entr])
        self._summary_winrate = tf.summary.merge([self._track_random, self._track_pubeval])
        
        self._s.run(tf.global_variables_initializer())
        
        self._file_writer = tf.summary.FileWriter("./Tboard",
                                    tf.get_default_graph())
        
        self._saver = tf.train.Saver()
        
        if os.path.isfile(self._path + ".index") and read_file:
            self._saver = tf.train.import_meta_graph(self._path)
            self._saver.restore(self._s)
            
    def __delete__(self):
        self._s.close()    
        
    def sample_action(self, afterstates):
        probs = self._s.run(self._actor_policy, ({self._possible_states: afterstates})).flatten()
        
        action = np.random.choice(np.arange(len(probs)), p = probs)
        
        return action
        
    
    def update(self, currstate, possible_states, afterstate, reward, action, is_terminal):
        
        
        
        _, gstep, summary = self._s.run([self._update, self._iters, self._summary_losses], 
                    ({self._currstate: currstate,
                      self._possible_states: possible_states,
                      self._afterstate: afterstate, 
                      self._is_terminal: is_terminal,
                      self._reward: reward,
                      self._action: action}))
    
        self._file_writer.add_summary(summary, gstep)
        
        if (gstep % 1000) == 0:
            self.save_network()
        
    def get_cumulative_rewards(self, rewards):
        reward = rewards[-1]
        i = np.arange(len(rewards))[::-1]
        R = reward * (self._gamma ** i)
        return R
    
    def ExamplePolicy(self):
        _, st = B.legal_moves(B.init_board(), B.roll_dice(), 1)
        
        out = np.round(self._s.run(self._actor_policy, ({self._possible_states: st})) * 100)/100
        out = out.flatten()
        out.sort()
        return out[::-1]
    
    def __str__(self):
        return(str(self._network.summary()))
    
    def save_network(self):
        self._saver.save(self._s, "." + self._path)
        
    def legal_moves(self, board, dice, player):
        if player == -1:
            board = FA.flip_board(np.copy(board))
        moves, boards = B.legal_moves(board = board, dice = dice, player = 1)
        if len(boards) == 0:
            return [], []
        boards = np.vstack(boards)
        return moves, boards
    
    ### Function til að skila ###
    def Action(self, board, dice, player):
        
        possible_moves, possible_boards = self.legal_moves(board, dice, player)

        if len(possible_moves) == 0:
            return []

        return self.sample_action(possible_boards)
    
    def PlayRandomAgent(self, test_games = 1):
        wins = []

        for _ in range(test_games):

            env = backgammon()
            done = False

            while not done:
                dice = B.roll_dice()
                for __ in range(1 + int(dice[0] == dice[1])):

                    possible_moves, possible_boards = env.legal_moves(dice, 1)
                    n_actions = len(possible_moves)

                    if n_actions == 0:
                        break

                    action = self.sample_action(possible_boards)
                    old_board, new_board, reward, done = env.step(possible_moves[action])

                    if done:
                        break

                if not done:
                    dice = B.roll_dice()

                    for _ in range(1 + int(dice[0] == dice[1])):
                            old_board, new_board, reward, done = env.make_move(dice)
                            if done:
                                reward = 0
                                break

            wins.append(float(reward == 1))
        
        return reward
        
    def PlayOldSelf(self, old_self, test_games = 1):
        wins = []
    
        for _ in range(test_games):
    
            env = backgammon()
            done = False
    
            while not done:
                dice = B.roll_dice()
                for _ in range(1 + int(dice[0] == dice[1])):
    
                    possible_moves, possible_boards = env.legal_moves(dice, 1)
                    n_actions = len(possible_moves)
    
                    if n_actions == 0:
                        break
    
                    action = self.sample_action(possible_boards)
                    old_board, new_board, reward, done = env.step(possible_moves[action])
    
                    if done:
                        break
    
                if not done:
                    dice = B.roll_dice()
    
                    for _ in range(1 + int(dice[0] == dice[1])):
                            possible_moves, possible_boards = env.legal_moves(dice, 1)
                            n_actions = len(possible_moves)
            
                            if n_actions == 0:
                                break
            
                            action = old_self.sample_action(possible_boards)
                            old_board, new_board, reward, done = env.step(possible_moves[action])
                            if done:
                                reward = -1
                                break
    
            wins.append(float(reward == 1))
        
        return(np.mean(wins))
        
    def PlayPubEval(self, test_games = 1):
        wins = []
    
        for _ in range(test_games):
    
            env = backgammon()
            done = False
    
            while not done:
                dice = B.roll_dice()
                for _ in range(1 + int(dice[0] == dice[1])):
    
                    possible_moves, possible_boards = env.legal_moves(dice, 1)
                    n_actions = len(possible_moves)
    
                    if n_actions == 0:
                        break
    
                    action = self.sample_action(possible_boards)
                    old_board, new_board, reward, done = env.step(possible_moves[action])
    
                    if done:
                        break
    
                if not done:
                    dice = B.roll_dice()
    
                    for __ in range(1 + int(dice[0] == dice[1])):
                            action = pubeval.agent_pubeval(np.copy(env.board), dice, oplayer = -1)
                            old_board, new_board, reward, done = env.step(action, player = -1)
                            if done:
                                reward = -1
                                break
            wins.append(float(reward == 1))
        
        return(np.mean(wins))
    
    def SelfPlay(self, n_envs = 10, n_games = 1000, test_each = 100, test_games = 20, verbose = True):
        
        played_games = 0
        show = False
        
        envs = [backgammon() for i in range(n_envs)]
        
        CurrState_loser = [[] for i in range(n_envs)]
        AfterState_loser = [[] for i in range(n_envs)]
        PossibleStates_loser = [[] for i in range(n_envs)]
        Reward_loser = [[] for i in range(n_envs)]
        IsTerminal_loser = [[] for i in range(n_envs)]
        Action_loser = [[] for i in range(n_envs)]
        
        while played_games < n_games:
            for i in range(n_envs):
                dice = B.roll_dice()
                for _ in range(1 + int(dice[0] == dice[1])):

                    possible_moves, possible_boards = envs[i].legal_moves(dice, 1)
                    n_actions = len(possible_moves)

                    if n_actions == 0:
                        break

            
                    action = self.sample_action(possible_boards)
                    old_board, new_board, reward, done = envs[i].step(possible_moves[action], player = 1)
                    
                    CurrState = old_board.reshape(1, 29)
                    AfterState = new_board.reshape(1, 29)
                    PossibleStates = possible_boards
                    Reward = reward
                    IsTerminal = done
                    Action = np.zeros(n_actions)
                    Action[action] = 1
                    
                    self.update(currstate = CurrState, 
                                afterstate = AfterState, 
                                possible_states = PossibleStates,
                                reward = Reward,
                                action = Action,
                                is_terminal = IsTerminal)
                    if not done:
                        CurrState_loser[i] = old_board.reshape(1, 29)
                        AfterState_loser[i] = new_board.reshape(1, 29)
                        PossibleStates_loser[i] = possible_boards
                        Reward_loser[i] = -1
                        IsTerminal_loser[i] = done
                        Action_loser[i] = np.zeros(n_actions)
                        Action_loser[i][action] = 1
                        
                    else:
                        self.update(currstate = CurrState_loser[i], 
                                afterstate = AfterState_loser[i], 
                                possible_states = PossibleStates_loser[i],
                                reward = Reward_loser[i],
                                action = Action_loser[i],
                                is_terminal = IsTerminal_loser[i])
    
                        envs[i] = backgammon()

                        played_games += 1
                        show = True
                        break
                    
                    envs[i].swap_player()


                if (played_games + 1) % test_each == 0 and verbose and show:
                    show = False
                    outcome_random = float(self.PlayRandomAgent(test_games = test_games))
                    outcome_pubeval = float(self.PlayPubEval(test_games = 1))
                    
                    winrate_random, winrate_pubeval = self._s.run(self._winrate_random,
                                                                  self._winrate_pubeval, 
                                                                  ({self._iswin_random: outcome_random,
                                                                    self._iswin_pubeval: outcome_pubeval}))
    
                    self._s.run([tf.assign(self._meanwinrate_random, winrate_random),
                                 tf.assign(self._meanwinrate_pubeval, winrate_pubeval)])
    
                    summary, gstep = self._s.run([self._summary_winrate, self._iters],
                                                 feed_dict = ({self._iswin_random: outcome_random, 
                                                               self._iswin_pubeval: outcome_pubeval}))
                    self._file_writer.add_summary(summary, gstep)
                    
                    
