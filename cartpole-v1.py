#!/usr/bin/python3

import numpy as np
import tensorflow as tf
import gym
from tensorflow import keras

class PNetwork():
	def __init__(self, lr, out_dim, gamma, actor, input_dim = 4, activation_fn=None):
		self.l1 = keras.layers.Dense(30, activation="relu", input_shape=[None, input_dim])
		self.l2 = keras.layers.Dense(30, activation="relu")
		self.out = keras.layers.Dense(out_dim, activation=activation_fn)

		self.train_op = keras.optimizers.Adam(lr=lr)
		
		if actor:
			self.loss = self.actor_loss
		else:
			self.loss = self.critic_loss

		self.gamma = gamma
		self.out_dim = out_dim
		self.batch_size = 1

	def run(self, input_data):
		l1 = self.l1(input_data)
		l2 = self.l2(l1)
		out = self.out(l2)

		return out

	def actor_loss(self, state, A_t, action):
		l1 = self.l1(state)
		l2 = self.l2(l1)
		out_ = self.out(l2)

		out_ = tf.keras.backend.clip(out_, 1e-8, 1-1e-8)
		out_l = tf.math.log(out_) * action
		loss_v = -tf.tensordot(A_t, out_l, axes=1)

		return tf.reduce_mean(loss_v, axis=0)

	def critic_loss(self, state, Q_t, action):
		l1 = self.l1(state)
		l2 = self.l2(l1)
		v_fn = self.out(l2)

		return tf.reduce_mean(tf.square(np.reshape(Q_t, (len(Q_t), 1)) - v_fn))
	
	def learn(self, state, AQ_t, action):
		with tf.GradientTape() as tape:
			l = self.loss(state, AQ_t, action)
			g = tape.gradient(l, [self.l1.variables[0], self.l1.variables[1], self.l2.variables[0], self.l2.variables[1],
								self.out.variables[0], self.out.variables[1]])	
			self.train_op.apply_gradients(zip(g, [self.l1.variables[0], self.l1.variables[1], self.l2.variables[0], 
											self.l2.variables[1], self.out.variables[0], self.out.variables[1]]))

class Agent():
	def __init__(self, lr, gamma, epsilon, mem_size, input_dim = 4):
		self.lr = lr
		self.epsilon = epsilon
		self.mem_size = mem_size
		self.input_dim = input_dim
		self.mem_cntr = 0
		self.gamma = gamma

		self.actor = PNetwork(self.lr, 2, self.gamma, True, activation_fn="softmax")
		self.critic = PNetwork(self.lr, 1, self.gamma, False)

		self.state_memory = np.zeros((0,4))
		self.next_state_memory = np.zeros((0,4))
		self.reward_memory = np.zeros((0,1))
		self.action_memory = np.zeros((0,2))
		self.done_memory = np.zeros((0,1))
		self.value_fn_memory = np.zeros((0,1))
	
	def select_action(self, state):
		if np.random.random() < self.epsilon:
			action = np.random.choice([0,1])
		else:
			actions_prob_dist = self.actor.run(state)
			actions_prob_dist = tf.reshape(actions_prob_dist, 2)
			action = np.random.choice([0,1], p=actions_prob_dist)
		self.epsilon *= 0.9

		return action
	
	def store_experiences(self, state, reward, action, next_state, done):		
		action_encoded = np.zeros((1,2))
		action_encoded[0][action] = 1

		self.state_memory = np.append(self.state_memory, np.reshape(state, (1,4)), axis=0)
		self.next_state_memory = np.append(self.next_state_memory, np.reshape(next_state, (1,4)), axis=0) 
		self.action_memory = np.append(self.action_memory, action_encoded, axis=0)
		self.reward_memory = np.append(self.reward_memory, np.reshape(reward, (1,1)), axis=0)
		self.done_memory = np.append(self.done_memory, np.reshape(done, (1,1)), axis=0)
	
		value_fn = self.critic.run(state)
		self.value_fn_memory = np.append(self.value_fn_memory, np.reshape(value_fn, (1,1)), axis=0)

	def learn(self):
		A_t = np.zeros(len(self.state_memory))
		Q_t = np.zeros(len(self.state_memory))

		for i in range(len(self.state_memory) - 1):
				Q_t[i] = self.reward_memory[i] + self.gamma * self.value_fn_memory[i+1]
		
		Q_t[-1] = self.reward_memory[-1]
		A_t = Q_t - self.value_fn_memory

		self.actor.learn(self.state_memory, A_t, self.action_memory)
		self.critic.learn(self.state_memory, Q_t, self.action_memory)

		self.state_memory = np.zeros((0,4))
		self.next_state_memory = np.zeros((0,4))
		self.reward_memory = np.zeros((0,1))
		self.action_memory = np.zeros((0,2))
		self.done_memory = np.zeros((0,1))
		self.value_fn_memory = np.zeros((0,1))
	
	def save_model(self):
		pass #TODO

def average(lst):
	return np.sum(lst) / np.count_nonzero(lst)

def main():
	numberOfGames = 5000
	prev_state = np.zeros(4)

	# create the environment
	environment = gym.make("CartPole-v1")
	agent = Agent(0.001, 0.99, 1.0, 1)
	episodes = [0 for i in range(50)]

	print("Starting the game")
	for i in range(numberOfGames):
		# start by getting the first observation
		state = np.reshape(environment.reset(), (1,4))
		done = False
		steps = 0

		while not done:
			prev_state = state
			action = agent.select_action(state)
			state, reward, done, info = environment.step(action)
			state = np.reshape(state, (1,4))

			agent.store_experiences(prev_state, reward, action, state, int(done))
	
		episodes = [episodes[i+1] for i in range(49)]
		episodes.append(steps)
		
		print("Game ", i, " length: ", average(episodes))	
	
		agent.learn()

	agent.save_model()

if __name__ == "__main__":
	tf.keras.backend.set_floatx('float64')
	main()
