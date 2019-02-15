import tensorflow as tf
import numpy as np
import copy

class PolicyWithValue:
    def __init__(self, observation_space, action_space, name):
        self.observation_space = observation_space
        self.action_space = action_space

        with tf.variable_scope(name):
            self.observation = tf.placeholder(dtype=tf.float32, shape=[None] + list(observation_space), name='observation')
            
            #with tf.variable_scope('common'):
            #    layer_1 = tf.layers.dense(inputs=self.observation, units=32, activation=tf.tanh)
            #    layer_2 = tf.layers.dense(inputs=layer_1, units=32, activation=tf.tanh)

            with tf.variable_scope('policy_net'):
                layer_1 = tf.layers.dense(inputs=self.observation, units=32, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=32, activation=tf.tanh)
                self.action_probs = tf.layers.dense(inputs=layer_2, units=action_space, activation=tf.nn.softmax)

            with tf.variable_scope('value_net'):
                layer_1 = tf.layers.dense(inputs=self.observation, units=32, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=32, activation=tf.tanh)
                self.value_pred = tf.layers.dense(inputs=layer_2, units=1, activation=None)
            
            # for stochastic
            self.action_stochastic = tf.multinomial(tf.log(self.action_probs), num_samples=1)
            self.action_stochastic = tf.reshape(self.action_stochastic, shape=[-1])
            
            # for deterministic
            self.action_deterministic = tf.argmax(self.action_probs, axis=1)

            self.scope = tf.get_variable_scope().name
    
    def _get_action(self, sess, observation, stochastic=True):
        if stochastic:
            return sess.run([self.action_stochastic, self.value_pred], 
                            feed_dict={self.observation: observation})
        else:
            return sess.run([self.action_deterministic, self.value_pred], 
                            feed_dict={self.observation: observation})
    
    def _get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class PPOAgent:
    def __init__(self, policy, old_policy, horizon, learning_rate, epochs, 
                 batch_size, gamma, lmbd, clip_value, value_coeff, entropy_coeff):
        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter('./log/train', self.sess.graph)

        self.policy = policy
        self.old_policy = old_policy

        self.horizon = horizon
        self.batch_size = batch_size
        self.epochs = epochs

        self.gamma = gamma
        self.lmbd = lmbd

        self.record_observations = []
        self.record_actions = []
        self.record_value_preds = []
        self.record_rewards = []
        
        policy_variables = self.policy._get_trainable_variables()
        old_policy_variables = self.old_policy._get_trainable_variables()
        
        # assignment operation to update old_policy with policy
        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for variable, old_variable in zip(policy_variables, old_policy_variables):
                self.assign_ops.append(tf.assign(old_variable, variable))

        # inputs for train operation
        with tf.variable_scope('train_input'):
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
            self.value_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='value_preds_next')
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')

        action_probs = self.policy.action_probs
        old_action_probs = self.old_policy.action_probs
        
        value_pred = self.policy.value_pred

        # probability of actions chosen with the policy
        action_probs = action_probs * tf.one_hot(indices=self.actions, depth=action_probs.shape[1])
        action_probs = tf.reduce_sum(action_probs, axis=1)

        # probabilities of actions which agent took with old policy
        old_action_probs = old_action_probs * tf.one_hot(indices=self.actions, depth=old_action_probs.shape[1])
        old_action_probs = tf.reduce_sum(old_action_probs, axis=1)

        # clipped surrogate objective (7)
        # TODO adaptive KL penalty coefficient can be added (8)
        with tf.variable_scope('loss/clip'):
            ratios = tf.exp(tf.log(action_probs) - tf.log(old_action_probs))
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value)
            loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
            loss_clip = tf.reduce_mean(loss_clip)
            tf.summary.scalar('loss_clip', loss_clip)

        # squared difference between value (9)
        with tf.variable_scope('loss/value'):
            loss_value = tf.squared_difference(self.rewards + self.gamma * self.value_preds_next, value_pred)
            loss_value = tf.reduce_mean(loss_value)
            tf.summary.scalar('loss_value', loss_value)

        # entropy bonus (9)
        with tf.variable_scope('loss/entropy'):
            loss_entropy = -tf.reduce_sum(self.policy.action_probs * 
                                          tf.log(tf.clip_by_value(self.policy.action_probs, 1e-10, 1.0)), axis=1)
            loss_entropy = tf.reduce_mean(loss_entropy, axis=0)
            tf.summary.scalar('loss_entropy', loss_entropy)
        
        # loss (9)
        with tf.variable_scope('loss'):
            # c_1 = value_coeff, c_2 = entropy_coeff
            # loss = (clipped loss) - c_1 * (value loss) + c_2 * (entropy bonus)
            loss = loss_clip - value_coeff * loss_value + entropy_coeff * loss_entropy
            loss = -loss # gradient ascent
            tf.summary.scalar('loss', loss)

        self.merged = tf.summary.merge_all()
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5)
        self.train_op = optimizer.minimize(loss, var_list=policy_variables)
        
        self.sess.run(tf.global_variables_initializer())

    def action(self, observation, stochastic=True):
        observation = np.stack([observation]).astype(dtype=np.float32)
        action, value_pred = self.policy._get_action(sess=self.sess, observation=observation, stochastic=stochastic)
        
        action = np.asscalar(action)
        value_pred = np.asscalar(value_pred)

        self.record_observations.append(observation)
        self.record_actions.append(action)
        self.record_value_preds.append(value_pred)

        return action, value_pred

    def observe_and_learn(self, reward, terminal):
        self.record_rewards.append(reward)
        
        if terminal == False:
            # if have not reached end of the episode yet, wait
            return
        else:
            # if have reached end of the episode, train

            # make value_preds_next from value_preds
            self.record_value_preds_next = self.record_value_preds[1:] + [0]
            
            # get generalized advantage estimations
            self.record_gaes = self._get_gaes(self.record_rewards, 
                                              self.record_value_preds, 
                                              self.record_value_preds_next)

            # make record_* into numpy array to feed to placeholders
            # TODO FIX
            np_observations = np.reshape(self.record_observations, newshape=[-1] + list(self.policy.observation_space))
            np_actions = np.array(self.record_actions).astype(dtype=np.int32)
            np_rewards = np.array(self.record_rewards).astype(dtype=np.float32)
            np_value_preds_next = np.array(self.record_value_preds_next).astype(dtype=np.float32)
            np_gaes = np.array(self.record_gaes).astype(dtype=np.float32)
            np_gaes = (np_gaes - np_gaes.mean()) / np_gaes.std()

            input_samples = [np_observations, np_actions, np_rewards, np_value_preds_next, np_gaes]
            
            # update old policy with current policy
            self._update_old_policy()
            
            # sample horizon
            #horizon_indices = np.random.randint(low=0, high=np_observations.shape[0], size=self.horizon)
            #horizon_samples = [np.take(a=input_sample, indices=horizon_indices, axis=0) for input_sample in input_samples]

            #batch_indices = np.random.randint(low=0, high=np_observations.shape[0], size=self.batch_size)
            #batch_samples = [np.take(a=input_sample, indices=batch_indices, axis=0) for input_sample in input_samples]
            
            # learn
            for epoch in range(self.epochs):
                # sample batch
                #batch_indices = np.random.randint(low=0, high=self.horizon, size=self.batch_size)
                #batch_samples = [np.take(a=input_sample, indices=batch_indices, axis=0) for input_sample in horizon_samples]
                
                batch_indices = np.random.randint(low=0, high=np_observations.shape[0], size=self.batch_size)
                batch_samples = [np.take(a=input_sample, indices=batch_indices, axis=0) for input_sample in input_samples]

                self._learn(observations=batch_samples[0], 
                            actions=batch_samples[1], 
                            rewards=batch_samples[2], 
                            value_preds_next=batch_samples[3], 
                            gaes=batch_samples[4])
               
            self.record_observations = []
            self.record_actions = []
            self.record_value_preds = []
            self.record_rewards = []

    def _learn(self, observations, actions, rewards, value_preds_next, gaes):
        self.sess.run([self.train_op], feed_dict={self.policy.observation: observations,
                                                  self.old_policy.observation: observations,
                                                  self.actions: actions,
                                                  self.rewards: rewards,
                                                  self.value_preds_next: value_preds_next,
                                                  self.gaes: gaes})

    def _get_gaes(self, rewards, value_preds, value_preds_next):
        deltas = [r + self.gamma * v_next - v for r, v_next, v in zip(rewards, value_preds_next, value_preds)]
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):
            gaes[t] = gaes[t] + self.gamma * self.lmbd * gaes[t+1]

        return gaes

    def _update_old_policy(self):
        self.sess.run(self.assign_ops)
