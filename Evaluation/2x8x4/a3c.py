import tensorflow.contrib.layers as tl
import numpy as np
import tensorflow as tf


ENTROPY_WEIGHT = 0.01
ENTROPY_EPS = 1e-6
GAMMA = 0.99


class ActorNetwork(object):
    """
    Input to the network is the state, output is the distribution
    of all actions.
    """
    def __init__(self, sess, state_dim, action_dim, learning_rate,Rows,Cols):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.Rows = Rows
        self.Cols = Cols

        # Create the actor network
        #self.inputs = tf.placeholder(tf.float32, [None, self.s_dim])
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.Rows,self.Cols*5,1))
        self.out = self.create_actor_network()

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Selected action, 0-1 vector
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])

        # This gradient will be provided by the critic network
        self.act_grad_weights = tf.placeholder(tf.float32, [None, 1])

        # Compute the objective (log action_vector and entropy)
        self.obj = tf.reduce_sum(tf.multiply(
                       tf.log(tf.reduce_sum(tf.multiply(self.out, self.acts),
                                            reduction_indices=1, keep_dims=True)),
                       -self.act_grad_weights)) \
                   + ENTROPY_WEIGHT * tf.reduce_sum(tf.multiply(self.out,
                                                           tf.log(self.out + ENTROPY_EPS)))

        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.obj, self.network_params)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.lr_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

    def create_actor_network(self):
        with tf.variable_scope('actor'):
            conv1 = tl.conv2d(self.inputs, 48, [self.Rows,20], scope='conv1',activation_fn=tf.nn.leaky_relu,trainable = True)
            conv2 = tl.conv2d(conv1, 64, [self.Rows,10], scope='conv2',activation_fn=tf.nn.leaky_relu,trainable = True)
            #conv3 = tl.conv2d(conv2, 64, [self.Rows,2], scope='conv3')
            flat = tl.flatten(conv2)
            hid_1 = tl.fully_connected(flat, 256, activation_fn=tf.nn.leaky_relu)
            hid_2 = tl.fully_connected(hid_1, 256, activation_fn=tf.nn.leaky_relu)
            hid_3 = tl.fully_connected(hid_2, 128, activation_fn=tf.nn.leaky_relu)
            out = tl.fully_connected(hid_3, self.a_dim, activation_fn=tf.nn.softmax)
            return out


    def train(self, inputs, acts, act_grad_weights):

        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_gradients(self, inputs, acts, act_grad_weights):
        return self.sess.run(self.actor_gradients, feed_dict={
            self.inputs: inputs,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def apply_gradients(self, actor_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.actor_gradients, actor_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is V(s).
    On policy: the action must be obtained from the output of the Actor network.
    """
    def __init__(self, sess, state_dim, learning_rate,Rows,Cols):
        self.sess = sess
        self.s_dim = state_dim
        self.lr_rate = learning_rate
        self.Rows = Rows
        self.Cols = Cols

        # Create the critic network
        #self.inputs = tf.placeholder(tf.float32, [None, self.s_dim])
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.Rows,self.Cols*5,1))
        self.out = self.create_critic_network()

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Network target V(s)
        self.td_target = tf.placeholder(tf.float32, [None, 1])

        # Temporal Difference, will also be weights for actor_gradients
        self.td = tf.subtract(self.td_target, self.out)

        # Mean square error
        self.loss = tf.reduce_mean(tf.square(self.td_target - self.out))

        # Compute critic gradient
        self.critic_gradients = tf.gradients(self.loss, self.network_params)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.lr_rate).\
            apply_gradients(zip(self.critic_gradients, self.network_params))

    def create_critic_network(self):
        with tf.variable_scope('critic'):
            conv1 = tl.conv2d(self.inputs, 48, [self.Rows,20], scope='conv1',activation_fn=tf.nn.leaky_relu,trainable = True)
            conv2 = tl.conv2d(conv1, 64, [self.Rows,10], scope='conv2',activation_fn=tf.nn.leaky_relu,trainable = True)
            #conv3 = tl.conv2d(conv2, 64, [self.Rows,2], scope='conv3')
            flat = tl.flatten(conv2)
            hid_1 = tl.fully_connected(flat, 256, activation_fn=tf.nn.leaky_relu)
            hid_2 = tl.fully_connected(hid_1, 256, activation_fn=tf.nn.leaky_relu)
            hid_3 = tl.fully_connected(hid_2, 128, activation_fn=tf.nn.leaky_relu)
            out = tl.fully_connected(hid_3, 1, activation_fn=None)
            return out

    def train(self, inputs, td_target):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_td(self, inputs, td_target):
        return self.sess.run(self.td, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def get_gradients(self, inputs, td_target):
        return self.sess.run(self.critic_gradients, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def apply_gradients(self, critic_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.critic_gradients, critic_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })


def compute_gradients(s_batch, a_batch, r_batch, terminal, actor, critic):
    """
    batch of s, a, r is from samples in a sequence
    the format is in np.array([batch_size, s/a/r_dim])
    terminal is True when sequence ends as a terminal state
    """
    #print s_batch.shape
    #print a_batch.shape
    assert s_batch.shape[0] == a_batch.shape[0]
    assert s_batch.shape[0] == r_batch.shape[0]
    ba_size = s_batch.shape[0]

    v_batch = critic.predict(s_batch)

    R_batch = np.zeros(r_batch.shape)

    if terminal:
        R_batch[-1, 0] = 0  # terminal state
    else:
        R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state

    for t in reversed(xrange(ba_size - 1)):
        R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

    td_batch = R_batch - v_batch

    actor_gradients = actor.get_gradients(s_batch, a_batch, td_batch)
    critic_gradients = critic.get_gradients(s_batch, R_batch)

    return actor_gradients, critic_gradients, td_batch


def build_summaries():
    td_loss = tf.Variable(0.)
    tf.summary.scalar("TD_loss", td_loss)
    eps_total_reward = tf.Variable(0.)
    tf.summary.scalar("Expected_total_reward", eps_total_reward)

    summary_vars = [td_loss, eps_total_reward]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars
