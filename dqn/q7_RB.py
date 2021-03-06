import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from q7_linear_RB import Linear


from configs.q3_nature import config


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """
    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n
        out = state
        ##############################################################
        """
        TODO: implement the computation of Q values like in the paper
                https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
                https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

              you may find the section "model architecture" of the appendix of the 
              nature paper particulary useful.

              store your result in out of shape = (batch_size, num_actions)

        HINT: you may find tensorflow.contrib.layers useful (imported)
              make sure to understand the use of the scope param

              you can use any other methods from tensorflow
              you are not allowed to import extra packages (like keras,
              lasagne, cafe, etc.)

        """
        ##############################################################
        ################ YOUR CODE HERE - 10-15 lines ################ 
        '''
        print('Nature model')
        with tf.variable_scope(scope, reuse=reuse):
            conv0 = out
            conv1 = tf.contrib.layers.conv2d(conv0, num_outputs = 32, kernel_size = 8, stride = 4)
            conv2 = tf.contrib.layers.conv2d(conv1, num_outputs = 64, kernel_size = 4, stride = 2)
            conv3 = tf.contrib.layers.conv2d(conv2, num_outputs = 64, kernel_size = 3, stride = 1)
            n = conv3.get_shape().as_list()[1]*conv3.get_shape().as_list()[2]*conv3.get_shape().as_list()[3] 
            full_0 = tf.reshape(conv3, shape = [-1, n]) 
            full_1 = tf.contrib.layers.fully_connected(full_0, 512) # (batch_size, num_actions)
            out = tf.contrib.layers.fully_connected(full_1, num_actions, activation_fn=None) # (batch_size, num_actions)
        # out = tf.Print(out, [out])

        assert out.get_shape().as_list() == [None, num_actions], "predictions are not of the right shape. Expected {}, got {}".format([None, num_actions], out.get_shape().as_list())
        '''

        with tf.variable_scope(scope, reuse=reuse):
            conv1 = tf.layers.conv2d(state, 2, kernel_size=(8, 8), 
                activation=tf.nn.relu, strides=(4, 4), trainable=True, padding='same', 
                reuse=reuse, name='conv1')
            conv2 = tf.layers.conv2d(conv1, 64, kernel_size=(4, 4), 
                activation=tf.nn.relu, strides=(2, 2), trainable=True, padding='same', 
                reuse=reuse, name='conv2')
            conv3 = tf.layers.conv2d(conv2, 64, kernel_size=(3, 3), 
                activation=tf.nn.relu, strides=(1, 1), trainable=True, padding='same', 
                reuse=reuse, name='conv3')
            flatten_conv3 = layers.flatten(conv3, scope=scope)
            fc = layers.fully_connected(inputs=flatten_conv3, num_outputs=512, reuse=reuse, 
                activation_fn=tf.nn.relu, trainable=True, scope='fc1')
            out = layers.fully_connected(inputs=fc, num_outputs=num_actions, reuse=reuse, 
                activation_fn=None, trainable=True, scope='fc2')
        ##############################################################
        ######################## END YOUR CODE #######################
        return out


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((80, 80, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
