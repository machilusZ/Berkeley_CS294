import tensorflow as tf
import numpy as np
import gym

class BCModel:
    def __init__(self, env_name, input_dim, output_dim, learning_rate, batch_size, layer_size=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_size = layer_size or [128, 512, 64]
        self.lr = learning_rate
        self.default_batch_size = batch_size
        self.env = None
        self.env_name = env_name

    def add_placeholder(self):
        self.input_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=(None,self.input_dim))
        self.output_placeholder = tf.placeholder(dtype=tf.float32,
                                                 shape=(None,self.input_dim))

    def add_prediction_op(self):
        initializer = tf.contrib.layers.xavier_initializer()
        regularizer = tf.contrib.layers.l2_regularizer(scale=1e-5)
        inputs = self.input_placeholder
        layer1 = tf.layers.dense(
            input,
            self.layers_size[0],
            activation=tf.nn.relu,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer
        )
        layer2 = tf.layers.dense(
            layer1,
            self.layers_size[1],
            activation=tf.nn.relu,
            kernel_regularizer=regularizer,
            kernel_initializer=initializer
        )
        layer3 = tf.layers.dense(
            layer2,
            self.layers_size[2],
            activation=tf.nn.relu,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer
        )
        outputs = tf.layers.dense(
            layer3,
            self.output_dim,
            activation=None,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer
        )

        return outputs


    def add_loss_op(self, logits):
        #expert behavior serves as the right action, imitation = supervised
        labels = self.expert_output_placeholder
        loss= tf.losses.mean_squared_error(labels=labels, predictions=logits)
        return loss

    def add_train_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.lr)
        prop = optimizer.minimize(loss)
        return prop

    def init_session(self):
        sess = tf.Session()
        return sess

    def build(self):
        with tf.variable_scope(self.env_name,reuse=tf.AUTO_REUSE):
            self.add_placeholder()
            self.logits = self.add_prediction_op()
            self.loss = self.add_loss_op(self.logits)
            self.prop = self.add_train_op(self.loss)
            self.sess = self.init_session()
            var_init = tf.global_variables_initializer()
            self.sess.run(var_init)

    def create_feed_dict(self, input_batch, output_batch):
        feeds = {
            self.input_placeholder: input_batch,
            self.expert_outputholder: output_batch
        }
        if output_batch is None:
            feeds.pop(self.expert_output_placeholder)
        return feeds

    def train_on_batch(self, input_batch, output_batch):
        feeds = self.create_feed_dict(input_batch, output_batch)
        _, loss = self.sess.run([self.prop, self.loss], feed_dict=feeds)
        return loss

    def generate_batch(self, inputs, outputs, batch_size=None):
        batch_size = batch_size or self.default_batch_size
        mask = np.random.choice(np.arrange(inputs.shape[0], batch_size))
        inputs_batch = inputs[mask]
        outputs_batch = outputs[mask]
        return inputs_batch,outputs_batch

    def get_action(self, obs):
        #use trained model to get action
        inputs_batch = np.array([obs])
        feeds = self.create_feed_dict(inputs_batch, None)
        logit = self.sess.run(self.logits, feed_dict=feeds)
        return logit


    def evaluate_reward(self, rollouts=20):
        #this function get action from trained model, feed it to
        #env, and get reward
        if not self.env:
            self.env =gym.make(self.env_name)
        returns = []
        max_steps = 1000
        for _ in range(rollouts):
            obs = self.env.reset()
            done = False
            totalreward = 0
            steps = 0
            while not done and steps < max_steps:
                action = self.get_action()
                obs, reward, done, _ = self.env.step(action)
                totalreward += reward
            returns.append(totalreward)

        return returns