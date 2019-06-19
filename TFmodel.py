import tensorflow as tf
import tensorflow.contrib as tc
import os
from preprocess import load_data
import tqdm
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

config = {'learning_rate': 0.005,
          'epoch': 20,
          'clip_grad': 5,
          'batch_size': 32,
          'timestep': 10,
          'rnn_type': 'lstm',
          'lstm_hidden1': 100,
          'lstm_hidden2': 100,
          'attn_hidden': 100,
          'num_heads': 5,
          'dropout_rate': 0.2
          }


class Model():

    '''
    MultiFocus Self-Attention simplify model
    '''

    def __init__(self, config):
        self.config = config
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self._build_graph()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        if not os.path.exists('./tfModel'):
            os.mkdir('./tfModel')
        if not os.path.exists('./summary'):
            os.mkdir('./summary')
        self.writer = tf.summary.FileWriter('./summary')

    def _build_graph(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.config['timestep'], 1])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.dropout_rate = tf.placeholder(dtype=tf.float32)


        length = [self.config['timestep'] for i in range(self.config['batch_size'])]

        with tf.name_scope('LSTM_ENCODER_LAYER'):
            enc, state = self.rnn(self.config['rnn_type'], self.x, length,
                                  self.config['lstm_hidden1'], scope_name='Encoder')

        with tf.name_scope('MultiFocus_Self_Attention'):
            Q = tf.layers.dense(enc, self.config['attn_hidden'], activation=tf.nn.relu)
            K = tf.layers.dense(enc, self.config['attn_hidden'], activation=tf.nn.relu)
            V = tf.layers.dense(enc, self.config['attn_hidden'], activation=tf.nn.relu)
            Q_ = tf.concat(tf.split(Q, self.config['num_heads'], axis=2), axis=0)
            K_ = tf.concat(tf.split(K, self.config['num_heads'], axis=2), axis=0)
            V_ = tf.concat(tf.split(V, self.config['num_heads'], axis=2), axis=0)
            similarities = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # [Batch*head, query_time, key_dim/head]
            weight = similarities / (K_.get_shape().as_list()[-1] ** 0.5)  # [Batch*head, query_time, key_dim/head]
            weight = tf.nn.softmax(weight)  # (h*N, T_q, T_k)
            weight = tf.layers.dropout(weight, rate=self.dropout_rate, training=tf.convert_to_tensor(True))
            to_dec = tf.matmul(weight, V_)  # ( h*N, T_q, C/h)
            to_dec = tf.concat(tf.split(to_dec, self.config['num_heads'], axis=0), axis=2)  # (N, T_q, C)
            to_dec += enc
            to_dec = self._normalize(to_dec)  # (N, T_q, C)

        with tf.name_scope('LSTM_DECODER_LAYER'):
            dec, state = self.rnn(self.config['rnn_type'], to_dec, length,
                                  self.config['lstm_hidden2'], scope_name='decoder')
            self.y_hat = tc.layers.fully_connected(state[0], 1, activation_fn=None)

        mseloss = tf.reduce_mean(tf.losses.mean_squared_error(self.y, self.y_hat))
        l2_loss = tc.layers.apply_regularization(regularizer=tc.layers.l2_regularizer(0.0001),
                                                 weights_list=tf.trainable_variables())
        self.loss = mseloss + l2_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config['learning_rate'])
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.config['clip_grad'])
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def get_cell(self, rnn_type, hidden_size, layer_num=1, dropout_keep_prob=None):
        cells = []
        for i in range(layer_num):
            if rnn_type.endswith('lstm'):
                cell = tc.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
            elif rnn_type.endswith('gru'):
                cell = tc.rnn.GRUCell(num_units=hidden_size)
            elif rnn_type.endswith('rnn'):
                cell = tc.rnn.BasicRNNCell(num_units=hidden_size)
            else:
                raise NotImplementedError('Unsuported rnn type: {}'.format(rnn_type))
            if dropout_keep_prob is not None:
                cell = tc.rnn.DropoutWrapper(cell,
                                             input_keep_prob=dropout_keep_prob,
                                             output_keep_prob=dropout_keep_prob)
            cells.append(cell)
        cells = tc.rnn.MultiRNNCell(cells, state_is_tuple=True)
        return cells

    def rnn(self, rnn_type, inputs, length, hidden_size, layer_num=1, dropout_keep_prob=None, concat=True,
            scope_name=None):
        if not rnn_type.startswith('bi'):
            cell = self.get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            outputs, states = tf.nn.dynamic_rnn(cell, inputs, sequence_length=length, dtype=tf.float32,
                                                scope=scope_name)
            if rnn_type.endswith('lstm'):
                c = [state.c for state in states]
                h = [state.h for state in states]
                states = h
        else:
            cell_fw = self.get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            cell_bw = self.get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_bw, cell_fw, inputs, sequence_length=length, dtype=tf.float32, scope=scope_name
            )
            states_fw, states_bw = states
            if rnn_type.endswith('lstm'):
                c_fw = [state_fw.c for state_fw in states_fw]
                h_fw = [state_fw.h for state_fw in states_fw]
                c_bw = [state_bw.c for state_bw in states_bw]
                h_bw = [state_bw.h for state_bw in states_bw]
                states_fw, states_bw = h_fw, h_bw
            if concat:
                outputs = tf.concat(outputs, 2)
                states = tf.concat([states_fw, states_bw], 1)
            else:
                outputs = outputs[0] + outputs[1]
                states = states_fw + states_bw
        return outputs, states

    def _normalize(self, inputs,
                   epsilon=1e-8,
                   scope="LAYER_NORMAL",
                   reuse=None):

        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta

        return outputs

    def _batch_iter(self, x, y, batch_size,train=True):
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size)
        indices = np.random.permutation(np.arange(data_len))
        if train:
            x = x[indices]
            y = y[indices]
        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x[start_id:end_id], y[start_id:end_id]


    def fit(self, file_name, sequence_length=None, split=1):
        train_x, train_y = load_data(file_name, sequence_length=self.config['timestep'], split=1, train=True)
        step = 0
        for e in (range(self.config['epoch'])):
            losses = []
            batch_data = self._batch_iter(train_x, train_y, self.config['batch_size'])
            for x, y in tqdm.tqdm(batch_data):
                step += 1
                loss, train_op = self.sess.run([self.loss, self.train_op],
                                               feed_dict={self.x: x,
                                                          self.y: y,
                                                          self.dropout_rate: self.config['dropout_rate']})
                loss_sum = tf.Summary(value=[tf.Summary.Value(
                                                            tag = 'model/acc',simple_value=loss)])
                self.writer.add_summary(loss_sum,step)
                losses.append(loss)
            print('>>> Average MSE on %d Epoch: %.4f' % (e, sum(losses) / len(losses)))
            filename = os.path.join('tfModel','model_{}.ckpt'.format(e))
            self.saver.save(self.sess,filename)

    def predict(self,file_name,sequence_length=None,split=1):
        self.saver.restore(self.sess,tf.train.latest_checkpoint('./tfmodel'))
        test_x,test_y = load_data(file_name, sequence_length=self.config['timestep'], split=1, train=False)
        batch_data = self._batch_iter(test_x,test_y,self.config['batch_size'],train=False)
        result = []
        for x,y in batch_data:
            y_hat = self.sess.run(self.y_hat,feed_dict={self.x:x,
                                                        self.dropout_rate:0.})
            result = result.append(y_hat)
        result = np.array(result)
        np.save('./tfmodel.npy',result)


        print(y_hat)





model = Model(config)
model.fit('./data/nyc_taxi_train.csv', sequence_length=model.config['timestep'], split=1)
model.predict('./data/nyc_taxi_test.csv',sequence_length=model.config['timestep'],split=1)
