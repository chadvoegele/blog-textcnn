"""
Modified Implementation from Denny Britz.
https://github.com/dennybritz/cnn-text-classification-tf
"""
import logging
import math
import os
import tempfile
import time
import numpy as np
import sklearn.metrics
import tensorflow as tf

class BatchDataset(object):
    def __init__(self, *datas, shuffle=True):
        if len(set([np.shape(data)[0] for data in datas])) != 1:
            raise Exception('number of samples mistmatch')
        self.datas = datas
        self.n = np.shape(datas[0])[0]
        if shuffle:
            self.permutation = np.random.permutation(self.n)
        else:
            self.permutation = [x for x in range(self.n)]

    def create_batches(n, batch_size):
        n_batches = math.ceil(n/batch_size)
        n_per_batch = math.floor(n/n_batches)
        batches = [n_per_batch]*n_batches
        addon = np.zeros(n_batches, dtype=int)
        addon[:(n-n_per_batch*n_batches)] = 1
        batches = batches + addon
        return batches

    def n_rounds(self, batch_size):
        n_rounds = len(BatchDataset.create_batches(self.n, batch_size))
        return n_rounds

    def batch_iter(self, batch_size):
        batches = BatchDataset.create_batches(self.n, batch_size)
        batch_idx = [0] + list(np.cumsum(batches))

        for i in range(1, len(batch_idx)):
            next_index = self.permutation[batch_idx[i-1]:batch_idx[i]]
            yield [data[next_index] for data in self.datas] if len(self.datas) > 1 else self.datas[0][next_index]

class TextCNN(object):
    # X: s x nsents x nwords
    #   s = #samples, nsents = #sentences, nwords = #words/sentence
    #
    # word_embeddings: mxn matrix
    #   m = number of words, n = embedding dimension
    #   w11 w12 ... w1n
    #    ...
    #   wm1 wm2 ... wmn
    def __init__(self, nsents, nwords, word_embeddings,
            train_batch_size=128, test_batch_size=128, n_epochs=100,
            filter_sizes=[3,4,5], num_filters=128, l2_reg_lambda=0.0,
            learning_rate=0.001, dropout_prob=0.5, eval_X=None, eval_Y=None,
            eval_iter=50):

        self.nwords = nwords
        self.nsents = nsents
        self.word_embeddings = word_embeddings
        self.embedding_size = np.shape(word_embeddings)[1]
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.n_epochs = n_epochs
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob
        self.eval_X = eval_X
        self.eval_Y = eval_Y
        self.eval_iter = eval_iter

        self.variable_scope = 'textcnn'
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.results = { 'train': [], 'train_val': [], 'test_val': [] }

        timestamp = str(int(time.time()))
        outdir = os.path.abspath(os.path.join(tempfile.gettempdir(), 'tf_runs', timestamp))
        logging.info('Writing to %s' % (outdir))
        checkpoint_dir = os.path.abspath(os.path.join(outdir, 'checkpoints'))
        self.checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        with tf.variable_scope(self.variable_scope, reuse=None), self.graph.as_default():
            self.init_net()
            self.saver = tf.train.Saver(tf.all_variables())

    def init_net(self):
        # Placeholders for input, output and dropout
        self.X = tf.placeholder(tf.int32, [None, self.nsents, self.nwords], name='X')
        self.Y = tf.placeholder(tf.float32, [None, 1], name='Y')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            tf_word_embeddings = tf.convert_to_tensor(self.word_embeddings)
            self.embedded_chars = tf.nn.embedding_lookup(tf_word_embeddings, self.X)
            #  self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            self.embedded_chars_expanded = tf.reshape(self.embedded_chars, [-1, self.nwords, self.embedding_size, self.nsents])

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, self.nsents, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name='b')
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv')
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.nwords - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope('output'):
            W = tf.get_variable(
                'W',
                shape=[num_filters_total, 1],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[1]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')

        # CalculateMean cross-entropy loss
        with tf.name_scope('loss'):
            losses = tf.squared_difference(self.scores, self.Y)
            all_losses = tf.reduce_mean(losses)
            self.loss_objective = tf.sqrt(all_losses) + self.l2_reg_lambda * l2_loss
            self.sum_squared_errors = tf.reduce_sum(losses)

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_objective)

    def evaluate(self, batcher):
        losses = []
        n = 0
        for (X, Y) in batcher.batch_iter(self.train_batch_size):
            (sum_squared_loss) = self.sess.run([self.sum_squared_errors],
                    {self.X: X, self.Y: Y})
            n = n + np.shape(Y)[0]
            losses.extend(sum_squared_loss)

        sqrt_mean_loss = math.sqrt(sum(losses)/n)

        return { 'sqrt_mean_loss' : sqrt_mean_loss }

    def fit(self, X, Y):
        with tf.variable_scope(self.variable_scope, reuse=None), self.graph.as_default():
            tf.initialize_all_variables().run(session = self.sess)

        logging.info('Fitting TextCNN')
        train_batcher = BatchDataset(X, Y)
        if self.eval_X is not None and self.eval_Y is not None:
            test_batcher = BatchDataset(self.eval_X, self.eval_Y)
        else:
            test_batcher = None

        iteration = 0
        eval_iter = min(math.ceil(train_batcher.n/self.train_batch_size), self.eval_iter)
        for i in range(0,self.n_epochs):
            batch = 0
            for (X, Y) in train_batcher.batch_iter(self.train_batch_size):
                (_, loss) = self.sess.run(
                        [self.train_op, self.loss_objective],
                        {self.X: X, self.Y: Y})
                batch_results = {
                        'iteration' : iteration,
                        'epoch' : i,
                        'batch' : batch,
                        'loss' : loss,
                        }
                self.results['train'].append(batch_results)
                logging.info(str(batch_results))
                batch = batch + 1
                iteration = iteration + 1

                if iteration % eval_iter == 0:
                    results = self.evaluate(train_batcher)
                    results = dict(list(results.items()) + [('iteration', iteration),
                                                            ('epoch', i),
                                                            ('batch', batch)])
                    self.results['train_val'].append(results)
                    logging.info('Train Epoch %d: %s' % (i, str(results)))

                    if test_batcher is not None:
                        test_results = self.evaluate(test_batcher)
                        test_results = dict(list(test_results.items())
                                            + [('iteration', iteration),
                                               ('epoch', i),
                                               ('batch', batch)])
                        logging.info('Test Epoch %d: %s' % (i, str(test_results)))
                        self.results['test_val'].append(test_results)
                    self.saver.save(self.sess, self.checkpoint_prefix, global_step=iteration)
        return self.results
