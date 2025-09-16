# import sys,os
# sys.path.append('/share/scratch/fengyuan/Projects/RecSys/')
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

import time


import tensorflow as tf
import numpy as np

from DAUtils import flip_gradient
import inspect

# import Utils.RecEval as evl
# import Utils.MatUtils as mtl
import Utils.GenUtils as gtl

########################################### The Cross-domain DA Model ##################################################
class DARec(object):
    def __init__(self,
                 sess,
                 training_mode = 'dann',
                 input_dim = 200,

                 pred_dim = 100,
                 pred_sc_tg_lambda = 1.0,
                 pred_reg = 0.0,

                 shared_dim = 100,

                 cls_layers=[100, 2],
                 cls_reg = 0.0,

                 domain_loss_ratio = 1.0,

                 grl_lambda = 1.0,

                 drop_out_rate = 0.0,

                 dec_nn_dim_sc = 500,
                 dec_nn_dim_tg = 200,

                 dec_nn_layers_sc = [200],
                 dec_nn_layers_tg = [300],

                 mode = 'user',

                 lr=0.001,
                 epochs=100, batch_size=128, T=10**3, verbose=False):
        '''Constructor'''
        # Parse the arguments and store them in the model
        self.session = sess

        self.input_dim = input_dim

        self.pred_dim = pred_dim
        self.shared_dim = shared_dim
        self.sc_tg_lambda = pred_sc_tg_lambda
        self.pred_reg = pred_reg

        self.cls_layers = cls_layers
        self.cls_reg = cls_reg

        # self.pred_cls_lambda = pred_cls_lambda

        self.grl_coeff = grl_lambda
        self.drop_out = drop_out_rate

        self.dec_nn_dim_sc = dec_nn_dim_sc
        self.dec_nn_dim_tg = dec_nn_dim_tg

        self.dec_nn_layers_sc = dec_nn_layers_sc
        self.dec_nn_layers_tg = dec_nn_layers_tg

        self.domain_loss_ratio = domain_loss_ratio

        self.training_mode = training_mode
        self.lr = lr

        self.mode = mode

        self.epochs = epochs
        self.batch_size = batch_size
        self.skip_step = T
        self.verbose = verbose

        gtl.print_paras(inspect.currentframe())

    def prepare_data(self, original_matrix_sc, train_matrix_sc, test_matrix_sc,
                            original_matrix_tg, train_matrix_tg, test_matrix_tg,
                            embedding_arr_sc, embedding_arr_tg):
        self.num_user_sc, self.num_item_sc = original_matrix_sc.shape
        self.num_user_tg, self.num_item_tg = original_matrix_tg.shape

        if self.mode == 'user':
            assert self.num_user_sc == self.num_user_tg

            self.train_array_sc, self.test_array_sc = train_matrix_sc.toarray(), test_matrix_sc.toarray()
            self.train_array_tg, self.test_array_tg = train_matrix_tg.toarray(), test_matrix_tg.toarray()

            self.embed_arr = np.vstack((embedding_arr_sc, embedding_arr_tg))
            self.domain_arr = np.vstack((np.tile([1., 0.], [self.num_user_sc, 1]),
                                        np.tile([0., 1.], [self.num_user_tg, 1])))

            self.num_training = self.num_user_sc * 2  # There are two domains with the same number of users
            self.num_batch = self.num_training // self.batch_size

        else:
            self.train_array_sc, self.test_array_sc = train_matrix_sc.toarray().T, test_matrix_sc.toarray().T
            self.train_array_tg, self.test_array_tg = train_matrix_tg.toarray().T, test_matrix_tg.toarray().T
            if self.train_array_sc.shape[0] < self.train_array_tg.shape[0]:
                times = self.train_array_tg.shape[0] // self.train_array_sc.shape[0]
                residue = self.train_array_tg.shape[0] % self.train_array_sc.shape[0]
                self.train_array_sc = np.vstack(
                                                (np.tile(self.train_array_sc,(times,1)),
                                                self.train_array_sc[:residue]))
                embedding_arr_sc = np.vstack(
                                            (np.tile(embedding_arr_sc, (times,1)),
                                             embedding_arr_sc[:residue]))
                self.num_item = self.train_array_tg.shape[0]

            elif self.train_array_sc.shape[0] > self.train_array_tg.shape[0]:
                times = self.train_array_sc.shape[0] // self.train_array_tg.shape[0]
                residue = self.train_array_sc.shape[0] % self.train_array_tg.shape[0]
                self.train_array_tg = np.vstack(
                                                (np.tile(self.train_array_tg, (times, 1)),
                                                self.train_array_tg[:residue]))
                embedding_arr_tg = np.vstack(
                                            (np.tile(embedding_arr_tg, (times, 1)),
                                            embedding_arr_tg[:residue]))
                self.num_item = self.train_array_sc.shape[0]
            else:
                self.num_item = self.train_array_sc.shape[0]

            self.embed_arr = np.vstack((embedding_arr_sc, embedding_arr_tg))
            self.domain_arr = np.vstack((np.tile([1., 0.], [self.num_item, 1]),
                                         np.tile([0., 1.], [self.num_item, 1])))

            self.num_training = self.num_item * 2  # There are two domains with the same number of users
            self.num_batch = self.num_training // self.batch_size

            print(self.num_item)

        print(self.train_array_sc.shape)
        print(self.train_array_tg.shape)
        print("Data Preparation Completed.")

    def build_model(self):
        with tf.variable_scope("DARec_Model", reuse=tf.AUTO_REUSE):
            self.istraining = tf.placeholder(dtype=tf.bool, shape=[], name='training_flag')
            self.dropout_rate = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_rate')

            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name='features')

            if self.mode == 'user':
                self.ratings_sc = tf.placeholder(dtype=tf.float32, shape=[None, self.num_item_sc], name='ratings_sc')
                self.ratings_tg = tf.placeholder(dtype=tf.float32, shape=[None, self.num_item_tg], name='ratings_tg')
            else:
                self.ratings_sc = tf.placeholder(dtype=tf.float32, shape=[None, self.num_user_sc], name='ratings_sc')
                self.ratings_tg = tf.placeholder(dtype=tf.float32, shape=[None, self.num_user_tg], name='ratings_tg')

            self.domain = tf.placeholder(dtype=tf.float32, shape=[None, 2])
            self.grl = tf.placeholder(dtype=tf.float32, shape=[])

            with tf.variable_scope("Rating_Predictions",reuse=tf.AUTO_REUSE): # Map the two domains into a shared space
                pred_w1 = tf.get_variable(name='shared_w1',
                                            shape=[self.input_dim, self.pred_dim],
                                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                pred_b1 = tf.get_variable(name='shared_b1',
                                            shape=[self.pred_dim],
                                            initializer=tf.zeros_initializer())

                pred_w2 = tf.get_variable(name='shared_w2',
                                          shape=[self.pred_dim, self.shared_dim],
                                          initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                pred_b2 = tf.get_variable(name='shared_b2',
                                          shape=[self.shared_dim],
                                          initializer=tf.zeros_initializer())


                pred_w2_sc = tf.get_variable(name='shared_w2_sc',
                                                shape=[self.shared_dim, self.dec_nn_dim_sc],
                                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                pred_b2_sc = tf.get_variable(name='shared_b2_sc',
                                                shape=[self.dec_nn_dim_sc],
                                                initializer=tf.zeros_initializer())

                pred_w2_tg = tf.get_variable(name='shared_w2_tg',
                                                shape=[self.shared_dim, self.dec_nn_dim_tg],
                                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                pred_b2_tg = tf.get_variable(name='shared_b2_tg',
                                                shape=[self.dec_nn_dim_tg],
                                                initializer=tf.zeros_initializer())
                if self.mode == 'user':
                    pred_out_w_sc = tf.get_variable(name='shared_out_w_sc',
                                                     shape=[self.dec_nn_dim_sc, self.num_item_sc],
                                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                    pred_out_b_sc = tf.get_variable(name='shared_out_b_sc',
                                                     shape=[self.num_item_sc],
                                                     initializer=tf.zeros_initializer())

                    pred_out_w_tg = tf.get_variable(name='shared_out_w_tg',
                                                     shape=[self.dec_nn_dim_tg, self.num_item_tg],
                                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                    pred_out_b_tg = tf.get_variable(name='shared_out_b_tg',
                                                     shape=[self.num_item_tg],
                                                     initializer=tf.zeros_initializer())
                else:
                    pred_out_w_sc = tf.get_variable(name='shared_out_w_sc',
                                                    shape=[self.dec_nn_dim_sc, self.num_user_sc],
                                                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                    pred_out_b_sc = tf.get_variable(name='shared_out_b_sc',
                                                    shape=[self.num_user_sc],
                                                    initializer=tf.zeros_initializer())

                    pred_out_w_tg = tf.get_variable(name='shared_out_w_tg',
                                                    shape=[self.dec_nn_dim_tg, self.num_user_tg],
                                                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                    pred_out_b_tg = tf.get_variable(name='shared_out_b_tg',
                                                    shape=[self.num_user_tg],
                                                    initializer=tf.zeros_initializer())

                # AutoEncoder for Shared Feature Extraction
                # print(np.shape(self.input), np.shape(pred_w1), np.shape(pred_b1))
                pred_encode = tf.nn.relu(tf.matmul(self.input, pred_w1) + pred_b1)
                # print(np.shape(pred_encode))
                shared_vec = tf.cond(self.istraining,
                                     lambda: tf.layers.dropout(pred_encode, rate=self.dropout_rate, name='encode_dropout'),
                                     lambda: pred_encode)

                shared_vec = tf.nn.relu(tf.matmul(shared_vec, pred_w2) + pred_b2)

                shared_vec = tf.cond(self.istraining,
                                     lambda: tf.layers.dropout(shared_vec, rate=self.dropout_rate,
                                                               name='encode_dropout2'),
                                     lambda: shared_vec)

                pred_decode_sc = shared_vec
                pred_decode_tg = shared_vec

                # for i in range(len(self.dec_nn_layers_sc)):
                #     pred_decode_sc = tf.layers.dense(pred_decode_sc,
                #                                  units=self.dec_nn_layers_sc[i],
                #                                  activation=tf.nn.relu,
                #                                  use_bias=True,
                #                                  kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01),
                #                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.pred_reg),
                #                                  bias_initializer=tf.zeros_initializer(),
                #                                  bias_regularizer=tf.contrib.layers.l2_regularizer(scale=self.pred_reg),
                #                                  name='pred_dec_layer_sc{0}'.format(i))
                #     pred_decode_sc = tf.cond(self.istraining,
                #                          lambda: tf.layers.dropout(pred_decode_sc, rate=self.dropout_rate,
                #                                                    name='decode_dropout_sc{0}'.format(i)),
                #                          lambda: pred_decode_sc)
                #
                # for i in range(len(self.dec_nn_layers_tg)):
                #     pred_decode_tg = tf.layers.dense(pred_decode_tg,
                #                                  units=self.dec_nn_layers_tg[i],
                #                                  activation=tf.nn.relu,
                #                                  use_bias=True,
                #                                  kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01),
                #                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.pred_reg),
                #                                  bias_initializer=tf.zeros_initializer(),
                #                                  bias_regularizer=tf.contrib.layers.l2_regularizer(scale=self.pred_reg),
                #                                  name='pred_dec_layer_tg{0}'.format(i))
                #     pred_decode_tg = tf.cond(self.istraining,
                #                              lambda: tf.layers.dropout(pred_decode_tg, rate=self.dropout_rate,
                #                                                        name='decode_dropout_tg{0}'.format(i)),
                #                              lambda: pred_decode_tg)

                # print(np.shape(shared_vec), np.shape(pred_w2_sc))
                pred_decode_sc= tf.nn.relu(tf.matmul(shared_vec, pred_w2_sc) + pred_b2_sc)
                pred_decode_tg = tf.nn.relu(tf.matmul(shared_vec, pred_w2_tg) + pred_b2_tg)

                # pred_decode_sc = tf.cond(self.istraining,
                #                          lambda: tf.layers.dropout(pred_decode_sc, rate=self.dropout_rate,
                #                                                    name='decode_dropout_sc'),
                #                          lambda: pred_decode_sc)
                #
                # pred_decode_tg = tf.cond(self.istraining,
                #                          lambda: tf.layers.dropout(pred_decode_tg, rate=self.dropout_rate,
                #                                                    name='decode_dropout_tg'),
                #                          lambda: pred_decode_tg)

                # Prediction Output Layer
                pred_decode_sc = tf.identity(tf.matmul(pred_decode_sc, pred_out_w_sc) + pred_out_b_sc)
                pred_decode_tg = tf.identity(tf.matmul(pred_decode_tg, pred_out_w_tg) + pred_out_b_tg)

                # print(np.shape(pred_decode_sc), np.shape(pred_decode_tg))

                # Rating Prediction
                self.pred_y_sc = tf.cond(self.istraining,
                                         lambda: tf.multiply(pred_decode_sc, tf.sign(self.ratings_sc)),
                                         lambda: pred_decode_sc)

                self.pred_y_tg = tf.cond(self.istraining,
                                         lambda: tf.multiply(pred_decode_tg, tf.sign(self.ratings_tg)),
                                         lambda: pred_decode_tg)

                # Losses for Rating Prediction

                # Set the loss for the target domain input to zero
                diff_sc = self.ratings_sc - self.pred_y_sc

                mask_loss_sc = np.array([True, False] * (self.batch_size//2))
                loss_mat_sc = tf.boolean_mask(diff_sc, mask_loss_sc)
                diff_tg = self.ratings_tg - self.pred_y_tg
                mask_loss_tg = np.array([False, True] * (self.batch_size//2))
                loss_mat_tg = tf.boolean_mask(diff_tg, mask_loss_tg)

                base_loss = tf.reduce_mean(tf.square(loss_mat_sc)) \
                            + self.sc_tg_lambda * tf.reduce_mean(tf.square(loss_mat_tg))

                reg_loss = self.pred_reg * \
                           (tf.nn.l2_loss(pred_w1) + tf.nn.l2_loss(pred_b1) +
                            tf.nn.l2_loss(pred_w2) + tf.nn.l2_loss(pred_b2) +
                            tf.nn.l2_loss(pred_w2_sc) + tf.nn.l2_loss(pred_b2_sc) +
                            tf.nn.l2_loss(pred_w2_tg) + tf.nn.l2_loss(pred_b2_tg) +
                            tf.nn.l2_loss(pred_out_w_sc) + tf.nn.l2_loss(pred_out_b_sc) +
                            tf.nn.l2_loss(pred_out_w_tg) + tf.nn.l2_loss(pred_out_b_tg))
                            # + tf.reduce_sum(tf.losses.get_regularization_loss(scope="DARec_Model/Rating_Predictions"))

                self.pred_loss = base_loss + reg_loss
                # print(np.shape(self.pred_loss))

            with tf.variable_scope("Domain_Classifier",reuse=tf.AUTO_REUSE):
                # Flip the gradient when backpropagating through this operation
                feat = flip_gradient(shared_vec, self.grl)

                # MLP for domain classification
                mlp_vector = feat

                assert self.cls_layers[-1] == 2 # Check whether the classification is binary
                for i in range(len(self.cls_layers)):
                    mlp_vector = tf.layers.dense(mlp_vector,
                                                 units=self.cls_layers[i],
                                                 activation=tf.nn.sigmoid,
                                                 use_bias=True,
                                                 kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01),
                                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.cls_reg),
                                                 bias_initializer=tf.zeros_initializer(),
                                                 bias_regularizer=tf.contrib.layers.l2_regularizer(scale=self.cls_reg),
                                                 name='cls_layer{0}'.format(i))

                self.cls_y = tf.nn.softmax(mlp_vector)
                self.domain_base_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mlp_vector, labels=self.domain))
                self.domain_reg_loss = tf.reduce_sum(tf.losses.get_regularization_loss(scope="DARec_Model/Domain_Classifier"))
                self.domain_loss = self.domain_base_loss + self.domain_reg_loss
                # print(np.shape(self.domain))
                # print(np.shape(self.domain_loss))

            self.total_loss = self.pred_loss + self.domain_loss_ratio * self.domain_loss

            self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss)
            # self.opt_pred = tf.train.AdamOptimizer(self.lr).minimize(self.pred_loss)

            # Metrics

            self.train_rms_sc = tf.sqrt(tf.reduce_mean(tf.square(loss_mat_sc)))
            self.train_rms_tg = tf.sqrt(tf.reduce_mean(tf.square(loss_mat_tg)))
            self.train_rms = tf.sqrt(tf.square(self.train_rms_sc) + tf.square(self.train_rms_tg))

            self.correct_domain_pred = tf.equal(tf.argmax(self.domain, 1), tf.argmax(self.cls_y, 1))

            # print(np.shape(tf.argmax(self.domain, 1)), tf.argmax(self.cls_y, 1))

            self.domain_acc = tf.reduce_mean(tf.cast(self.correct_domain_pred, tf.float32))

            print('Model Building Completed.')

############################################### Functions to run the model #############################################
    def train_one_epoch(self, epoch):
        n_batches = 0
        total_loss = 0
        total_rms = 0
        total_domain_loss = 0
        total_domain_acc = 0
        #
        # if self.training_mode == 'dann':
            # Select users from the two different domains in an unbalanced way
            # random_idx = np.random.permutation(self.num_user_sc * 2)

            # Select users from the two different domains in a balanced way
        if self.mode == 'user':
            random_idx = np.vstack((np.random.permutation(self.num_user_sc),
                                    np.random.permutation(self.num_user_sc) + self.num_user_sc))\
                            .flatten('F')
        else:
            random_idx = np.vstack((np.random.permutation(self.num_item),
                                    np.random.permutation(self.num_item) + self.num_item))\
                            .flatten('F')

        for i in range(self.num_batch):
            # start_time = time.time()
            # print("Start Time: {0} \n".format(start_time))

            # Prepare feeding data for one batch
            if i == self.num_batch - 1:
                batch_idx = random_idx[i * self.batch_size:]
            else:
                batch_idx = random_idx[i * self.batch_size: (i + 1) * self.batch_size]

            # The rows
            if self.mode == 'user':
                idx = [k if k < self.num_user_sc else k - self.num_user_sc for k in batch_idx]
            else:
                idx = [k if k < self.num_item else k - self.num_item for k in batch_idx]

            # print("Time Used: {0} \n".format(time.time()-start_time))

            feed_config = {
                    self.istraining:    True,
                    self.dropout_rate:  self.drop_out,
                    self.input:         self.embed_arr[batch_idx],
                    self.ratings_sc:    self.train_array_sc[idx],
                    self.ratings_tg:    self.train_array_tg[idx],
                    self.domain:        self.domain_arr[batch_idx],
                    self.grl:           self.grl_coeff
                }

            _, pred_loss, domain_loss, batch_loss, batch_rms, batch_domain_acc, batch_domain_loss \
                = self.session.run([self.opt,
                                    self.pred_loss,
                                    self.domain_loss,
                                    self.total_loss,
                                    self.train_rms,
                                    self.domain_acc,
                                    self.domain_base_loss],
                                    feed_dict=feed_config)

            # print(pred_loss/domain_loss)

            n_batches += 1
            total_loss += batch_loss
            total_rms += batch_rms
            total_domain_loss += batch_domain_loss
            total_domain_acc += batch_domain_acc
            # print("Total_Loss: {0}".format(batch_loss))

        if self.verbose:
            print("="*80)
            print("Training Epoch {0}: [Loss] {1}, [RMSE] {2}, [DomAcc] {3}"
                    .format(epoch, total_loss/n_batches, total_rms/n_batches, total_domain_acc/n_batches))
            print("Training Epoch {0}: [Domain Loss] {1}".format(epoch, total_domain_loss/n_batches))

    def eval_one_epoch(self, epoch):
        # Input the training data to predict the ratings in the test array
        # feed_config = {
        #     self.istraining: False,
        #     self.dropout_rate: self.drop_out,
        #     self.input: self.embed_arr,
        #     self.ratings_sc: self.train_array_sc,
        #     self.ratings_tg: self.train_array_tg,
        #     self.domain: self.domain_arr,
        #     self.grl: self.grl_coeff
        # }

        pred_y_sc, pred_y_tg, domain_acc = self.session.run([self.pred_y_sc, self.pred_y_tg, self.domain_acc],
                                                            feed_dict={
                                                                self.istraining: False,
                                                                self.dropout_rate: self.drop_out,
                                                                self.input: self.embed_arr,
                                                                self.ratings_sc: self.train_array_sc,
                                                                self.ratings_tg: self.train_array_tg,
                                                                self.domain: self.domain_arr,
                                                                self.grl: self.grl_coeff
                                                            })

        pred_y_sc, pred_y_tg = pred_y_sc.clip(min=1, max=5), pred_y_tg.clip(min=1, max=5)

        # Extract the nonzero rating indices of the test array from the prediction and test array
        test_pred_y_sc, test_pred_y_tg = pred_y_sc[self.test_array_sc.nonzero()], \
                                         pred_y_tg[self.test_array_tg.nonzero()]

        truth_sc, truth_tg = self.test_array_sc[self.test_array_sc.nonzero()], \
                             self.test_array_tg[self.test_array_tg.nonzero()]

        # Calculate Metrics
        mae_sc, mae_tg = np.mean(np.abs(truth_sc - test_pred_y_sc)), \
                         np.mean(np.abs(truth_tg - test_pred_y_tg))
        rms_sc, rms_tg = np.sqrt(np.mean(np.square(truth_sc - test_pred_y_sc))), \
                         np.sqrt(np.mean(np.square(truth_tg - test_pred_y_tg)))
        mae_total = (mae_sc + mae_tg) / 2
        rms_total = np.sqrt(np.square(rms_sc) + np.square(rms_tg))

        print("Testing Epoch {0} :  [Source_MAE] {1} and [Source_RMS] {2}".format(epoch, mae_sc, rms_sc))
        print("Testing Epoch {0} :  [Target_MAE] {1} and [Target_RMS] {2}".format(epoch, mae_tg, rms_tg))
        print("Testing Epoch {0} :  [MAE] {1} and [RMS] {2}".format(epoch, mae_total, rms_total))
        print("Testing Epoch {0} :  [DomAcc] {1}".format(epoch, domain_acc))
        return mae_total, rms_total

    # Final Training of the model
    def train(self, restore=False, save=False, datafile=None):

        if restore:  # Restore the model from checkpoint
            self.restore_model(datafile, verbose=True)
        else:
            self.session.run(tf.global_variables_initializer())

        if not save:  # Do not save the model
            self.eval_one_epoch(-1)
            for i in range(self.epochs):
                self.train_one_epoch(i)
                self.eval_one_epoch(i)

        else:  # Save the model while training
            _, previous_rms = self.eval_one_epoch(-1)
            for i in range(self.epochs):
                self.train_one_epoch(i)
                _, rms = self.eval_one_epoch(i)
                if rms < previous_rms:
                    previous_rms = rms
                    self.save_model(datafile, verbose=False)

    # Save the model
    def save_model(self, datafile, verbose=False):
        saver = tf.train.Saver()
        path = saver.save(self.session, datafile)
        if verbose:
            print("Model Saved in Path: {0}".format(path))

    # Restore the model
    def restore_model(self, datafile, verbose=False):
        saver = tf.train.Saver()
        saver.restore(self.session, datafile)
        if verbose:
            print("Model Restored from Path: {0}".format(datafile))

    # Evaluate the model
    def evaluate(self, datafile):
        self.restore_model(datafile,True)
        self.eval_one_epoch(-1)

########################################################################################################################
