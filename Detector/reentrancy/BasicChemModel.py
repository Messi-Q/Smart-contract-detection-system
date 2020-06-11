#!/usr/bin/env/python
from __future__ import print_function
from typing import List, Any, Sequence

import tensorflow as tf
import time
import os
import json
import numpy as np
import random
import warnings

warnings.filterwarnings('ignore')

from utils import MLP, ThreadedIterator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ChemModel(object):
    @classmethod
    def default_params(cls):
        return {
            'num_epochs': 20,
            'patience': 20,
            'learning_rate': 0.002,
            'clamp_gradient_norm': 1,  # 1.0->0.8
            'out_layer_dropout_keep_prob': 1,  # 1.0->0.8

            'hidden_size': 250,  # 256/512/1024/2048
            'use_graph': True,

            'tie_fwd_bkwd': False,  # True->False
            'task_ids': [0],

            'train_file': '../data/reentrancy/train.json',
            'valid_file': '../data/reentrancy/valid.json'
        }

    def __init__(self, args):
        self.args = args

        # Collect argument things:
        data_dir = ''
        if '--data_dir' in args and args['--data_dir'] is not None:
            data_dir = args['--data_dir']
        self.data_dir = data_dir

        # random_seed = None
        random_seed = args.get('--random_seed')
        self.random_seed = int(9930)

        threshold = args.get('--thresholds')
        self.threshold = float(0.425)

        self.run_id = "_".join([time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])
        log_dir = args.get('--log_dir') or '.'
        self.log_file = os.path.join(log_dir, "%s_log.json" % self.run_id)
        self.best_model_file = os.path.join(log_dir, "%s_model_best.pickle" % self.run_id)

        # Collect parameters:
        params = self.default_params()
        config_file = args.get('--config-file')
        if config_file is not None:
            with open(config_file, 'r') as f:
                params.update(json.load(f))
        config = args.get('--config')
        if config is not None:
            params.update(json.loads(config))
        self.params = params

        # print("Run %s starting with following parameters:\n%s" % (self.run_id, json.dumps(self.params)))
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        # print("Run with current seed %s " % self.random_seed)

        # Load baseline:
        self.max_num_vertices = 0
        self.num_edge_types = 0
        self.annotation_size = 0
        self.num_graph = 1
        self.train_num_graph = 0
        self.valid_num_graph = 0

        self.train_data, self.train_num_graph = self.load_data(params['train_file'], is_training_data=True)
        self.valid_data, self.valid_num_graph = self.load_data(params['valid_file'], is_training_data=False)

        # Build the actual model
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            self.placeholders = {}
            self.weights = {}
            self.ops = {}
            self.make_model()
            self.make_train_step()
            self.initialize_model()

    def load_data(self, file_name, is_training_data: bool):
        full_path = os.path.join(self.data_dir, file_name)

        # print("Loading baseline from %s" % full_path)
        with open(full_path, 'r') as f:
            data = json.load(f)

        restrict = self.args.get("--restrict_data")
        if restrict is not None and restrict > 0:
            data = data[:restrict]

        # Get some common baseline out:
        num_fwd_edge_types = 0
        for g in data:
            self.max_num_vertices = max(self.max_num_vertices, max([v for e in g['graph'] for v in [e[0], e[2]]]))
            num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))
        self.num_edge_types = max(self.num_edge_types, num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2))
        self.annotation_size = max(self.annotation_size, len(data[0]["node_features"][0]))

        return self.process_raw_graphs(data, is_training_data)

    @staticmethod
    def graph_string_to_array(graph_string: str) -> List[List[int]]:
        return [[int(v) for v in s.split(' ')]
                for s in graph_string.split('\n')]

    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool) -> Any:
        raise Exception("Models have to implement process_raw_graphs!")

    def make_model(self):
        self.placeholders['target_values'] = tf.placeholder(tf.float32, [len(self.params['task_ids']), None],
                                                            name='target_values')
        self.placeholders['target_mask'] = tf.placeholder(tf.float32, [len(self.params['task_ids']), None],
                                                          name='target_mask')
        self.placeholders['num_graphs'] = tf.placeholder(tf.int32, [], name='num_graphs')

        self.placeholders['out_layer_dropout_keep_prob'] = tf.placeholder(tf.float32, [],
                                                                          name='out_layer_dropout_keep_prob')

        with tf.variable_scope("graph_model"):
            self.prepare_specific_graph_model()
            # This does the actual graph work:
            if self.params['use_graph']:
                self.ops['final_node_representations'] = self.compute_final_node_representations()
            else:
                self.ops['final_node_representations'] = tf.zeros_like(self.placeholders['process_raw_graphs'])

        self.ops['losses'] = []
        for (internal_id, task_id) in enumerate(self.params['task_ids']):
            with tf.variable_scope("out_layer_task%i" % task_id):
                with tf.variable_scope("regression_gate"):
                    self.weights['regression_gate_task%i' % task_id] = MLP(2 * self.params['hidden_size'], 1, [],
                                                                           self.placeholders[
                                                                               'out_layer_dropout_keep_prob'])
                with tf.variable_scope("regression"):
                    self.weights['regression_transform_task%i' % task_id] = MLP(self.params['hidden_size'], 1, [],
                                                                                self.placeholders[
                                                                                    'out_layer_dropout_keep_prob'])
                computed_values, sigm_val, initial_re = self.gated_regression(self.ops['final_node_representations'],
                                                                              self.weights[
                                                                                  'regression_gate_task%i' % task_id],
                                                                              self.weights[
                                                                                  'regression_transform_task%i' % task_id])

                def f(x):
                    x = 1 * x
                    x = x.astype(np.float32)
                    return x

                new_computed_values = tf.nn.sigmoid(computed_values)
                new_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=computed_values,
                                                                                  labels=self.placeholders[
                                                                                             'target_values'][
                                                                                         internal_id, :]))
                a = tf.math.greater_equal(new_computed_values, self.threshold)
                a = tf.py_func(f, [a], tf.float32)
                self.ops['real_label'] = self.placeholders['target_values'][internal_id, :]
                correct_pred = tf.equal(a, self.placeholders['target_values'][internal_id, :])
                self.ops['new_computed_values'] = new_computed_values
                self.ops['sigm_val'] = sigm_val  # QP:graph feature
                self.ops['initial_re'] = initial_re  # QP:inital nodes
                self.ops['accuracy_task%i' % task_id] = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

                b = tf.multiply(self.placeholders['target_values'][internal_id, :], 2)
                b = tf.py_func(f, [b], tf.float32)
                c = tf.cast(a, tf.float32)
                d = tf.math.add(b, c)
                self.ops['sigm_c'] = correct_pred

                d_TP = tf.math.equal(d, 3)
                TP = tf.reduce_sum(tf.cast(d_TP, tf.float32))  # true positive
                d_FN = tf.math.equal(d, 2)
                FN = tf.reduce_sum(tf.cast(d_FN, tf.float32))  # false negative
                d_FP = tf.math.equal(d, 1)
                FP = tf.reduce_sum(tf.cast(d_FP, tf.float32))  # false positive
                d_TN = tf.math.equal(d, 0)
                TN = tf.reduce_sum(tf.cast(d_TN, tf.float32))  # true negative
                self.ops['sigm_sum'] = tf.add_n([TP, FN, FP, TN])
                self.ops['sigm_TP'] = TP
                self.ops['sigm_FN'] = FN
                self.ops['sigm_FP'] = FP
                self.ops['sigm_TN'] = TN

                R = tf.cast(tf.divide(TP, tf.add(TP, FN)), tf.float32)  # Recall
                P = tf.cast(tf.divide(TP, tf.add(TP, FP)), tf.float32)  # Precision
                FPR = tf.cast(tf.divide(FP, tf.add(TN, FP)), tf.float32)  # FPR: false positive rate
                D_TP = tf.add(TP, TP)
                F1 = tf.cast(tf.divide(D_TP, tf.add_n([D_TP, FP, FN])), tf.float32)  # F1
                self.ops['sigm_Recall'] = R
                self.ops['sigm_Precision'] = P
                self.ops['sigm_F1'] = F1
                self.ops['sigm_FPR'] = FPR
                self.ops['losses'].append(new_loss)
        self.ops['loss'] = tf.reduce_sum(self.ops['losses'])

    def make_train_step(self):
        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if self.args.get('--freeze-graph-model'):
            graph_vars = set(self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="graph_model"))
            filtered_vars = []
            for var in trainable_vars:
                if var not in graph_vars:
                    filtered_vars.append(var)
                else:
                    # print("Freezing weights of variable %s." % var.name)
                    pass
            trainable_vars = filtered_vars
        optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
        grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        self.ops['train_step'] = optimizer.apply_gradients(clipped_grads)
        # Initialize newly-introduced variables:
        self.sess.run(tf.local_variables_initializer())

    def gated_regression(self, last_h, regression_gate, regression_transform):
        raise Exception("Models have to implement gated_regression!")

    def prepare_specific_graph_model(self) -> None:
        raise Exception("Models have to implement prepare_specific_graph_model!")

    def compute_final_node_representations(self) -> tf.Tensor:
        raise Exception("Models have to implement compute_final_node_representations!")

    def make_minibatch_iterator(self, data: Any, is_training: bool):

        raise Exception("Models have to implement make_minibatch_iterator!")

    def run_epoch(self, epoch_name: str, data, is_training: bool):
        chemical_accuracies = np.array([0.066513725, 0.012235489, 0.071939046, 0.033730778, 0.033486113, 0.004278493,
                                        0.001330901, 0.004165489, 0.004128926, 0.00409976, 0.004527465, 0.012292586,
                                        0.037467458])

        loss = 0
        accuracies = []
        start_time = time.time()
        processed_graphs = 0
        label_pred = 0
        real_label = 0
        accuracy_ops = [self.ops['accuracy_task%i' % task_id] for task_id in self.params['task_ids']]
        batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data, is_training), max_queue_size=5)
        for step, batch_data in enumerate(batch_iterator):
            num_graphs = batch_data[self.placeholders['num_graphs']]
            processed_graphs += num_graphs
            if is_training:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = self.params[
                    'out_layer_dropout_keep_prob']
                fetch_list = [self.ops['loss'], accuracy_ops, self.ops['train_step']]
            else:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                fetch_list = [self.ops['loss'], accuracy_ops]

            # compute the prediction result
            result = self.sess.run(fetch_list, feed_dict=batch_data)
            (batch_loss, batch_accuracies) = (result[0], result[1])

            loss += batch_loss * num_graphs
            accuracies.append(np.array(batch_accuracies) * num_graphs)

            # print("Running %s, batch %i (has %i graphs). "
            #       "Loss so far: %.4f" % (epoch_name, step, num_graphs, loss / processed_graphs), end='\r')

            if is_training is not True:
                label_pred = self.sess.run([self.ops['new_computed_values']], feed_dict=batch_data)[0]
                real_label = float(self.sess.run([self.ops['real_label']], feed_dict=batch_data)[0])

        accuracies = np.sum(accuracies, axis=0) / processed_graphs
        loss = loss / processed_graphs
        error_ratios = accuracies / chemical_accuracies[self.params["task_ids"]]
        instance_per_sec = processed_graphs / (time.time() - start_time)

        return loss, accuracies, error_ratios, instance_per_sec, label_pred, real_label

    def train(self):
        label_pred = 0
        real_label = 0
        log_to_save = []
        with self.graph.as_default():
            for epoch in range(1, self.params['num_epochs'] + 1):
                # print("== Epoch %i" % epoch)
                self.num_graph = self.train_num_graph
                train_loss, train_accs, train_errs, train_speed, _, _ = self.run_epoch("epoch %i (training)" % epoch,
                                                                                    self.train_data, True)
                accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], train_accs)])
                errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], train_errs)])
                # print("\r\x1b[K Train: loss: %.5f | acc: %s | error_ratio: %s | instances/sec: %.2f" % (train_loss,
                #                                                                                         accs_str,
                #                                                                                         errs_str,
                #                                                                                         train_speed))

                self.num_graph = self.valid_num_graph
                valid_loss, valid_accs, valid_errs, valid_speed, label_pred, real_label = self.run_epoch(
                    "epoch %i (validation)" % epoch,
                    self.valid_data, False)
                accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], valid_accs)])
                errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], valid_errs)])
                # print("\r\x1b[K Valid: loss: %.5f | acc: %s | error_ratio: %s | instances/sec: %.2f" % (valid_loss,
                #                                                                                         accs_str,
                #                                                                                         errs_str,
                #                                                                                         valid_speed))
                log_entry = {
                    'epoch': epoch,
                    'train_results': (train_loss, train_accs.tolist(), train_errs.tolist(), train_speed),
                    'valid_results': (valid_loss, valid_accs.tolist(), valid_errs.tolist(), valid_speed),
                }
                log_to_save.append(log_entry)

        if label_pred >= self.threshold:
            label_pred = 1
        else:
            label_pred = 0

        return label_pred, int(real_label)

    def initialize_model(self) -> None:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)
