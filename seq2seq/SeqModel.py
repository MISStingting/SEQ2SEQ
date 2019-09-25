import tensorflow as tf
from tensorflow.contrib import rnn
from .dataset import data_util
from tensorflow.python.layers import core
from tensorflow.contrib import lookup
from SEQ2SEQ.seq2seq.utils import utils


# tf.enable_eager_execution()


class Seq2SeqForAddress(object):
    @staticmethod
    def input_fn(params, mode):
        return data_util.get_dataset(params, mode)

    def model_fn(self, features, labels, params, mode):
        # embedding input and target sequence
        vocab_size = params["vocab_size"]
        num_units = params["num_units"]
        input_keep_prob = params["input_keep_prob"]
        layer_size = params["encode_layzer_size"]
        in_seq = features["input"]
        in_seq_length = features["input_length"]
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            out_seq = labels["output_out"]
            out_seq_length = labels["output_length"]
        else:
            out_seq = None
            out_seq_length = None
        table = lookup.index_table_from_file(params["vocabs_labels_file"], default_value=0)
        features_ids = table.lookup(in_seq)
        with tf.variable_scope("embedding"):
            embedding = tf.Variable(tf.random.uniform([vocab_size, num_units], -1.0, 1.0), dtype=tf.float32,
                                    name='word_embedding')
            # embedding = tf.get_variable(name='embed', shape=[vocab_size, num_units])
            embed_input = tf.nn.embedding_lookup(params=embedding, ids=features_ids, name="embed_input")
            # encode and decode
            bi_layer_size = int(layer_size / 2)
            encoder_cell_fw = self.get_layered_cell(bi_layer_size, num_units, input_keep_prob)
            encoder_cell_bw = self.get_layered_cell(bi_layer_size, num_units, input_keep_prob)
            # sequence_length 应该是序列真实的长度
            bi_encoder_output, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell_fw,
                                                                                  cell_bw=encoder_cell_bw,
                                                                                  inputs=embed_input,
                                                                                  sequence_length=in_seq_length,
                                                                                  dtype=embed_input.dtype,
                                                                                  time_major=False)
            # concat encode output and state
            encoder_output = tf.concat(bi_encoder_output, -1)
            encoder_state = []
            for layers_id in range(bi_layer_size):
                encoder_state.append(bi_encoder_state[0][layers_id])
                encoder_state.append(bi_encoder_state[1][layers_id])
            encoder_state = tuple(encoder_state)
            decoder_cell = self.attention_decoder_cell(encoder_output, in_seq_length, num_units, layer_size,
                                                       input_keep_prob)
            batch_size = tf.shape(in_seq_length)[0]
            init_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)
            # str2int
            labels_ids = None
            if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                labels_ids = table.lookup(out_seq)
                embed_target = tf.nn.embedding_lookup(embedding, labels_ids, name="embed_target")
                helper = tf.contrib.seq2seq.TrainingHelper(embed_target, out_seq_length, time_major=False)
            else:
                # TODO: start tokens and end tokens are hard code
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, tf.fill([batch_size], 0), 1)
            predict_output_layer = core.Dense(vocab_size, use_bias=False)
            basic_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper, initial_state=init_state,
                                                            output_layer=predict_output_layer)
            # 执行动态解码 (final_outputs, final_state, final_sequence_lengths)
            final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=basic_decoder,
                maximum_iterations=50)
            loss = None
            train_op = None
            predictions = None

            if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                outputs = final_outputs.rnn_output
                loss = self.compute_loss(outputs, labels_ids, out_seq_length)
                trainable_params = tf.trainable_variables()
                global_step = tf.train.get_global_step()
                clipped_gradients, _ = tf.clip_by_global_norm(tf.gradients(loss, trainable_params), 0.8)
                train_op = tf.train.AdamOptimizer(learning_rate=0.001).apply_gradients(
                    zip(clipped_gradients, trainable_params), global_step=global_step)
                # train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
                # tf.summary.scalar('loss', loss)
                tf.add_to_collection("loss", loss)
            else:
                outputs = final_outputs.sample_id
                predict_ids = tf.cast(outputs, dtype=tf.int64)
                mask = tf.sequence_mask(in_seq_length, maxlen=tf.shape(predict_ids)[1])
                real_ids = predict_ids * tf.cast(mask, dtype=tf.int64)
                index_to_string = lookup.index_to_string_table_from_file(params["vocabs_labels_file"])
                predict_labels = index_to_string.lookup(real_ids)
                predictions = {
                    "predict_labels": predict_labels,
                    "predict_ids": real_ids,
                }
                tf.add_to_collection("predictions", predictions)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)

    @staticmethod
    def get_layered_cell(bi_layer_size, num_units, input_keep_prob, output_keep_prob=1.0):
        cells = []
        for i in range(bi_layer_size):
            cells.append(rnn.DropoutWrapper(rnn.BasicLSTMCell(num_units=num_units), input_keep_prob, output_keep_prob))
        return rnn.MultiRNNCell(cells=cells)

    def attention_decoder_cell(self, encoder_output, in_seq_len, num_units, layer_size, input_keep_prob):
        # num_units：查询机制的深度。
        # memory：要查询的内存; 通常是RNN编码器的输出。应该塑造这种张量[batch_size, max_time, ...]。
        # memory_sequence_length:(可选）内存中批处理条目的序列长度。如果提供，则对于超过相应序列长度的值，用零掩码存储器张量行。
        # normalize：Python布尔值。是否规范能源项。
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=num_units, memory=encoder_output,
                                                                   memory_sequence_length=in_seq_len, normalize=True,
                                                                   name='BahdanauAttention')
        # attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=num_units, memory=encoder_output,
        #                                                         memory_sequence_length=in_seq_len,
        #                                                         scale=True, name='LuongAttention')
        cell = self.get_layered_cell(layer_size, num_units=num_units, input_keep_prob=input_keep_prob)
        return tf.contrib.seq2seq.AttentionWrapper(cell=cell, attention_mechanism=attention_mechanism,
                                                   attention_layer_size=num_units)

    @staticmethod
    def compute_loss(outputs, labels_ids, out_seq_len):
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=labels_ids)
        # out_seq_len 不定长的长度，序列的真实长度,计算的是真实数据的损失，不包括补充位置的损失
        loss_mask = tf.sequence_mask(out_seq_len, tf.shape(labels_ids)[1])
        cost = cost * tf.cast(loss_mask, dtype=tf.float32)
        return tf.reduce_sum(cost) / tf.cast(tf.shape(labels_ids)[0], dtype=tf.float32)
