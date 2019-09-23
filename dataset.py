#!encoding=utf8
import tensorflow as tf
import yaml
import os

# tf.compat.v1.enable_eager_execution()
cur_dir = os.path.dirname(__file__)


class Dataset(object):

    def get_dataset(self, params, mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
            features_path = params["train_features_file"]
            labels_path = params["train_labels_file"]
        elif mode == tf.estimator.ModeKeys.EVAL:
            features_path = params["eval_features_file"]
            labels_path = params["eval_labels_file"]

        elif mode == tf.estimator.ModeKeys.PREDICT:
            features_path = params["test_features_file"]
            labels_path = params["test_labels_file"]
        else:
            raise ValueError("wrong mode!!!")

        features_dataset, labels_dataset = self._load_dataset(features_path, labels_path, mode)
        if mode == tf.estimator.ModeKeys.PREDICT:
            dataset = features_dataset.map(lambda x: tf.string_split([x]).values)
            dataset = dataset.shuffle(buffer_size=params["buffer_size"],
                                      reshuffle_each_iteration=params["reshuffle_each_iteration"])
            dataset = dataset.padded_batch(batch_size=params["batch_size"], padded_shapes=(tf.TensorShape([None])),
                                           padding_values=(tf.constant("<PAD>", dtype=tf.string)))
            iterator = dataset.make_one_shot_iterator()
            batch_features = iterator.get_next()
            input_length = tf.map_fn(
                fn=lambda x: tf.shape(tf.string_split(x, sep="<PAD>").values)[0],
                elems=batch_features,
                dtype=tf.int32)
            features = {
                "input": batch_features,
                "input_length": input_length
            }
            labels = None
            # print("input_len:", input_length)
            # print("input：", batch_features)
        else:
            dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))
            dataset = dataset.map(lambda x, y: (tf.string_split([x]).values, tf.string_split([y]).values))
            dataset = dataset.repeat(params["repeat"]).shuffle(buffer_size=params["buffer_size"],
                                                               reshuffle_each_iteration=params[
                                                                   "reshuffle_each_iteration"])
            dataset = dataset.padded_batch(batch_size=params["batch_size"],
                                           padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None])),
                                           padding_values=(
                                               tf.constant("<PAD>", dtype=tf.string),
                                               tf.constant("<UNK>", dtype=tf.string)))
            iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
            batch_features, batch_labels = iterator.get_next()
            input_length = tf.map_fn(fn=lambda x: tf.shape(tf.string_split(x, delimiter="<PAD>").values)[0],
                                     elems=batch_features, dtype=tf.int32)
            output_length = tf.map_fn(fn=lambda y: tf.shape(tf.string_split(y, delimiter="<UNK>").values)[0],
                                      elems=batch_labels, dtype=tf.int32)
            features = {
                "input": batch_features,
                "input_length": input_length
            }
            labels = {
                "output": batch_labels,
                "output_length": output_length
            }
            # print("input_len:", input_length)
            # print("output_len:", output_length)
            # print("input：", batch_features)
            # print("output:", batch_labels)

        return features, labels

    @staticmethod
    def _load_dataset(features_path, labels_path, mode):
        '''
        :param mode:
        :return:
        '''
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            features_dataset = tf.data.TextLineDataset(filenames=features_path)
            labels_dataset = tf.data.TextLineDataset(filenames=labels_path)

            return features_dataset, labels_dataset
        elif mode == tf.estimator.ModeKeys.PREDICT:
            features_dataset = tf.data.TextLineDataset(filenames=features_path)
            return features_dataset, None


data_util = Dataset()

if __name__ == '__main__':
    config_file = os.path.join(cur_dir, "config.yml")
    f = open(config_file, 'rt', encoding="utf8", buffering=8142)
    params = yaml.load(stream=f.read(), Loader=yaml.FullLoader)
    data_util.get_dataset(params, mode=tf.estimator.ModeKeys.EVAL)
