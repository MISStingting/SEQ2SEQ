import tensorflow as tf
import yaml
from .SeqModel import Seq2SeqForAddress
from .train_hooks import EvalLoggerHook, TrainLoggerHook
import os
import argparse
import numpy as np

# tf.compat.v1.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)


# 训练具体实现

class Runner(object):
    '''
    读取配置参数
    '''

    def __init__(self, params_file):
        self._config = self.load_config_file(params_file)
        self.model = Seq2SeqForAddress()
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=tf.compat.v1.GPUOptions(
                allow_growth=False
            )
        )
        run_config = tf.estimator.RunConfig(model_dir=self._config["model_dir"],
                                            tf_random_seed=self._config["random_seed"],
                                            session_config=session_config,
                                            save_summary_steps=self._config["save_summary_steps"],
                                            keep_checkpoint_max=self._config["keep_checkpoint_max"],
                                            log_step_count_steps=self._config["log_step_count_steps"])
        self.estimator = tf.estimator.Estimator(
            model_fn=self.model.model_fn,
            model_dir=self._config["model_dir"],
            config=run_config,
            params=self._config)

    @staticmethod
    def load_config_file(params_file):
        with open(params_file, 'rt', encoding="utf8", buffering=8142) as f:
            params = yaml.load(stream=f.read(), Loader=yaml.FullLoader)
        return params

    def train(self):
        input_fn = lambda: self.model.input_fn(params=self._config, mode=tf.estimator.ModeKeys.TRAIN)
        train_hooks = [TrainLoggerHook()]
        train_spec = tf.estimator.TrainSpec(
            input_fn=input_fn,
            hooks=train_hooks,
            max_steps=self._config["max_steps"]
        )
        return self.estimator.train(input_fn=train_spec.input_fn, hooks=train_spec.hooks,
                                    max_steps=train_spec.max_steps)

    def eval(self):
        input_fn = lambda: self.model.input_fn(params=self._config, mode=tf.estimator.ModeKeys.EVAL)
        eval_spec = tf.estimator.EvalSpec(input_fn=input_fn, steps=self._config["eval_steps"], hooks=[EvalLoggerHook()])
        return self.estimator.evaluate(input_fn=eval_spec.input_fn,
                                       steps=eval_spec.steps,
                                       hooks=eval_spec.hooks,
                                       checkpoint_path=None)

    def train_and_eval(self):
        train_input_fn = lambda: self.model.input_fn(params=self._config, mode=tf.estimator.ModeKeys.TRAIN)
        eval_input_fn = lambda: self.model.input_fn(params=self._config, mode=tf.estimator.ModeKeys.EVAL)
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=self._config["max_steps"],
                                            hooks=[])
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, hooks=[EvalLoggerHook()],
                                          steps=self._config["eval_steps"])
        return tf.estimator.train_and_evaluate(estimator=self.estimator,
                                               train_spec=train_spec,
                                               eval_spec=eval_spec)

    def export_model(self):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir=self._config["model_dir"])
        export_dir = self.estimator.export_saved_model(
            export_dir_base=os.path.join(self._config["model_dir"], "export"),
            checkpoint_path=checkpoint_path,
            # assets_extra="my_asset_file.txt",
            # as_text=True,
            serving_input_receiver_fn=self.serving_input_receiver_fn)
        print("export_dir:", export_dir)
        return export_dir

    @staticmethod
    def serving_input_receiver_fn():
        """An input receiver that expects a serialized tf.Example."""
        receive_tensors = {
            "input": tf.compat.v1.placeholder(dtype=tf.string, shape=[None, None]),
            "input_length": tf.compat.v1.placeholder(dtype=tf.int32, shape=(None,))
        }
        features = receive_tensors.copy()
        return tf.estimator.export.ServingInputReceiver(
            features=features,
            receiver_tensors=receive_tensors
        )

    def infer(self):
        # 推断模型效果
        predict_input_fn = lambda: self.model.input_fn(params=self._config, mode=tf.estimator.ModeKeys.PREDICT)
        # checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir=self._config["model_dir"])
        predictions = self.estimator.predict(input_fn=predict_input_fn, hooks=[])
        for i, e in enumerate(predictions):
            ids = np.trim_zeros(e["predict_ids"])
            lables = e["predict_labels"][:ids.shape[0]]
            print("ids:   ", ids)
            print("labels:", [e.decode() for e in lables])
            print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_file", type=str, default="config.yml",
                        required=True, help="The params configuration in YML format.")
    parser.add_argument("--mode", type=str, default="train_and_eval",
                        choices=["train", "eval",
                                 "train_and_eval", "infer", "export"],
                        help="run mode.")

    args, _ = parser.parse_known_args()
    config_file = args.params_file
    runner = Runner(config_file)
    tf.logging.set_verbosity(tf.logging.INFO)

    mode = args.mode
    if mode == "train":
        runner.train()
    elif mode == "eval":
        runner.eval()
    elif mode == "train_and_eval":
        runner.train_and_eval()
    elif mode == "infer":
        runner.infer()
    elif mode == "export":
        runner.export_model()
    else:
        raise ValueError("Unknown mode: %s" % mode)
