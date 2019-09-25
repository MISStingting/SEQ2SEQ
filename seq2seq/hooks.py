import tensorflow as tf
from SEQ2SEQ.seq2seq.utils import utils


class EvalLoggerHook(tf.train.SessionRunHook):
    """Logs loss and runtime."""

    def __init__(self):
        self.global_steps = None
        self.loss = None

    def begin(self):
        print("eval:begining :----------------------->\n")
        self.global_steps = tf.train.get_global_step()
        self.loss = utils.get_from_collection("loss")

    def after_create_session(self, session, coord):
        super().after_create_session(session, coord)

    def before_run(self, run_context):
        if not self.loss:
            raise ValueError("Model does not define loss.")
        if not self.global_steps:
            raise ValueError("Not created global steps.")
        return tf.train.SessionRunArgs([self.global_steps, self.loss])

    def after_run(self, run_context, run_values):
        global_steps, loss = run_values.results
        step_loss = "global_step ={}.and loss ={}".format(global_steps, loss)
        tf.logging.info(step_loss)

    def end(self, session):
        tf.logging.info("ONE EPOCH EVAL END!")


class InitHook(tf.train.SessionRunHook):

    def after_create_session(self, session, coord):
        # iterator_init_op = tf.get_collection(collection_utils.ITERATOR)
        tables_init_op = tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS)
        variables_init_op = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        session.run(tables_init_op)
        session.run(variables_init_op)


class CountParamsHook(tf.train.SessionRunHook):
    """Logs the number of trainable parameters."""

    def begin(self):
        total = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            count = 1
            for dim in shape:
                count *= dim.value
            total += count
        tf.logging.info("Number of trainable parameters: %d", total)


class SaveEvaluationPredictionsHook(tf.train.SessionRunHook):
    """Do evaluation and save prediction results to file."""

    def __init__(self,
                 output_file,
                 eos="</s>",
                 subword_option=""):
        self.eos = eos
        self.subword_option = subword_option
        self.predictions = None
        self.output_file = output_file
        self.global_steps = None
        self.output_path = None

    def begin(self):
        self.predictions = tf.get_collection(key="predictions")
        self.global_steps = tf.train.get_global_step()

    def before_run(self, run_context):
        if not self.predictions:
            raise ValueError("Model does not define predictions.")
        if not self.global_steps:
            raise ValueError("Not created global steps.")
        return tf.train.SessionRunArgs([self.predictions, self.global_steps])

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):
        predictions, global_steps = run_values.results
        predictions = utils.get_predictions(predictions, self.eos, self.subword_option)

        self.output_path = "{}.{}".format(self.output_file, global_steps)

        with open(self.output_path, mode="a", encoding="utf8") as f:
            if isinstance(predictions, str):
                f.write(predictions + "\n")
            elif isinstance(predictions, list):
                for p in predictions:
                    f.write(p + "\n")

    def end(self, session):
        tf.logging.info("Evaluation predictions saved to %s" % self.output_path)
