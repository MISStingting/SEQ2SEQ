import tensorflow as tf
import time


class TrainLoggerHook(tf.train.SessionRunHook):
    """Logs loss and runtime."""

    def begin(self):
        super().begin()
        print("train :begin:----------------------->\n")

    def after_create_session(self, session, coord):
        super().after_create_session(session, coord)

    def before_run(self, run_context):
        super().before_run(run_context)

    def after_run(self, run_context, run_values):
        super().after_run(run_context, run_values)

    def end(self, session):
        super().end(session)


class EvalLoggerHook(tf.train.SessionRunHook):
    """Logs loss and runtime."""

    def begin(self):
        super().begin()
        print("eval:begining :----------------------->\n")

    def after_create_session(self, session, coord):
        super().after_create_session(session, coord)

    def before_run(self, run_context):
        super().before_run(run_context)

    def after_run(self, run_context, run_values):
        super().after_run(run_context, run_values)

    def end(self, session):
        super().end(session)
