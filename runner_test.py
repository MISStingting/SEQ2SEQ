import tensorflow as tf
from .runner import Runner
import os
import unittest

cur_dir = os.path.dirname(__file__)


class RunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.params = os.path.join(cur_dir, "config.yml")

    def _build_runner(self):
        return Runner(self.params)

    def test_train(self):
        runner = self._build_runner()
        runner.train()

    def test_eval(self):
        runner = self._build_runner()
        runner.eval()

    def test_train_and_eval(self):
        runner = self._build_runner()
        runner.train_and_eval()

    def test_infer(self):
        runner = self._build_runner()
        runner.infer()

    def test_export_model(self):
        runner = self._build_runner()
        runner.export_model()
