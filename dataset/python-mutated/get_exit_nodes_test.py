import unittest
from unittest.mock import MagicMock
from ..get_exit_nodes import ExitNodeGenerator
from .test_functions import __name__ as qualifier, all_functions

class GetExitNodesTest(unittest.TestCase):

    def test_compute_models(self) -> None:
        if False:
            i = 10
            return i + 15
        self.maxDiff = None
        sink = 'TaintSink[ReturnedToUser]'
        self.assertEqual([*map(str, ExitNodeGenerator(django_urls=MagicMock()).compute_models(all_functions))], [f'def {qualifier}.TestClass.methodA(self, x) -> {sink}: ...', f'def {qualifier}.TestClass.methodB(self, *args) -> {sink}: ...', f'def {qualifier}.testA() -> {sink}: ...', f'def {qualifier}.testB(x) -> {sink}: ...', f'def {qualifier}.testC(x) -> {sink}: ...', f'def {qualifier}.testD(x, *args) -> {sink}: ...', f'def {qualifier}.testE(x, **kwargs) -> {sink}: ...'])
        self.assertEqual([*map(str, ExitNodeGenerator(django_urls=MagicMock(), whitelisted_views=[f'{qualifier}.TestClass.methodA']).compute_models(all_functions))], [f'def {qualifier}.TestClass.methodB(self, *args) -> {sink}: ...', f'def {qualifier}.testA() -> {sink}: ...', f'def {qualifier}.testB(x) -> {sink}: ...', f'def {qualifier}.testC(x) -> {sink}: ...', f'def {qualifier}.testD(x, *args) -> {sink}: ...', f'def {qualifier}.testE(x, **kwargs) -> {sink}: ...'])