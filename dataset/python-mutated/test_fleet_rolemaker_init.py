"""Test cloud role maker."""
import os
import unittest
from paddle.distributed.fleet.base import role_maker

class TestPSCloudRoleMakerCase1(unittest.TestCase):
    """
    Test cases for PaddleCloudRoleMake Parameter Server.
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        os.environ['PADDLE_PSERVERS_IP_PORT_LIST'] = '127.0.0.1:4001,127.0.0.1:4002'

    def test_paddle_trainers_num(self):
        if False:
            return 10
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertRaises(ValueError, ro._generate_role)

class TestPSCloudRoleMakerCase2(unittest.TestCase):
    """
    Test cases for PaddleCloudRoleMake Parameter Server.
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        os.environ['PADDLE_PSERVERS_IP_PORT_LIST'] = '127.0.0.1:4001,127.0.0.1:4002'
        os.environ['PADDLE_TRAINERS_NUM'] = str(2)

    def test_training_role(self):
        if False:
            for i in range(10):
                print('nop')
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertRaises(ValueError, ro._generate_role)

class TestPSCloudRoleMakerCase3(unittest.TestCase):
    """
    Test cases for PaddleCloudRoleMake Parameter Server.
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        os.environ['PADDLE_PSERVERS_IP_PORT_LIST'] = '127.0.0.1:4001,127.0.0.1:4002'
        os.environ['PADDLE_TRAINERS_NUM'] = str(2)
        os.environ['TRAINING_ROLE'] = 'TRAINER'

    def test_trainer_id(self):
        if False:
            i = 10
            return i + 15
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertRaises(ValueError, ro._generate_role)

class TestPSCloudRoleMakerCase4(unittest.TestCase):
    """
    Test cases for PaddleCloudRoleMake Parameter Server.
    """

    def setUp(self):
        if False:
            return 10
        os.environ['PADDLE_PSERVERS_IP_PORT_LIST'] = '127.0.0.1:4001,127.0.0.1:4002'
        os.environ['PADDLE_TRAINERS_NUM'] = str(2)
        os.environ['TRAINING_ROLE'] = 'PSERVER'

    def test_ps_port(self):
        if False:
            for i in range(10):
                print('nop')
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertRaises(ValueError, ro._generate_role)

class TestPSCloudRoleMakerCase5(unittest.TestCase):
    """
    Test cases for PaddleCloudRoleMake Parameter Server.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        os.environ['PADDLE_PSERVERS_IP_PORT_LIST'] = '127.0.0.1:4001,127.0.0.1:4002'
        os.environ['PADDLE_TRAINERS_NUM'] = str(2)
        os.environ['TRAINING_ROLE'] = 'PSERVER'
        os.environ['PADDLE_PORT'] = str(4001)

    def test_ps_ip(self):
        if False:
            return 10
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertRaises(ValueError, ro._generate_role)

class TestPSCloudRoleMakerCase6(unittest.TestCase):
    """
    Test cases for PaddleCloudRoleMake Parameter Server.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        os.environ['PADDLE_PSERVERS_IP_PORT_LIST'] = '127.0.0.1:4001,127.0.0.1:4002'
        os.environ['PADDLE_HETER_TRAINER_IP_PORT_LIST'] = '127.0.0.1:4003,127.0.0.1:4004'
        os.environ['PADDLE_TRAINERS_NUM'] = str(2)
        os.environ['TRAINING_ROLE'] = 'HETER_TRAINER'

    def test_heter_port(self):
        if False:
            i = 10
            return i + 15
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertRaises(ValueError, ro._generate_role)

class TestPSCloudRoleMakerCase7(unittest.TestCase):
    """
    Test cases for PaddleCloudRoleMake Parameter Server.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        os.environ['PADDLE_PSERVERS_IP_PORT_LIST'] = '127.0.0.1:4001,127.0.0.1:4002'
        os.environ['PADDLE_HETER_TRAINER_IP_PORT_LIST'] = '127.0.0.1:4003,127.0.0.1:4004'
        os.environ['PADDLE_TRAINERS_NUM'] = str(2)
        os.environ['TRAINING_ROLE'] = 'HETER_TRAINER'
        os.environ['PADDLE_PORT'] = str(4003)

    def test_heter_ip(self):
        if False:
            for i in range(10):
                print('nop')
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertRaises(ValueError, ro._generate_role)
if __name__ == '__main__':
    unittest.main()