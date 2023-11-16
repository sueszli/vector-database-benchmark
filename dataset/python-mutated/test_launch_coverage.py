import unittest
from argparse import REMAINDER, ArgumentParser
from paddle.distributed.fleet.launch_utils import find_free_ports
from paddle.distributed.utils.launch_utils import _print_arguments, get_cluster_from_args, get_gpus

def _parse_args():
    if False:
        return 10
    parser = ArgumentParser(description='start paddle training using multi-process mode.\nNOTE: your train program ***must*** run as distributed nccl2 mode,\nsee: http://www.paddlepaddle.org/documentation/docs/zh/1.6/user_guides/howto/training/cluster_howto.html#permalink-8--nccl2-\nAnd your train program must read environment variables below in order to let different\nprocess init properly:\nFLAGS_selected_gpus\nPADDLE_TRAINER_ID\nPADDLE_CURRENT_ENDPOINT\nPADDLE_TRAINERS_NUM\nPADDLE_TRAINER_ENDPOINTS\nPOD_IP (current node ip address, not needed for local training)\n')
    parser.add_argument('--cluster_node_ips', type=str, default='127.0.0.1', help='Paddle cluster nodes ips, such as 192.168.0.16,192.168.0.17..')
    parser.add_argument('--node_ip', type=str, default='127.0.0.1', help='The current node ip. ')
    parser.add_argument('--use_paddlecloud', action='store_true', help='wheter to use paddlecloud platform to run your multi-process job. If false, no need to set this argument.')
    parser.add_argument('--started_port', type=int, default=None, help="The trainer's started port on a single node")
    parser.add_argument('--print_config', type=bool, default=True, help='Print the config or not')
    parser.add_argument('--selected_gpus', type=str, default=None, help="It's for gpu training and the training process will run on the selected_gpus,each process is bound to a single GPU. And if it's not set, this module will use all the gpu cards for training.")
    parser.add_argument('--log_level', type=int, default=20, help='Logging level, default is logging.INFO')
    parser.add_argument('--log_dir', type=str, help="The path for each process's log.If it's not set, the log will printed to default pipe.")
    parser.add_argument('training_script', type=str, help='The full path to the single GPU training program/script to be launched in parallel, followed by all the arguments for the training script')
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()

class TestCoverage(unittest.TestCase):

    def test_gpus(self):
        if False:
            return 10
        args = _parse_args()
        if args.print_config:
            _print_arguments(args)
        gpus = get_gpus(None)
        args.use_paddlecloud = True
        (cluster, pod) = get_cluster_from_args(args, '0')

    def test_find_free_ports(self):
        if False:
            print('Hello World!')
        find_free_ports(2)
if __name__ == '__main__':
    unittest.main()