import time
import weakref
from golem.core.ordereddict import SizedOrderedDict
from golem.network.p2p.p2pservice import FORWARD_BATCH_SIZE
REMOVE_OLD_INTERVAL = 180
FORWARD_QUEUE_LEN = FORWARD_BATCH_SIZE * 10

class TaskConnectionsHelper(object):
    """ Keeps information about task connections that should be set with a help of p2p network """

    def __init__(self):
        if False:
            print('Hello World!')
        ' Create a new instance of task connection helper that keeps information about information\n        that has been passed and processed by a node.\n        '
        self.task_server = None
        self.conn_to_set = {}
        self.conn_to_set_queue = SizedOrderedDict(FORWARD_QUEUE_LEN)
        self.last_remove_old = time.time()
        self.remove_old_interval = REMOVE_OLD_INTERVAL
        self.conn_to_start = {}

    def is_new_conn_request(self, key_id, node_info):
        if False:
            while True:
                i = 10
        ' Check whether request for start connection with given conn_id has\n        occurred before (in a latest remove_old_interval)\n        :param key_id: public key of a node that is asked to start task session\n        with node from node info\n        :param Node node_info: node that asks for a task connection to be\n        started with him\n        :return bool: return False if connection with given id is known,\n        True otherwise\n        '
        id_tuple = (key_id, node_info.key)
        if id_tuple in self.conn_to_set:
            return False
        self.conn_to_set[id_tuple] = time.time()
        return True

    def want_to_start(self, conn_id, node_info, super_node_info):
        if False:
            while True:
                i = 10
        " Process request to start task session from this node to a node from node_info. If it's a first request\n        with given id pass information to task server, otherwise do nothing.\n        :param conn_id: connection id\n        :param Node node_info: node that requests task session with this node\n        :param Node|None super_node_info: information about supernode that has passed this information\n        "
        if conn_id in self.conn_to_start:
            return
        self.conn_to_start[conn_id] = (node_info, super_node_info, time.time())
        self.task_server.start_task_session(node_info, super_node_info, conn_id)

    def sync(self):
        if False:
            print('Hello World!')
        ' Remove old entries about connections '
        cur_time = time.time()
        if cur_time - self.last_remove_old <= self.remove_old_interval:
            return
        self.last_remove_old = cur_time
        self.conn_to_set = dict([y_z for y_z in self.conn_to_set.items() if cur_time - y_z[1] < self.remove_old_interval])
        self.conn_to_start = dict([y_z1 for y_z1 in self.conn_to_start.items() if cur_time - y_z1[1][2] < self.remove_old_interval])

    def cannot_start_task_session(self, conn_id):
        if False:
            return 10
        " Inform task server that cannot pass request with given conn id\n        :param conn_id: id of a connection that can't be established\n        "
        self.task_server.final_conn_failure(conn_id)

    def forward_queue_put(self, peer, key_id, node_info, conn_id, super_node_info):
        if False:
            return 10
        '\n        Append a forwarded request to the queue. Any existing request issued by\n        this particular sender (node_info.key) will be removed.\n\n        :param peer: peer session to send the message to\n        :param key_id: key id of a node that should open a task session\n        :param node_info: information about node that requested session\n        :param conn_id: connection id for reference\n        :param super_node_info: information about node with public ip that took\n        part in message transport\n        :return: None\n        '
        sender = node_info.key
        args = (key_id, node_info, conn_id, super_node_info)
        self.conn_to_set_queue.pop(sender, None)
        self.conn_to_set_queue[sender] = (weakref.ref(peer), args)

    def forward_queue_get(self, count=5):
        if False:
            return 10
        '\n        Get <count> forward requests from the queue.\n\n        :param count: number of requests to retrieve\n        :return: list of min(len(queue), count) queued requests\n        '
        entries = []
        try:
            for _ in range(count):
                (_, entry) = self.conn_to_set_queue.popitem(last=False)
                entries.append(entry)
        except KeyError:
            pass
        return entries