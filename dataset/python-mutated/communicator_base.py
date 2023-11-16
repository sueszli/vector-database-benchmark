from abc import ABCMeta
from abc import abstractmethod
import contextlib
import six
import warnings

class CommunicatorBase(six.with_metaclass(ABCMeta)):
    """Interface definition of all communicators.

    All communicators that have compatible set of methods with this
    class is supposed to work in ChainerMN's parallel computation
    implementation. The methods are named after MPI functions, such
    as ``bcast()`` came from ``MPI_Bcast()``.

    There are two types of methods: one that treats Python objects
    have ``_obj`` suffix.  The other has methods without any suffix
    and it handles ndarray and arrays filled with scaler values.  So
    the number of methods would be ::

        [send, recv, bcast, gather, allreduce] * [ '_obj', '']


    (with single exception ``alltoall``, ``multi_node_mean_grad``, ``split``
    and ``bcast_data`` so far). Also methods are supposed to be
    written in this order. All those methods must be implemented in
    its implementation class, or otherwise it cannot be instantiated
    in runtime.

    .. note:: As most implementation of ``_obj``-sufficed methods
      involves Python object pickling and unpickling, there is an
      implicit size limit.

    TODO(kuenishi): as of now no implementation class actually has
    ``allreduce`` method.

    """
    _configs = {}

    def __init__(self):
        if False:
            print('Hello World!')
        self._within_config_scope = False

    @property
    def rank(self):
        if False:
            while True:
                i = 10
        'Rank (process id in the cluster) of this process in integer.'
        raise NotImplementedError()

    @property
    def size(self):
        if False:
            print('Hello World!')
        'Number of processes of the cluster.'
        raise NotImplementedError()

    @property
    def intra_rank(self):
        if False:
            i = 10
            return i + 15
        'Intra rank (process id in the machine) of this process.'
        raise NotImplementedError()

    @property
    def intra_size(self):
        if False:
            return 10
        'Number of processes in the machine of this process.'
        raise NotImplementedError()

    @property
    def inter_rank(self):
        if False:
            i = 10
            return i + 15
        'The rank of this node in the cluster.'
        raise NotImplementedError()

    @property
    def inter_size(self):
        if False:
            return 10
        'Number of nodes that participates the cluster.'
        raise NotImplementedError()

    def set_config(self, name, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Set configurations(s) on/off\n\n        The usage of configurations depends on each communicator. See\n        :meth:`~chainermn.create_communicator` for available\n        configurations.\n\n        Args:\n            name (str):\n                Name of configuration to set.\n            value:\n                Give arbitrary object to set.\n            kwargs:\n                Arbitrary arguments depending on each configuration.\n\n        '
        raise ValueError('Unknown config: {}'.format(name))

    def get_config(self, name=None):
        if False:
            return 10
        'Get configuration value(s)\n\n        Args:\n            name (str):\n                Name of the configuration to get. If it is ``None``,\n                all config names and values are returned.\n\n        Returns:\n            Actual value of the configuration if it is on. ``None`` if it\n            is off. If ``None`` is given as ``name``, ``None`` or\n            dictionary of names and configuration values is returned.\n\n        '
        if name is not None:
            return self._configs[name]
        return self._configs

    @abstractmethod
    def split(self, color, key):
        if False:
            return 10
        'A function anologous to ``MPI_Comm_Split`` .\n\n        This method splits the inter MPI commnicator and return a wrapped\n        ChainerMN communicator.\n\n        Args:\n            color (int):\n                Index of new group. The process with the same color will be\n                assigned to the same group.\n            key (int):\n                Control of rank assignment. The process will be assigned\n                a rank in the new group ordered by the value of key.\n                If you do not care of the rank, you can just simply specify\n                the original rank.\n\n        Returns:\n            CommunicatorBase\n\n        '
        raise NotImplementedError()

    @abstractmethod
    def alltoall(self, xs):
        if False:
            print('Hello World!')
        'All-to-all implementation for ndarray\n\n        Args:\n            xs (tuple of numpy/cupy array)\n\n        Returns:\n            ys (tuple of numpy/cupy array):\n                Received arrays. The length of tuple equals to\n                the communicator size.\n\n        '
        raise NotImplementedError()

    @abstractmethod
    def send(self, data, dest, tag):
        if False:
            return 10
        'Sends an ndarray to destination\n\n        Receiver must invoke ``recv()`` to wait for the message.\n\n        Args:\n            data: data to be sent (tuple, list or raw numpy/cupy array)\n            dest (int): Rank of the destination process\n            tag (int): The tag to identify the message\n\n        '
        raise NotImplementedError()

    @abstractmethod
    def recv(self, source, tag):
        if False:
            return 10
        'Receives an ndarray from source.\n\n        To receive the message, sender must send the data.\n\n        Args:\n            source (int): Rank of the source process\n            tag (int): The tag to specifically receive the message\n\n        Returns:\n            The data sent from source process\n\n        '
        raise NotImplementedError()

    @abstractmethod
    def bcast(self, data, max_buf_len=None, root=0):
        if False:
            for i in range(10):
                print('nop')
        'Broadcasts an ndarray from root process to all processes\n\n        Args:\n            data (numpy/cupy array): for root process, the data to broadcast.\n                For non-root processes, this argument is ignored.\n            max_buf_len (int): Length of send buffer.\n            root (int): the process who has the data to broadcast.\n\n        Returns:\n            ys (numpy/cupy array) : The data sent from root process\n\n        '
        raise NotImplementedError()

    @abstractmethod
    def gather(self, data, root=0):
        if False:
            i = 10
            return i + 15
        'Gathers an ndarray from all processes to root process\n\n        Args:\n            data (ndarray, or scaler): for root process this is ignored. For\n                For non-root processes, the data to send to root process.\n            root (int): rank of the process who receives the data.\n\n        Returns:\n            For root process, the ndarray sent from non-root processes.\n            For non-root processes, what?\n\n        '
        raise NotImplementedError()

    @abstractmethod
    def allgather(self, x):
        if False:
            while True:
                i = 10
        'A primitive of inter-process all-gather communication.\n\n        This method tries to invoke all-gather communication within the\n        communicator. All processes in the communicator are expected to\n        invoke ``allgather()``. This method relies on mpi4py fast communication\n        optimized for numpy arrays, as well as ``send()`` and ``recv()``.\n\n        Note that this method can only handle the same shapes of data\n        over all processes, and cannot handle tuple data.\n\n        Args:\n            x (numpy/cupy array): Array to be gathered.\n\n        Returns:\n            ys (tuple of numpy/cupy array): Received arrays.\n        '
        raise NotImplementedError()

    @abstractmethod
    def allreduce(self, data):
        if False:
            for i in range(10):
                print('nop')
        "Allreduce operation among processes\n\n        Processes one of several aggregation operations using all data from\n        all processes and returns the result of the aggregation to all\n        processes.\n\n        TODO(kuenishi): add ``op`` argument once we find a use case\n        for operations other than 'SUM'.\n\n        Args:\n            data (ndarray): the data to aggregate among all nodes.\n\n        Returns:\n            Sum of all data from all processes.\n\n        "
        raise NotImplementedError()

    @abstractmethod
    def scatter(self, xs, root=0):
        if False:
            i = 10
            return i + 15
        'A primitive of inter-process scatter communication.\n\n        This method tries to invoke scatter communication within the\n        communicator. All processes in the communicator are expected to\n        invoke ``scatter()``.\n\n        Args:\n            xs (tuple of numpy/cupy array): Arrays to be scattered.\n            root (int): Rank of root process.\n        Returns:\n            ys (numpy/cupy array): Received arrays.\n        '
        raise NotImplementedError()

    def finalize(self):
        if False:
            while True:
                i = 10
        'Finalizes and cleans up internal resource.\n\n        The communicator SHALL NOT be used after calling this ``finalize()``.\n        The behaviour is undefined when calling ``finalize`` on the same\n        communicator multiple times.\n\n        '
        pass

    @abstractmethod
    def send_obj(self, obj, dest, tag):
        if False:
            print('Hello World!')
        'Sends an arbitrary Python object to destination with a tag.\n\n        Args:\n            obj: Arbitrary object to send to receiver.\n            dest (int): Rank number of receiver process (destination).\n            tag: tag to identify the message.\n\n        '
        raise NotImplementedError()

    @abstractmethod
    def recv_obj(self, source, tag):
        if False:
            while True:
                i = 10
        'Receives an arbitrary Python object from source process with a tag.\n\n        Args:\n           source (int): Rank number of sender process, to selectively receive\n               the object.\n           tag: tag to identify the message.\n\n        Returns:\n           an object sent from the source by ``send_obj``.\n\n        '
        raise NotImplementedError()

    @abstractmethod
    def bcast_obj(self, obj, max_buf_len=None, root=0):
        if False:
            i = 10
            return i + 15
        'Broadcasts an arbitrary object from root to all non-root processes.\n\n        Args:\n            obj: arbitrary object to broadcast to all other non-root processes.\n                Will be ignored at all non-root processes.\n            max_buf_len (int): max length of the send buffer\n            root (int): rank of the root processes who sends an object\n\n        Returns:\n            an object sent from the root process.\n\n        '
        raise NotImplementedError()

    @abstractmethod
    def gather_obj(self, obj, root=0):
        if False:
            return 10
        'Gathers arbitrary objects from all non-root processes to the root.\n\n        Args:\n            obj: arbtrary object to send to root process. Root process will\n                receive this argument included in returned list.\n            root (int): rank of the root node who receives all objects.\n\n        Returns:\n            A list of objects sent from all processes.\n\n        TODO(kuenishi): make sure the ordering of objects in the returned list.\n\n        '
        raise NotImplementedError()

    @abstractmethod
    def allreduce_obj(self, obj):
        if False:
            i = 10
            return i + 15
        "Apply a reduce operation to all objects and spread the result.\n\n        For example of integers and summation, equivalent local code is::\n\n          >>> from functools import reduce\n          >>> reduce(lambda x, y: x + y, [1, 2, 3, 4, 5])\n          15\n\n        The only operation currently supported is summation.\n\n        TODO(kuenishi): support other operations such as 'MAX', 'MIN'\n        and 'PROD' with ``op`` argument once we need any of them.\n\n        Args:\n           obj: An arbitrary object to apply reduce operation. Must have\n               corresponding operation method e.g. ``__plus__()``.\n\n        Returns:\n           The result of the operation applied to all objects.\n\n        "
        raise NotImplementedError()

    @abstractmethod
    def bcast_data(self, model):
        if False:
            print('Hello World!')
        'Broadcast Chainer model parameter data'
        raise NotImplementedError()

    def broadcast_data(self, model):
        if False:
            for i in range(10):
                print('nop')
        'Broadcast Chainer model parameter data\n\n        Left for backward compatibility, but ill be deprecated in\n        future version. Use ``bcast_data()`` method instad.\n\n        '
        self.bcast_data(model)

    @abstractmethod
    def multi_node_mean_grad(self, model, zero_fill=False):
        if False:
            while True:
                i = 10
        'mean Chainer model gradients.\n\n        Args:\n            link (~chainer.Link): Link object.\n            zero_fill: A knob to control whether to fill gradients of\n              initialized and unused Link (which is None internally) with\n              zero-valued array, because the all gradients must be an array\n              among processes for performing all-reduce, which might be an\n              array or None after backward computation. Gradients of\n              uninitialized Link are skipped. If it is False, gradients of\n              unused Link are just skipped.\n\n        '
        raise NotImplementedError()

    def allreduce_grad(self, model, zero_fill=False):
        if False:
            return 10
        'mean Chainer model gradients.\n\n        .. deprecated:: v7.0.0\n            This API is deprecated. Please use\n            :func:`~chainermn.CommunicatorBase.multi_node_mean_grad` instead.\n\n        Args:\n            link (~chainer.Link): Link object.\n            zero_fill: A knob to control whether to fill gradients of\n              initialized and unused Link (which is None internally) with\n              zero-valued array, because the all gradients must be an array\n              among processes for performing all-reduce, which might be an\n              array or None after backward computation. Gradients of\n              uninitialized Link are skipped. If it is False, gradients of\n              unused Link are just skipped.\n\n        '
        warnings.warn('allreduce_grad() is deprecated.', DeprecationWarning)
        self.multi_node_mean_grad(model, zero_fill)

    @property
    def within_config_scope(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'True if the current code is inside of an initialization scope.\n\n        See :meth:`init_scope` for the details of the initialization scope.\n\n        '
        return getattr(self, '_within_config_scope', False)

    @contextlib.contextmanager
    def config_scope(self):
        if False:
            print('Hello World!')
        'Creates an configuration scope.\n\n        '
        old_flag = self.within_config_scope
        self._within_config_scope = True
        try:
            yield
        finally:
            self._within_config_scope = old_flag

    def __setattr__(self, name, value):
        if False:
            while True:
                i = 10
        if self.within_config_scope:
            self._configs[name] = value
        super(CommunicatorBase, self).__setattr__(name, value)