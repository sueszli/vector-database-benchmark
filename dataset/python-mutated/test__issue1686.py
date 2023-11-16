"""
Tests for https://github.com/gevent/gevent/issues/1686
which is about destroying a hub when there are active
callbacks or IO in operation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import unittest
from gevent import testing as greentest

@greentest.skipOnWindows('Uses os.fork')
class TestDestroyInChildWithActiveSpawn(unittest.TestCase):

    def test(self):
        if False:
            i = 10
            return i + 15
        from time import sleep as hang
        from gevent import get_hub
        from gevent import spawn
        from gevent.socket import wait_read
        from gevent.os import nb_read
        from gevent.os import nb_write
        from gevent.os import make_nonblocking
        from gevent.os import fork
        from gevent.os import waitpid
        (pipe_read_fd, pipe_write_fd) = os.pipe()
        make_nonblocking(pipe_read_fd)
        make_nonblocking(pipe_write_fd)
        run = []

        def reader():
            if False:
                return 10
            run.append(1)
            return nb_read(pipe_read_fd, 4096)
        DATA = b'test'
        nb_write(pipe_write_fd, DATA)
        wait_read(pipe_read_fd)
        reader = spawn(reader)
        hub = get_hub()
        pid = fork()
        if pid == 0:
            hub.destroy(destroy_loop=True)
            self.assertFalse(run)
            os._exit(0)
            return
        hang(0.5)
        wait_child_result = waitpid(pid, 0)
        self.assertEqual(wait_child_result, (pid, 0))
        data = reader.get()
        self.assertEqual(run, [1])
        self.assertEqual(data, DATA)
if __name__ == '__main__':
    greentest.main()