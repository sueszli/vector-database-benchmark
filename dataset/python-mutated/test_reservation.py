import os
import threading
import time
import unittest
from tensorflowonspark import util
from tensorflowonspark.reservation import Reservations, Server, Client
from unittest import mock

class ReservationTest(unittest.TestCase):

    def test_reservation_class(self):
        if False:
            return 10
        'Test core reservation class, expecting 2 reservations'
        r = Reservations(2)
        self.assertFalse(r.done())
        r.add({'node': 1})
        self.assertFalse(r.done())
        self.assertEqual(r.remaining(), 1)
        r.add({'node': 2})
        self.assertTrue(r.done())
        self.assertEqual(r.remaining(), 0)
        reservations = r.get()
        self.assertEqual(len(reservations), 2)

    def test_reservation_server(self):
        if False:
            print('Hello World!')
        'Test reservation server, expecting 1 reservation'
        s = Server(1)
        addr = s.start()
        c = Client(addr)
        resp = c.register({'node': 1})
        self.assertEqual(resp, 'OK')
        reservations = c.get_reservations()
        self.assertEqual(len(reservations), 1)
        reservations = c.await_reservations()
        self.assertEqual(len(reservations), 1)
        c.request_stop()
        time.sleep(1)
        self.assertEqual(s.done, True)

    def test_reservation_environment_exists_get_server_ip_return_environment_value(self):
        if False:
            for i in range(10):
                print('nop')
        tfos_server = Server(5)
        with mock.patch.dict(os.environ, {'TFOS_SERVER_HOST': 'my_host_ip'}):
            assert tfos_server.get_server_ip() == 'my_host_ip'

    def test_reservation_environment_not_exists_get_server_ip_return_actual_host_ip(self):
        if False:
            return 10
        tfos_server = Server(5)
        assert tfos_server.get_server_ip() == util.get_ip_address()

    def test_reservation_environment_exists_start_listening_socket_return_socket_listening_to_environment_port_value(self):
        if False:
            for i in range(10):
                print('nop')
        tfos_server = Server(1)
        with mock.patch.dict(os.environ, {'TFOS_SERVER_PORT': '9999'}):
            assert tfos_server.start_listening_socket().getsockname()[1] == 9999

    def test_reservation_environment_not_exists_start_listening_socket_return_socket(self):
        if False:
            while True:
                i = 10
        tfos_server = Server(1)
        print(tfos_server.start_listening_socket().getsockname()[1])
        assert type(tfos_server.start_listening_socket().getsockname()[1]) == int

    def test_reservation_environment_exists_port_spec(self):
        if False:
            while True:
                i = 10
        tfos_server = Server(1)
        with mock.patch.dict(os.environ, {'TFOS_SERVER_PORT': '9999'}):
            self.assertEqual(tfos_server.get_server_ports(), [9999])
        with mock.patch.dict(os.environ, {'TFOS_SERVER_PORT': '9997-9999'}):
            self.assertEqual(tfos_server.get_server_ports(), [9997, 9998, 9999])

    def test_reservation_environment_exists_start_listening_socket_return_socket_listening_to_environment_port_range(self):
        if False:
            print('Hello World!')
        tfos_server1 = Server(1)
        tfos_server2 = Server(1)
        tfos_server3 = Server(1)
        with mock.patch.dict(os.environ, {'TFOS_SERVER_PORT': '9998-9999'}):
            s1 = tfos_server1.start_listening_socket()
            self.assertEqual(s1.getsockname()[1], 9998)
            s2 = tfos_server2.start_listening_socket()
            self.assertEqual(s2.getsockname()[1], 9999)
            with self.assertRaises(Exception):
                tfos_server3.start_listening_socket()
        tfos_server1.stop()
        tfos_server2.stop()

    def test_reservation_server_multi(self):
        if False:
            i = 10
            return i + 15
        'Test reservation server, expecting multiple reservations'
        num_clients = 4
        s = Server(num_clients)
        addr = s.start()

        def reserve(num):
            if False:
                print('Hello World!')
            c = Client(addr)
            resp = c.register({'node': num})
            self.assertEqual(resp, 'OK')
            c.await_reservations()
            c.close()
        threads = [None] * num_clients
        for i in range(num_clients):
            threads[i] = threading.Thread(target=reserve, args=(i,))
            threads[i].start()
        for i in range(num_clients):
            threads[i].join()
        print('all done')
        c = Client(addr)
        reservations = c.get_reservations()
        self.assertEqual(len(reservations), num_clients)
        c.request_stop()
        time.sleep(1)
        self.assertEqual(s.done, True)
if __name__ == '__main__':
    unittest.main()