import os
import sys
import unittest
import subprocess

class TestLightning(unittest.TestCase):

    @staticmethod
    def run_shell(args, timeout=30):
        if False:
            while True:
                i = 10
        process = subprocess.Popen(['electrum/tests/regtest/regtest.sh'] + args, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, universal_newlines=True)
        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)
            sys.stdout.flush()
        process.wait(timeout=timeout)
        process.stdout.close()
        assert process.returncode == 0

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        test_name = self.id().split('.')[-1]
        sys.stdout.write('***** %s ******\n' % test_name)
        for (agent, config_options) in self.agents.items():
            self.run_shell(['init', agent])
            for (k, v) in config_options.items():
                self.run_shell(['setconfig', agent, k, v])
        self.run_shell(['new_block'])
        for agent in self.agents:
            self.run_shell(['start', agent])

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        for agent in self.agents:
            self.run_shell(['stop', agent])

class TestUnixSockets(TestLightning):
    agents = {}

    def test_unixsockets(self):
        if False:
            while True:
                i = 10
        self.run_shell(['unixsockets'])

class TestLightningAB(TestLightning):
    agents = {'alice': {}, 'bob': {'lightning_listen': 'localhost:9735'}}

    def test_collaborative_close(self):
        if False:
            i = 10
            return i + 15
        self.run_shell(['collaborative_close'])

    def test_backup(self):
        if False:
            while True:
                i = 10
        self.run_shell(['backup'])

    def test_backup_local_forceclose(self):
        if False:
            while True:
                i = 10
        self.run_shell(['backup_local_forceclose'])

    def test_breach(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_shell(['breach'])

    def test_extract_preimage(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_shell(['extract_preimage'])

    def test_redeem_htlcs(self):
        if False:
            return 10
        self.run_shell(['redeem_htlcs'])

    def test_breach_with_unspent_htlc(self):
        if False:
            i = 10
            return i + 15
        self.run_shell(['breach_with_unspent_htlc'])

    def test_breach_with_spent_htlc(self):
        if False:
            i = 10
            return i + 15
        self.run_shell(['breach_with_spent_htlc'])

class TestLightningSwapserver(TestLightning):
    agents = {'alice': {}, 'bob': {'lightning_listen': 'localhost:9735', 'use_swapserver': 'true'}}

    def test_swapserver_success(self):
        if False:
            i = 10
            return i + 15
        self.run_shell(['swapserver_success'])

    def test_swapserver_refund(self):
        if False:
            i = 10
            return i + 15
        self.run_shell(['swapserver_refund'])

class TestLightningWatchtower(TestLightning):
    agents = {'alice': {}, 'bob': {'lightning_listen': 'localhost:9735', 'watchtower_url': 'http://wtuser:wtpassword@127.0.0.1:12345'}, 'carol': {'run_watchtower': 'true', 'watchtower_user': 'wtuser', 'watchtower_password': 'wtpassword', 'watchtower_port': '12345'}}

    def test_watchtower(self):
        if False:
            print('Hello World!')
        self.run_shell(['watchtower'])

class TestLightningJIT(TestLightning):
    agents = {'alice': {'accept_zeroconf_channels': 'true'}, 'bob': {'lightning_listen': 'localhost:9735', 'lightning_forward_payments': 'true', 'accept_zeroconf_channels': 'true'}, 'carol': {}}

    def test_just_in_time(self):
        if False:
            print('Hello World!')
        self.run_shell(['just_in_time'])

class TestLightningJITTrampoline(TestLightningJIT):
    agents = {'alice': {'use_gossip': 'false', 'accept_zeroconf_channels': 'true'}, 'bob': {'lightning_listen': 'localhost:9735', 'lightning_forward_payments': 'true', 'lightning_forward_trampoline_payments': 'true', 'accept_zeroconf_channels': 'true'}, 'carol': {'use_gossip': 'false'}}