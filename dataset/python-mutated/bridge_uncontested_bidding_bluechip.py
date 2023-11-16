"""Two BlueChip bridge bots bid with each other.

The bot_cmd FLAG should contain a command-line to launch an external bot, e.g.
`Wbridge5 Autoconnect {port}`.

"""
import re
import socket
import subprocess
from absl import app
from absl import flags
import numpy as np
from open_spiel.python.bots import bluechip_bridge_uncontested_bidding
import pyspiel
FLAGS = flags.FLAGS
flags.DEFINE_float('timeout_secs', 60, 'Seconds to wait for bot to respond')
flags.DEFINE_integer('rng_seed', 1234, 'Seed to use to generate hands')
flags.DEFINE_integer('num_deals', 10, 'How many deals to play')
flags.DEFINE_string('bot_cmd', None, 'Command to launch the external bot; must include {port} which will be replaced by the port number to attach to.')

def _run_once(state, bots):
    if False:
        while True:
            i = 10
    'Plays bots with each other, returns terminal utility for each player.'
    for bot in bots:
        bot.restart_at(state)
    while not state.is_terminal():
        if state.is_chance_node():
            (outcomes, probs) = zip(*state.chance_outcomes())
            state.apply_action(np.random.choice(outcomes, p=probs))
        else:
            state.apply_action(bots[state.current_player()].step(state)[1])
    return state

def main(argv):
    if False:
        for i in range(10):
            print('nop')
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    game = pyspiel.load_game('bridge_uncontested_bidding', {'relative_scoring': True, 'rng_seed': FLAGS.rng_seed})
    bots = [bluechip_bridge_uncontested_bidding.BlueChipBridgeBot(game, 0, _WBridge5Client(FLAGS.bot_cmd)), bluechip_bridge_uncontested_bidding.BlueChipBridgeBot(game, 1, _WBridge5Client(FLAGS.bot_cmd))]
    results = []
    for i_deal in range(FLAGS.num_deals):
        state = _run_once(game.new_initial_state(), bots)
        print('Deal #{}; final state:\n{}'.format(i_deal, state))
        results.append(state.returns())
    stats = np.array(results)
    mean = np.mean(stats, axis=0)
    stderr = np.std(stats, axis=0, ddof=1) / np.sqrt(FLAGS.num_deals)
    print(u'Absolute score: {:+.1f}±{:.1f}'.format(mean[0], stderr[0]))
    print(u'Relative score: {:+.1f}±{:.1f}'.format(mean[1], stderr[1]))

class _WBridge5Client(object):
    """Manages the connection to a WBridge5 bot."""

    def __init__(self, command):
        if False:
            print('Hello World!')
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(('', 0))
        self.port = self.sock.getsockname()[1]
        self.sock.listen(1)
        self.process = None
        self.command = command.format(port=self.port)

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        if self.process is not None:
            self.process.kill()
        self.process = subprocess.Popen(self.command.split(' '))
        (self.conn, self.addr) = self.sock.accept()

    def read_line(self):
        if False:
            print('Hello World!')
        line = ''
        while True:
            self.conn.settimeout(FLAGS.timeout_secs)
            data = self.conn.recv(1024)
            if not data:
                raise EOFError('Connection closed')
            line += data.decode('ascii')
            if line.endswith('\n'):
                return re.sub('\\s+', ' ', line).strip()

    def send_line(self, line):
        if False:
            print('Hello World!')
        self.conn.send((line + '\r\n').encode('ascii'))
if __name__ == '__main__':
    app.run(main)