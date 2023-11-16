import os
import subprocess
import time
from libqtile.command_client import InteractiveCommandClient
from libqtile.command_interface import IPCCommandInterface
from libqtile.ipc import Client as IPCClient
from libqtile.ipc import find_sockfile

class Client:
    COLORS = ['#44cc44', '#cc44cc', '#4444cc', '#cccc44', '#44cccc', '#cccccc', '#777777', '#ffa500', '#333333']

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.color = 0
        self.client = InteractiveCommandClient(IPCCommandInterface(IPCClient(find_sockfile())))

    def current_group(self):
        if False:
            while True:
                i = 10
        return self.client.group[self.client.group.info().get('name')]

    def switch_to_group(self, group):
        if False:
            return 10
        if isinstance(group, str):
            self.client.group[group].toscreen()
        else:
            group.toscreen()

    def spawn_window(self, color=None):
        if False:
            return 10
        if color is None:
            color = self.color
            self.color += 1
        if isinstance(color, int):
            color = Client.COLORS[color]
        self.client.spawn("xterm +ls -hold -e printf '\\e]11;{}\x07'".format(color))

    def prepare_layout(self, layout, windows, commands=None):
        if False:
            for i in range(10):
                print('nop')
        self.client.group.setlayout(layout)
        for i in range(windows):
            self.spawn_window()
            time.sleep(0.05)
        if commands:
            for cmd in commands:
                self.run_layout_command(cmd)
                time.sleep(0.05)

    def clean_layout(self, commands=None):
        if False:
            for i in range(10):
                print('nop')
        if commands:
            for cmd in commands:
                self.run_layout_command(cmd)
                time.sleep(0.05)
        self.kill_group_windows()

    def run_layout_command(self, cmd):
        if False:
            while True:
                i = 10
        if cmd == 'spawn':
            self.spawn_window()
        else:
            getattr(self.client.layout, cmd)()

    def kill_group_windows(self):
        if False:
            for i in range(10):
                print('nop')
        while len(self.client.layout.info().get('clients')) > 0:
            try:
                self.client.window.kill()
            except Exception:
                pass
        self.color = 0

class Screenshooter:

    def __init__(self, output_prefix, geometry, animation_delay):
        if False:
            return 10
        self.output_prefix = output_prefix
        self.geometry = geometry
        self.number = 1
        self.animation_delay = animation_delay
        self.output_paths = []

    def shoot(self, numbered=True, compress='lossless'):
        if False:
            for i in range(10):
                print('nop')
        if numbered:
            output_path = '{}.{}.png'.format(self.output_prefix, self.number)
        else:
            output_path = '{}.png'.format(self.output_prefix)
        thumbnail_path = output_path.replace('.png', '-thumb.png')
        subprocess.call(['scrot', '-o', '-t', self.geometry, output_path])
        os.rename(thumbnail_path, output_path)
        if compress:
            self.compress(compress, output_path)
        self.output_paths.append(output_path)
        self.number += 1

    def compress(self, method, file_path):
        if False:
            for i in range(10):
                print('nop')
        compress_command = ['pngquant', {'lossless': '--speed=1', 'lossy': '--quality=0-90'}.get(method), '--strip', '--skip-if-larger', '--force', '--output', file_path, file_path]
        try:
            subprocess.call(compress_command)
        except FileNotFoundError:
            pass

    def animate(self, delays=None, clear=False):
        if False:
            for i in range(10):
                print('nop')
        animate_command = ['convert', '-loop', '0', '-colors', '80', '-delay', self.animation_delay] + self.output_paths
        animate_command.extend(['-delay', '2x1', animate_command.pop(), '{}.gif'.format(self.output_prefix)])
        subprocess.call(animate_command)
        if clear:
            for output_path in self.output_paths:
                os.remove(output_path)