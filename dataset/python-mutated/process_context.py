import json
import os
import signal
import subprocess
import sys
import time
LIMIT_LEN_ENVS = ['TRAINER_IP_PORT_LIST', 'PADDLE_TRAINER_ENDPOINTS']

class ProcessContext:

    def __init__(self, cmd, env=os.environ, out=sys.stdout, err=sys.stderr, group=True, preexec_fn=None, shell=False):
        if False:
            while True:
                i = 10
        self._cmd = cmd
        self._env = env
        self._preexec_fn = preexec_fn
        self._stdout = out
        self._stderr = err
        self._group = group if os.name != 'nt' else False
        self._proc = None
        self._code = None
        self._shell = shell

    def _start(self):
        if False:
            print('Hello World!')
        pre_fn = os.setsid if self._group else None
        log_dir = self._env['PADDLE_LOG_DIR']
        os.makedirs(log_dir, exist_ok=True)
        rank = self._env.get('PADDLE_TRAINER_ID')
        if rank is not None:
            rank = int(rank)
            backup_env_path = str(os.path.join(log_dir, f'backup_env.{rank}.json'))
            envs = {'PADDLE_BACKUP_ENV_PATH': backup_env_path}
            max_len = int(os.getenv('PADDLE_ENV_LIMIT_LEN', 48000))
            for (k, v) in self._env.items():
                if k not in LIMIT_LEN_ENVS or len(v) < max_len:
                    envs[k] = v
            with open(backup_env_path, 'w') as f:
                json.dump(dict(self._env), f, indent=4, sort_keys=True)
        else:
            envs = self._env
        self._proc = subprocess.Popen(self._cmd, env=envs, stdout=self._stdout, stderr=self._stderr, preexec_fn=self._preexec_fn or pre_fn, shell=self._shell)

    def _close_std(self):
        if False:
            i = 10
            return i + 15
        try:
            if not self._stdout.isatty():
                self._stdout.close()
            if not self._stderr.isatty():
                self._stderr.close()
        except:
            pass

    def alive(self):
        if False:
            for i in range(10):
                print('nop')
        return self._proc and self._proc.poll() is None

    def exit_code(self):
        if False:
            while True:
                i = 10
        return self._proc.poll() if self._proc else None

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        self._start()

    def terminate(self, force=False, max_retry=3):
        if False:
            for i in range(10):
                print('nop')
        for i in range(max_retry):
            if self.alive():
                if self._group:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                else:
                    self._proc.terminate()
                time.sleep(0.2)
            else:
                break
        if force and self.alive():
            self._proc.kill()
        self._close_std()
        return self.alive()

    def wait(self, timeout=None):
        if False:
            print('Hello World!')
        self._proc.wait(timeout)