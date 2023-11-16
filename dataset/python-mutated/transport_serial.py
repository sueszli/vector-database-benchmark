import ast, io, errno, os, re, struct, sys, time
from collections import namedtuple
from errno import EPERM
from .console import VT_ENABLED
from .transport import TransportError, Transport

def stdout_write_bytes(b):
    if False:
        i = 10
        return i + 15
    b = b.replace(b'\x04', b'')
    sys.stdout.buffer.write(b)
    sys.stdout.buffer.flush()
listdir_result = namedtuple('dir_result', ['name', 'st_mode', 'st_ino', 'st_size'])

def reraise_filesystem_error(e, info):
    if False:
        print('Hello World!')
    if len(e.args) >= 3:
        if b'OSError' in e.args[2] and b'ENOENT' in e.args[2]:
            raise FileNotFoundError(info)
    raise

class SerialTransport(Transport):

    def __init__(self, device, baudrate=115200, wait=0, exclusive=True):
        if False:
            while True:
                i = 10
        self.in_raw_repl = False
        self.use_raw_paste = True
        self.device_name = device
        self.mounted = False
        import serial
        import serial.tools.list_ports
        serial_kwargs = {'baudrate': baudrate, 'interCharTimeout': 1}
        if serial.__version__ >= '3.3':
            serial_kwargs['exclusive'] = exclusive
        delayed = False
        for attempt in range(wait + 1):
            try:
                if device.startswith('rfc2217://'):
                    self.serial = serial.serial_for_url(device, **serial_kwargs)
                elif os.name == 'nt':
                    self.serial = serial.Serial(**serial_kwargs)
                    self.serial.port = device
                    portinfo = list(serial.tools.list_ports.grep(device))
                    if portinfo and portinfo[0].manufacturer != 'Microsoft':
                        self.serial.dtr = False
                        self.serial.rts = False
                    self.serial.open()
                else:
                    self.serial = serial.Serial(device, **serial_kwargs)
                break
            except OSError:
                if wait == 0:
                    continue
                if attempt == 0:
                    sys.stdout.write('Waiting {} seconds for pyboard '.format(wait))
                    delayed = True
            time.sleep(1)
            sys.stdout.write('.')
            sys.stdout.flush()
        else:
            if delayed:
                print('')
            raise TransportError('failed to access ' + device)
        if delayed:
            print('')

    def close(self):
        if False:
            print('Hello World!')
        self.serial.close()

    def read_until(self, min_num_bytes, ending, timeout=10, data_consumer=None):
        if False:
            i = 10
            return i + 15
        assert data_consumer is None or len(ending) == 1
        data = self.serial.read(min_num_bytes)
        if data_consumer:
            data_consumer(data)
        timeout_count = 0
        while True:
            if data.endswith(ending):
                break
            elif self.serial.inWaiting() > 0:
                new_data = self.serial.read(1)
                if data_consumer:
                    data_consumer(new_data)
                    data = new_data
                else:
                    data = data + new_data
                timeout_count = 0
            else:
                timeout_count += 1
                if timeout is not None and timeout_count >= 100 * timeout:
                    break
                time.sleep(0.01)
        return data

    def enter_raw_repl(self, soft_reset=True):
        if False:
            for i in range(10):
                print('nop')
        self.serial.write(b'\r\x03\x03')
        n = self.serial.inWaiting()
        while n > 0:
            self.serial.read(n)
            n = self.serial.inWaiting()
        self.serial.write(b'\r\x01')
        if soft_reset:
            data = self.read_until(1, b'raw REPL; CTRL-B to exit\r\n>')
            if not data.endswith(b'raw REPL; CTRL-B to exit\r\n>'):
                print(data)
                raise TransportError('could not enter raw repl')
            self.serial.write(b'\x04')
            data = self.read_until(1, b'soft reboot\r\n')
            if not data.endswith(b'soft reboot\r\n'):
                print(data)
                raise TransportError('could not enter raw repl')
        data = self.read_until(1, b'raw REPL; CTRL-B to exit\r\n')
        if not data.endswith(b'raw REPL; CTRL-B to exit\r\n'):
            print(data)
            raise TransportError('could not enter raw repl')
        self.in_raw_repl = True

    def exit_raw_repl(self):
        if False:
            for i in range(10):
                print('nop')
        self.serial.write(b'\r\x02')
        self.in_raw_repl = False

    def follow(self, timeout, data_consumer=None):
        if False:
            return 10
        data = self.read_until(1, b'\x04', timeout=timeout, data_consumer=data_consumer)
        if not data.endswith(b'\x04'):
            raise TransportError('timeout waiting for first EOF reception')
        data = data[:-1]
        data_err = self.read_until(1, b'\x04', timeout=timeout)
        if not data_err.endswith(b'\x04'):
            raise TransportError('timeout waiting for second EOF reception')
        data_err = data_err[:-1]
        return (data, data_err)

    def raw_paste_write(self, command_bytes):
        if False:
            for i in range(10):
                print('nop')
        data = self.serial.read(2)
        window_size = struct.unpack('<H', data)[0]
        window_remain = window_size
        i = 0
        while i < len(command_bytes):
            while window_remain == 0 or self.serial.inWaiting():
                data = self.serial.read(1)
                if data == b'\x01':
                    window_remain += window_size
                elif data == b'\x04':
                    self.serial.write(b'\x04')
                    return
                else:
                    raise TransportError('unexpected read during raw paste: {}'.format(data))
            b = command_bytes[i:min(i + window_remain, len(command_bytes))]
            self.serial.write(b)
            window_remain -= len(b)
            i += len(b)
        self.serial.write(b'\x04')
        data = self.read_until(1, b'\x04')
        if not data.endswith(b'\x04'):
            raise TransportError('could not complete raw paste: {}'.format(data))

    def exec_raw_no_follow(self, command):
        if False:
            while True:
                i = 10
        if isinstance(command, bytes):
            command_bytes = command
        else:
            command_bytes = bytes(command, encoding='utf8')
        data = self.read_until(1, b'>')
        if not data.endswith(b'>'):
            raise TransportError('could not enter raw repl')
        if self.use_raw_paste:
            self.serial.write(b'\x05A\x01')
            data = self.serial.read(2)
            if data == b'R\x00':
                pass
            elif data == b'R\x01':
                return self.raw_paste_write(command_bytes)
            else:
                data = self.read_until(1, b'w REPL; CTRL-B to exit\r\n>')
                if not data.endswith(b'w REPL; CTRL-B to exit\r\n>'):
                    print(data)
                    raise TransportError('could not enter raw repl')
            self.use_raw_paste = False
        for i in range(0, len(command_bytes), 256):
            self.serial.write(command_bytes[i:min(i + 256, len(command_bytes))])
            time.sleep(0.01)
        self.serial.write(b'\x04')
        data = self.serial.read(2)
        if data != b'OK':
            raise TransportError('could not exec command (response: %r)' % data)

    def exec_raw(self, command, timeout=10, data_consumer=None):
        if False:
            print('Hello World!')
        self.exec_raw_no_follow(command)
        return self.follow(timeout, data_consumer)

    def eval(self, expression, parse=False):
        if False:
            for i in range(10):
                print('nop')
        if parse:
            ret = self.exec('print(repr({}))'.format(expression))
            ret = ret.strip()
            return ast.literal_eval(ret.decode())
        else:
            ret = self.exec('print({})'.format(expression))
            ret = ret.strip()
            return ret

    def exec(self, command, data_consumer=None):
        if False:
            i = 10
            return i + 15
        (ret, ret_err) = self.exec_raw(command, data_consumer=data_consumer)
        if ret_err:
            raise TransportError('exception', ret, ret_err)
        return ret

    def execfile(self, filename):
        if False:
            i = 10
            return i + 15
        with open(filename, 'rb') as f:
            pyfile = f.read()
        return self.exec(pyfile)

    def fs_exists(self, src):
        if False:
            while True:
                i = 10
        try:
            self.exec('import os\nos.stat(%s)' % ("'%s'" % src if src else ''))
            return True
        except TransportError:
            return False

    def fs_ls(self, src):
        if False:
            for i in range(10):
                print('nop')
        cmd = "import os\nfor f in os.ilistdir(%s):\n print('{:12} {}{}'.format(f[3]if len(f)>3 else 0,f[0],'/'if f[1]&0x4000 else ''))" % ("'%s'" % src if src else '')
        self.exec(cmd, data_consumer=stdout_write_bytes)

    def fs_listdir(self, src=''):
        if False:
            i = 10
            return i + 15
        buf = bytearray()

        def repr_consumer(b):
            if False:
                i = 10
                return i + 15
            buf.extend(b.replace(b'\x04', b''))
        cmd = "import os\nfor f in os.ilistdir(%s):\n print(repr(f), end=',')" % ("'%s'" % src if src else '')
        try:
            buf.extend(b'[')
            self.exec(cmd, data_consumer=repr_consumer)
            buf.extend(b']')
        except TransportError as e:
            reraise_filesystem_error(e, src)
        return [listdir_result(*f) if len(f) == 4 else listdir_result(*f + (0,)) for f in ast.literal_eval(buf.decode())]

    def fs_stat(self, src):
        if False:
            print('Hello World!')
        try:
            self.exec('import os')
            return os.stat_result(self.eval('os.stat(%s)' % ("'%s'" % src), parse=True))
        except TransportError as e:
            reraise_filesystem_error(e, src)

    def fs_cat(self, src, chunk_size=256):
        if False:
            for i in range(10):
                print('nop')
        cmd = "with open('%s') as f:\n while 1:\n  b=f.read(%u)\n  if not b:break\n  print(b,end='')" % (src, chunk_size)
        self.exec(cmd, data_consumer=stdout_write_bytes)

    def fs_readfile(self, src, chunk_size=256):
        if False:
            return 10
        buf = bytearray()

        def repr_consumer(b):
            if False:
                print('Hello World!')
            buf.extend(b.replace(b'\x04', b''))
        cmd = "with open('%s', 'rb') as f:\n while 1:\n  b=f.read(%u)\n  if not b:break\n  print(b,end='')" % (src, chunk_size)
        try:
            self.exec(cmd, data_consumer=repr_consumer)
        except TransportError as e:
            reraise_filesystem_error(e, src)
        return ast.literal_eval(buf.decode())

    def fs_writefile(self, dest, data, chunk_size=256):
        if False:
            for i in range(10):
                print('nop')
        self.exec("f=open('%s','wb')\nw=f.write" % dest)
        while data:
            chunk = data[:chunk_size]
            self.exec('w(' + repr(chunk) + ')')
            data = data[len(chunk):]
        self.exec('f.close()')

    def fs_cp(self, src, dest, chunk_size=256, progress_callback=None):
        if False:
            i = 10
            return i + 15
        if progress_callback:
            src_size = self.fs_stat(src).st_size
            written = 0
        self.exec("fr=open('%s','rb')\nr=fr.read\nfw=open('%s','wb')\nw=fw.write" % (src, dest))
        while True:
            data_len = int(self.exec('d=r(%u)\nw(d)\nprint(len(d))' % chunk_size))
            if not data_len:
                break
            if progress_callback:
                written += data_len
                progress_callback(written, src_size)
        self.exec('fr.close()\nfw.close()')

    def fs_get(self, src, dest, chunk_size=256, progress_callback=None):
        if False:
            while True:
                i = 10
        if progress_callback:
            src_size = self.fs_stat(src).st_size
            written = 0
        self.exec("f=open('%s','rb')\nr=f.read" % src)
        with open(dest, 'wb') as f:
            while True:
                data = bytearray()
                self.exec('print(r(%u))' % chunk_size, data_consumer=lambda d: data.extend(d))
                assert data.endswith(b'\r\n\x04')
                try:
                    data = ast.literal_eval(str(data[:-3], 'ascii'))
                    if not isinstance(data, bytes):
                        raise ValueError('Not bytes')
                except (UnicodeError, ValueError) as e:
                    raise TransportError('fs_get: Could not interpret received data: %s' % str(e))
                if not data:
                    break
                f.write(data)
                if progress_callback:
                    written += len(data)
                    progress_callback(written, src_size)
        self.exec('f.close()')

    def fs_put(self, src, dest, chunk_size=256, progress_callback=None):
        if False:
            for i in range(10):
                print('nop')
        if progress_callback:
            src_size = os.path.getsize(src)
            written = 0
        self.exec("f=open('%s','wb')\nw=f.write" % dest)
        with open(src, 'rb') as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                if sys.version_info < (3,):
                    self.exec('w(b' + repr(data) + ')')
                else:
                    self.exec('w(' + repr(data) + ')')
                if progress_callback:
                    written += len(data)
                    progress_callback(written, src_size)
        self.exec('f.close()')

    def fs_mkdir(self, dir):
        if False:
            return 10
        self.exec("import os\nos.mkdir('%s')" % dir)

    def fs_rmdir(self, dir):
        if False:
            return 10
        self.exec("import os\nos.rmdir('%s')" % dir)

    def fs_rm(self, src):
        if False:
            while True:
                i = 10
        self.exec("import os\nos.remove('%s')" % src)

    def fs_touch(self, src):
        if False:
            for i in range(10):
                print('nop')
        self.exec("f=open('%s','a')\nf.close()" % src)

    def filesystem_command(self, args, progress_callback=None, verbose=False):
        if False:
            print('Hello World!')

        def fname_remote(src):
            if False:
                while True:
                    i = 10
            if src.startswith(':'):
                src = src[1:]
            return src.replace(os.path.sep, '/')

        def fname_cp_dest(src, dest):
            if False:
                return 10
            (_, src) = os.path.split(src)
            if dest is None or dest == '':
                dest = src
            elif dest == '.':
                dest = './' + src
            elif dest.endswith('/'):
                dest += src
            return dest
        cmd = args[0]
        args = args[1:]
        try:
            if cmd == 'cp':
                srcs = args[:-1]
                dest = args[-1]
                if dest.startswith(':'):
                    op_remote_src = self.fs_cp
                    op_local_src = self.fs_put
                else:
                    op_remote_src = self.fs_get
                    op_local_src = lambda src, dest, **_: __import__('shutil').copy(src, dest)
                for src in srcs:
                    if verbose:
                        print('cp %s %s' % (src, dest))
                    if src.startswith(':'):
                        op = op_remote_src
                    else:
                        op = op_local_src
                    src2 = fname_remote(src)
                    dest2 = fname_cp_dest(src2, fname_remote(dest))
                    op(src2, dest2, progress_callback=progress_callback)
            else:
                ops = {'cat': self.fs_cat, 'ls': self.fs_ls, 'mkdir': self.fs_mkdir, 'rm': self.fs_rm, 'rmdir': self.fs_rmdir, 'touch': self.fs_touch}
                if cmd not in ops:
                    raise TransportError("'{}' is not a filesystem command".format(cmd))
                if cmd == 'ls' and (not args):
                    args = ['']
                for src in args:
                    src = fname_remote(src)
                    if verbose:
                        print('%s :%s' % (cmd, src))
                    ops[cmd](src)
        except TransportError as er:
            if len(er.args) > 1:
                print(str(er.args[2], 'ascii'))
            else:
                print(er)
            self.exit_raw_repl()
            self.close()
            sys.exit(1)

    def mount_local(self, path, unsafe_links=False):
        if False:
            print('Hello World!')
        fout = self.serial
        if self.eval('"RemoteFS" in globals()') == b'False':
            self.exec(fs_hook_code)
        self.exec('__mount()')
        self.mounted = True
        self.cmd = PyboardCommand(self.serial, fout, path, unsafe_links=unsafe_links)
        self.serial = SerialIntercept(self.serial, self.cmd)

    def write_ctrl_d(self, out_callback):
        if False:
            for i in range(10):
                print('nop')
        self.serial.write(b'\x04')
        if not self.mounted:
            return
        INITIAL_TIMEOUT = 0.5
        BANNER_TIMEOUT = 2
        QUIET_TIMEOUT = 0.1
        FULL_TIMEOUT = 5
        t_start = t_last_activity = time.monotonic()
        data_all = b''
        soft_reboot_started = False
        soft_reboot_banner = False
        while True:
            t = time.monotonic()
            n = self.serial.inWaiting()
            if n > 0:
                data = self.serial.read(n)
                out_callback(data)
                data_all += data
                t_last_activity = t
            elif len(data_all) == 0:
                if t - t_start > INITIAL_TIMEOUT:
                    return
            else:
                if t - t_start > FULL_TIMEOUT:
                    if soft_reboot_started:
                        break
                    return
                next_data_timeout = QUIET_TIMEOUT
                if not soft_reboot_started and data_all.find(b'MPY: soft reboot') != -1:
                    soft_reboot_started = True
                if soft_reboot_started and (not soft_reboot_banner):
                    if data_all.find(b'\nMicroPython ') != -1:
                        soft_reboot_banner = True
                    elif data_all.find(b'\nraw REPL; CTRL-B to exit\r\n') != -1:
                        soft_reboot_banner = True
                    else:
                        next_data_timeout = BANNER_TIMEOUT
                if t - t_last_activity > next_data_timeout:
                    break
        if not soft_reboot_started:
            return
        if not soft_reboot_banner:
            out_callback(b'Warning: Could not remount local filesystem\r\n')
            return
        if data_all.endswith(b'>'):
            in_friendly_repl = False
            prompt = b'>'
        else:
            in_friendly_repl = True
            prompt = data_all.rsplit(b'\r\n', 1)[-1]
        self.mounted = False
        self.serial = self.serial.orig_serial
        out_callback(bytes(f'\r\nRemount local directory {self.cmd.root} at /remote\r\n', 'utf8'))
        self.serial.write(b'\x01')
        self.exec(fs_hook_code)
        self.exec('__mount()')
        self.mounted = True
        if in_friendly_repl:
            self.exit_raw_repl()
        self.read_until(len(prompt), prompt)
        out_callback(prompt)
        self.serial = SerialIntercept(self.serial, self.cmd)

    def umount_local(self):
        if False:
            while True:
                i = 10
        if self.mounted:
            self.exec('os.umount("/remote")')
            self.mounted = False
            self.serial = self.serial.orig_serial
fs_hook_cmds = {'CMD_STAT': 1, 'CMD_ILISTDIR_START': 2, 'CMD_ILISTDIR_NEXT': 3, 'CMD_OPEN': 4, 'CMD_CLOSE': 5, 'CMD_READ': 6, 'CMD_WRITE': 7, 'CMD_SEEK': 8, 'CMD_REMOVE': 9, 'CMD_RENAME': 10, 'CMD_MKDIR': 11, 'CMD_RMDIR': 12}
fs_hook_code = 'import os, io, struct, micropython\n\nSEEK_SET = 0\n\nclass RemoteCommand:\n    def __init__(self):\n        import select, sys\n        self.buf4 = bytearray(4)\n        self.fout = sys.stdout.buffer\n        self.fin = sys.stdin.buffer\n        self.poller = select.poll()\n        self.poller.register(self.fin, select.POLLIN)\n\n    def poll_in(self):\n        for _ in self.poller.ipoll(1000):\n            return\n        self.end()\n        raise Exception(\'timeout waiting for remote\')\n\n    def rd(self, n):\n        buf = bytearray(n)\n        self.rd_into(buf, n)\n        return buf\n\n    def rd_into(self, buf, n):\n        # implement reading with a timeout in case other side disappears\n        if n == 0:\n            return\n        self.poll_in()\n        r = self.fin.readinto(buf, n)\n        if r < n:\n            mv = memoryview(buf)\n            while r < n:\n                self.poll_in()\n                r += self.fin.readinto(mv[r:], n - r)\n\n    def begin(self, type):\n        micropython.kbd_intr(-1)\n        buf4 = self.buf4\n        buf4[0] = 0x18\n        buf4[1] = type\n        self.fout.write(buf4, 2)\n        # Wait for sync byte 0x18, but don\'t get stuck forever\n        for i in range(30):\n            self.poller.poll(1000)\n            self.fin.readinto(buf4, 1)\n            if buf4[0] == 0x18:\n                break\n\n    def end(self):\n        micropython.kbd_intr(3)\n\n    def rd_s8(self):\n        self.rd_into(self.buf4, 1)\n        n = self.buf4[0]\n        if n & 0x80:\n            n -= 0x100\n        return n\n\n    def rd_s32(self):\n        buf4 = self.buf4\n        self.rd_into(buf4, 4)\n        n = buf4[0] | buf4[1] << 8 | buf4[2] << 16 | buf4[3] << 24\n        if buf4[3] & 0x80:\n            n -= 0x100000000\n        return n\n\n    def rd_u32(self):\n        buf4 = self.buf4\n        self.rd_into(buf4, 4)\n        return buf4[0] | buf4[1] << 8 | buf4[2] << 16 | buf4[3] << 24\n\n    def rd_bytes(self, buf):\n        # TODO if n is large (eg >256) then we may miss bytes on stdin\n        n = self.rd_s32()\n        if buf is None:\n            ret = buf = bytearray(n)\n        else:\n            ret = n\n        self.rd_into(buf, n)\n        return ret\n\n    def rd_str(self):\n        n = self.rd_s32()\n        if n == 0:\n            return \'\'\n        else:\n            return str(self.rd(n), \'utf8\')\n\n    def wr_s8(self, i):\n        self.buf4[0] = i\n        self.fout.write(self.buf4, 1)\n\n    def wr_s32(self, i):\n        struct.pack_into(\'<i\', self.buf4, 0, i)\n        self.fout.write(self.buf4)\n\n    def wr_bytes(self, b):\n        self.wr_s32(len(b))\n        self.fout.write(b)\n\n    # str and bytes act the same in MicroPython\n    wr_str = wr_bytes\n\n\nclass RemoteFile(io.IOBase):\n    def __init__(self, cmd, fd, is_text):\n        self.cmd = cmd\n        self.fd = fd\n        self.is_text = is_text\n\n    def __enter__(self):\n        return self\n\n    def __exit__(self, a, b, c):\n        self.close()\n\n    def __iter__(self):\n        return self\n\n    def __next__(self):\n        l = self.readline()\n        if not l:\n            raise StopIteration\n        return l\n\n    def ioctl(self, request, arg):\n        if request == 1:  # FLUSH\n            self.flush()\n        elif request == 2:  # SEEK\n            # This assumes a 32-bit bare-metal machine.\n            import machine\n            machine.mem32[arg] = self.seek(machine.mem32[arg], machine.mem32[arg + 4])\n        elif request == 4:  # CLOSE\n            self.close()\n        elif request == 11:  # BUFFER_SIZE\n            # This is used as the vfs_reader buffer. n + 4 should be less than 255 to\n            # fit in stdin ringbuffer on supported ports. n + 7 should be multiple of 16\n            # to efficiently use gc blocks in mp_reader_vfs_t.\n            return 249\n        else:\n            return -1\n        return 0\n\n    def flush(self):\n        pass\n\n    def close(self):\n        if self.fd is None:\n            return\n        c = self.cmd\n        c.begin(CMD_CLOSE)\n        c.wr_s8(self.fd)\n        c.end()\n        self.fd = None\n\n    def read(self, n=-1):\n        c = self.cmd\n        c.begin(CMD_READ)\n        c.wr_s8(self.fd)\n        c.wr_s32(n)\n        data = c.rd_bytes(None)\n        c.end()\n        if self.is_text:\n            data = str(data, \'utf8\')\n        else:\n            data = bytes(data)\n        return data\n\n    def readinto(self, buf):\n        c = self.cmd\n        c.begin(CMD_READ)\n        c.wr_s8(self.fd)\n        c.wr_s32(len(buf))\n        n = c.rd_bytes(buf)\n        c.end()\n        return n\n\n    def readline(self):\n        l = \'\'\n        while 1:\n            c = self.read(1)\n            l += c\n            if c == \'\\n\' or c == \'\':\n                return l\n\n    def readlines(self):\n        ls = []\n        while 1:\n            l = self.readline()\n            if not l:\n                return ls\n            ls.append(l)\n\n    def write(self, buf):\n        c = self.cmd\n        c.begin(CMD_WRITE)\n        c.wr_s8(self.fd)\n        c.wr_bytes(buf)\n        n = c.rd_s32()\n        c.end()\n        return n\n\n    def seek(self, n, whence=SEEK_SET):\n        c = self.cmd\n        c.begin(CMD_SEEK)\n        c.wr_s8(self.fd)\n        c.wr_s32(n)\n        c.wr_s8(whence)\n        n = c.rd_s32()\n        c.end()\n        if n < 0:\n            raise OSError(n)\n        return n\n\n\nclass RemoteFS:\n    def __init__(self, cmd):\n        self.cmd = cmd\n\n    def mount(self, readonly, mkfs):\n        pass\n\n    def umount(self):\n        pass\n\n    def chdir(self, path):\n        if not path.startswith("/"):\n            path = self.path + path\n        if not path.endswith("/"):\n            path += "/"\n        if path != "/":\n            self.stat(path)\n        self.path = path\n\n    def getcwd(self):\n        return self.path\n\n    def remove(self, path):\n        c = self.cmd\n        c.begin(CMD_REMOVE)\n        c.wr_str(self.path + path)\n        res = c.rd_s32()\n        c.end()\n        if res < 0:\n            raise OSError(-res)\n\n    def rename(self, old, new):\n        c = self.cmd\n        c.begin(CMD_RENAME)\n        c.wr_str(self.path + old)\n        c.wr_str(self.path + new)\n        res = c.rd_s32()\n        c.end()\n        if res < 0:\n            raise OSError(-res)\n\n    def mkdir(self, path):\n        c = self.cmd\n        c.begin(CMD_MKDIR)\n        c.wr_str(self.path + path)\n        res = c.rd_s32()\n        c.end()\n        if res < 0:\n            raise OSError(-res)\n\n    def rmdir(self, path):\n        c = self.cmd\n        c.begin(CMD_RMDIR)\n        c.wr_str(self.path + path)\n        res = c.rd_s32()\n        c.end()\n        if res < 0:\n            raise OSError(-res)\n\n    def stat(self, path):\n        c = self.cmd\n        c.begin(CMD_STAT)\n        c.wr_str(self.path + path)\n        res = c.rd_s8()\n        if res < 0:\n            c.end()\n            raise OSError(-res)\n        mode = c.rd_u32()\n        size = c.rd_u32()\n        atime = c.rd_u32()\n        mtime = c.rd_u32()\n        ctime = c.rd_u32()\n        c.end()\n        return mode, 0, 0, 0, 0, 0, size, atime, mtime, ctime\n\n    def ilistdir(self, path):\n        c = self.cmd\n        c.begin(CMD_ILISTDIR_START)\n        c.wr_str(self.path + path)\n        res = c.rd_s8()\n        c.end()\n        if res < 0:\n            raise OSError(-res)\n        def next():\n            while True:\n                c.begin(CMD_ILISTDIR_NEXT)\n                name = c.rd_str()\n                if name:\n                    type = c.rd_u32()\n                    c.end()\n                    yield (name, type, 0)\n                else:\n                    c.end()\n                    break\n        return next()\n\n    def open(self, path, mode):\n        c = self.cmd\n        c.begin(CMD_OPEN)\n        c.wr_str(self.path + path)\n        c.wr_str(mode)\n        fd = c.rd_s8()\n        c.end()\n        if fd < 0:\n            raise OSError(-fd)\n        return RemoteFile(c, fd, mode.find(\'b\') == -1)\n\n\ndef __mount():\n    os.mount(RemoteFS(RemoteCommand()), \'/remote\')\n    os.chdir(\'/remote\')\n'
for (key, value) in fs_hook_cmds.items():
    fs_hook_code = re.sub(key, str(value), fs_hook_code)
fs_hook_code = re.sub(' *#.*$', '', fs_hook_code, flags=re.MULTILINE)
fs_hook_code = re.sub('\n\n+', '\n', fs_hook_code)
fs_hook_code = re.sub('    ', ' ', fs_hook_code)
fs_hook_code = re.sub('rd_', 'r', fs_hook_code)
fs_hook_code = re.sub('wr_', 'w', fs_hook_code)
fs_hook_code = re.sub('buf4', 'b4', fs_hook_code)

class PyboardCommand:

    def __init__(self, fin, fout, path, unsafe_links=False):
        if False:
            print('Hello World!')
        self.fin = fin
        self.fout = fout
        self.root = path + '/'
        self.data_ilistdir = ['', []]
        self.data_files = []
        self.unsafe_links = unsafe_links

    def rd_s8(self):
        if False:
            print('Hello World!')
        return struct.unpack('<b', self.fin.read(1))[0]

    def rd_s32(self):
        if False:
            while True:
                i = 10
        return struct.unpack('<i', self.fin.read(4))[0]

    def rd_bytes(self):
        if False:
            for i in range(10):
                print('nop')
        n = self.rd_s32()
        return self.fin.read(n)

    def rd_str(self):
        if False:
            return 10
        n = self.rd_s32()
        if n == 0:
            return ''
        else:
            return str(self.fin.read(n), 'utf8')

    def wr_s8(self, i):
        if False:
            for i in range(10):
                print('nop')
        self.fout.write(struct.pack('<b', i))

    def wr_s32(self, i):
        if False:
            print('Hello World!')
        self.fout.write(struct.pack('<i', i))

    def wr_u32(self, i):
        if False:
            while True:
                i = 10
        self.fout.write(struct.pack('<I', i))

    def wr_bytes(self, b):
        if False:
            print('Hello World!')
        self.wr_s32(len(b))
        self.fout.write(b)

    def wr_str(self, s):
        if False:
            i = 10
            return i + 15
        b = bytes(s, 'utf8')
        self.wr_s32(len(b))
        self.fout.write(b)

    def log_cmd(self, msg):
        if False:
            print('Hello World!')
        print(f'[{msg}]', end='\r\n')

    def path_check(self, path):
        if False:
            print('Hello World!')
        if not self.unsafe_links:
            parent = os.path.realpath(self.root)
            child = os.path.realpath(path)
        else:
            parent = os.path.abspath(self.root)
            child = os.path.abspath(path)
        if parent != os.path.commonpath([parent, child]):
            raise OSError(EPERM, '')

    def do_stat(self):
        if False:
            return 10
        path = self.root + self.rd_str()
        try:
            self.path_check(path)
            stat = os.stat(path)
        except OSError as er:
            self.wr_s8(-abs(er.errno))
        else:
            self.wr_s8(0)
            self.wr_u32(stat.st_mode)
            self.wr_u32(stat.st_size)
            self.wr_u32(int(stat.st_atime))
            self.wr_u32(int(stat.st_mtime))
            self.wr_u32(int(stat.st_ctime))

    def do_ilistdir_start(self):
        if False:
            return 10
        path = self.root + self.rd_str()
        try:
            self.path_check(path)
            self.data_ilistdir[0] = path
            self.data_ilistdir[1] = os.listdir(path)
            self.wr_s8(0)
        except OSError as er:
            self.wr_s8(-abs(er.errno))

    def do_ilistdir_next(self):
        if False:
            for i in range(10):
                print('nop')
        if self.data_ilistdir[1]:
            entry = self.data_ilistdir[1].pop(0)
            try:
                stat = os.lstat(self.data_ilistdir[0] + '/' + entry)
                mode = stat.st_mode & 49152
            except OSError:
                mode = 0
            self.wr_str(entry)
            self.wr_u32(mode)
        else:
            self.wr_str('')

    def do_open(self):
        if False:
            for i in range(10):
                print('nop')
        path = self.root + self.rd_str()
        mode = self.rd_str()
        try:
            self.path_check(path)
            f = open(path, mode)
        except OSError as er:
            self.wr_s8(-abs(er.errno))
        else:
            is_text = mode.find('b') == -1
            try:
                fd = self.data_files.index(None)
                self.data_files[fd] = (f, is_text)
            except ValueError:
                fd = len(self.data_files)
                self.data_files.append((f, is_text))
            self.wr_s8(fd)

    def do_close(self):
        if False:
            i = 10
            return i + 15
        fd = self.rd_s8()
        self.data_files[fd][0].close()
        self.data_files[fd] = None

    def do_read(self):
        if False:
            while True:
                i = 10
        fd = self.rd_s8()
        n = self.rd_s32()
        buf = self.data_files[fd][0].read(n)
        if self.data_files[fd][1]:
            buf = bytes(buf, 'utf8')
        self.wr_bytes(buf)

    def do_seek(self):
        if False:
            print('Hello World!')
        fd = self.rd_s8()
        n = self.rd_s32()
        whence = self.rd_s8()
        try:
            n = self.data_files[fd][0].seek(n, whence)
        except io.UnsupportedOperation:
            n = -1
        self.wr_s32(n)

    def do_write(self):
        if False:
            return 10
        fd = self.rd_s8()
        buf = self.rd_bytes()
        if self.data_files[fd][1]:
            buf = str(buf, 'utf8')
        n = self.data_files[fd][0].write(buf)
        self.wr_s32(n)

    def do_remove(self):
        if False:
            print('Hello World!')
        path = self.root + self.rd_str()
        try:
            self.path_check(path)
            os.remove(path)
            ret = 0
        except OSError as er:
            ret = -abs(er.errno)
        self.wr_s32(ret)

    def do_rename(self):
        if False:
            print('Hello World!')
        old = self.root + self.rd_str()
        new = self.root + self.rd_str()
        try:
            self.path_check(old)
            self.path_check(new)
            os.rename(old, new)
            ret = 0
        except OSError as er:
            ret = -abs(er.errno)
        self.wr_s32(ret)

    def do_mkdir(self):
        if False:
            print('Hello World!')
        path = self.root + self.rd_str()
        try:
            self.path_check(path)
            os.mkdir(path)
            ret = 0
        except OSError as er:
            ret = -abs(er.errno)
        self.wr_s32(ret)

    def do_rmdir(self):
        if False:
            for i in range(10):
                print('nop')
        path = self.root + self.rd_str()
        try:
            self.path_check(path)
            os.rmdir(path)
            ret = 0
        except OSError as er:
            ret = -abs(er.errno)
        self.wr_s32(ret)
    cmd_table = {fs_hook_cmds['CMD_STAT']: do_stat, fs_hook_cmds['CMD_ILISTDIR_START']: do_ilistdir_start, fs_hook_cmds['CMD_ILISTDIR_NEXT']: do_ilistdir_next, fs_hook_cmds['CMD_OPEN']: do_open, fs_hook_cmds['CMD_CLOSE']: do_close, fs_hook_cmds['CMD_READ']: do_read, fs_hook_cmds['CMD_WRITE']: do_write, fs_hook_cmds['CMD_SEEK']: do_seek, fs_hook_cmds['CMD_REMOVE']: do_remove, fs_hook_cmds['CMD_RENAME']: do_rename, fs_hook_cmds['CMD_MKDIR']: do_mkdir, fs_hook_cmds['CMD_RMDIR']: do_rmdir}

class SerialIntercept:

    def __init__(self, serial, cmd):
        if False:
            while True:
                i = 10
        self.orig_serial = serial
        self.cmd = cmd
        self.buf = b''
        self.orig_serial.timeout = 5.0

    def _check_input(self, blocking):
        if False:
            for i in range(10):
                print('nop')
        if blocking or self.orig_serial.inWaiting() > 0:
            c = self.orig_serial.read(1)
            if c == b'\x18':
                c = self.orig_serial.read(1)[0]
                self.orig_serial.write(b'\x18')
                PyboardCommand.cmd_table[c](self.cmd)
            elif not VT_ENABLED and c == b'\x1b':
                esctype = self.orig_serial.read(1)
                if esctype == b'[':
                    while not 64 < self.orig_serial.read(1)[0] < 126:
                        pass
            else:
                self.buf += c

    @property
    def fd(self):
        if False:
            i = 10
            return i + 15
        return self.orig_serial.fd

    def close(self):
        if False:
            print('Hello World!')
        self.orig_serial.close()

    def inWaiting(self):
        if False:
            i = 10
            return i + 15
        self._check_input(False)
        return len(self.buf)

    def read(self, n):
        if False:
            while True:
                i = 10
        while len(self.buf) < n:
            self._check_input(True)
        out = self.buf[:n]
        self.buf = self.buf[n:]
        return out

    def write(self, buf):
        if False:
            i = 10
            return i + 15
        self.orig_serial.write(buf)