from __future__ import absolute_import
import serial
import serial.tools.list_ports
try:
    basestring
except NameError:
    basestring = str

class Serial(serial.Serial):
    """Just inherit the native Serial port implementation and patch the port property."""

    @serial.Serial.port.setter
    def port(self, value):
        if False:
            i = 10
            return i + 15
        'translate port name before storing it'
        if isinstance(value, basestring) and value.startswith('hwgrep://'):
            serial.Serial.port.__set__(self, self.from_url(value))
        else:
            serial.Serial.port.__set__(self, value)

    def from_url(self, url):
        if False:
            for i in range(10):
                print('nop')
        'extract host and port from an URL string'
        if url.lower().startswith('hwgrep://'):
            url = url[9:]
        n = 0
        test_open = False
        args = url.split('&')
        regexp = args.pop(0)
        for arg in args:
            if '=' in arg:
                (option, value) = arg.split('=', 1)
            else:
                option = arg
                value = None
            if option == 'n':
                n = int(value) - 1
                if n < 1:
                    raise ValueError('option "n" expects a positive integer larger than 1: {!r}'.format(value))
            elif option == 'skip_busy':
                test_open = True
            else:
                raise ValueError('unknown option: {!r}'.format(option))
        for (port, desc, hwid) in sorted(serial.tools.list_ports.grep(regexp)):
            if test_open:
                try:
                    s = serial.Serial(port)
                except serial.SerialException:
                    continue
                else:
                    s.close()
            if n:
                n -= 1
                continue
            return port
        else:
            raise serial.SerialException('no ports found matching regexp {!r}'.format(url))
if __name__ == '__main__':
    s = Serial(None)
    s.port = 'hwgrep://ttyS0'
    print(s)