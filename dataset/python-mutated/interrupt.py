from platform import platform
from pyinstrument import Profiler
p = Profiler()
p.start()

def func():
    if False:
        for i in range(10):
            print('nop')
    fd = open('/dev/urandom', 'rb')
    _ = fd.read(1024 * 1024)
func()
platform()
p.stop()
print(p.output_text())
p.write_html('ioerror_out.html')