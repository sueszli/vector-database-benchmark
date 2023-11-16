"""An example of using GDB Python API with Pwntools."""
from pwn import *

def check_write(gdb, exp_buf):
    if False:
        while True:
            i = 10
    'Check that write() was called with the expected arguments.'
    fd = gdb.parse_and_eval('$rdi').cast(gdb.lookup_type('int'))
    assert fd == 1, fd
    buf_addr = gdb.parse_and_eval('$rsi').cast(gdb.lookup_type('long'))
    count = gdb.parse_and_eval('$rdx').cast(gdb.lookup_type('long'))
    buf = gdb.selected_inferior().read_memory(buf_addr, count).tobytes()
    assert buf == exp_buf, buf

def demo_sync_breakpoint(cat, gdb, txt):
    if False:
        return 10
    'Demonstrate a synchronous breakpoint.'
    gdb.Breakpoint('write', temporary=True)
    gdb.continue_nowait()
    cat.sendline(txt)
    gdb.wait()
    check_write(gdb, (txt + '\n').encode())
    gdb.continue_nowait()
    cat.recvuntil(txt)

def demo_async_breakpoint(cat, gdb, txt):
    if False:
        print('Hello World!')
    'Demonstrate asynchronous breakpoint.'

    class WriteBp(gdb.Breakpoint):

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__('write')
            self.count = 0

        def stop(self):
            if False:
                print('Hello World!')
            check_write(gdb, (txt + '\n').encode())
            self.count += 1
    bp = WriteBp()
    gdb.continue_nowait()
    cat.sendline(txt)
    cat.recvuntil(txt)
    assert bp.count == 1, bp.count
    gdb.interrupt_and_wait()
    bp.delete()
    gdb.continue_nowait()

def main():
    if False:
        while True:
            i = 10
    with gdb.debug('cat', gdbscript='\nset logging on\nset pagination off\n', api=True) as cat:
        cat.gdb.Breakpoint('read', temporary=True)
        cat.gdb.continue_and_wait()
        demo_sync_breakpoint(cat, cat.gdb, 'foo')
        cat.gdb.quit()
    with process('cat') as cat:
        (_, cat_gdb) = gdb.attach(cat, gdbscript='\nset logging on\nset pagination off\n', api=True)
        demo_async_breakpoint(cat, cat_gdb, 'bar')
        cat_gdb.quit()
if __name__ == '__main__':
    main()