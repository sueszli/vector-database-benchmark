"""A class supporting chat-style (command/response) protocols.

This class adds support for 'chat' style protocols - where one side
sends a 'command', and the other sends a response (examples would be
the common internet protocols - smtp, nntp, ftp, etc..).

The handle_read() method looks at the input stream for the current
'terminator' (usually '\\r\\n' for single-line responses, '\\r\\n.\\r\\n'
for multi-line output), calling self.found_terminator() on its
receipt.

for example:
Say you build an async nntp client using this class.  At the start
of the connection, you'll have self.terminator set to '\\r\\n', in
order to process the single-line greeting.  Just before issuing a
'LIST' command you'll set it to '\\r\\n.\\r\\n'.  The output of the LIST
command will be accumulated (using your own 'collect_incoming_data'
method) up to the terminator, and then control will be returned to
you - by calling your self.found_terminator() method.
"""
import asyncore
from collections import deque
from warnings import warn
warn('The asynchat module is deprecated and will be removed in Python 3.12. The recommended replacement is asyncio', DeprecationWarning, stacklevel=2)

class async_chat(asyncore.dispatcher):
    """This is an abstract class.  You must derive from this class, and add
    the two methods collect_incoming_data() and found_terminator()"""
    ac_in_buffer_size = 65536
    ac_out_buffer_size = 65536
    use_encoding = 0
    encoding = 'latin-1'

    def __init__(self, sock=None, map=None):
        if False:
            while True:
                i = 10
        self.ac_in_buffer = b''
        self.incoming = []
        self.producer_fifo = deque()
        asyncore.dispatcher.__init__(self, sock, map)

    def collect_incoming_data(self, data):
        if False:
            while True:
                i = 10
        raise NotImplementedError('must be implemented in subclass')

    def _collect_incoming_data(self, data):
        if False:
            i = 10
            return i + 15
        self.incoming.append(data)

    def _get_data(self):
        if False:
            return 10
        d = b''.join(self.incoming)
        del self.incoming[:]
        return d

    def found_terminator(self):
        if False:
            print('Hello World!')
        raise NotImplementedError('must be implemented in subclass')

    def set_terminator(self, term):
        if False:
            while True:
                i = 10
        'Set the input delimiter.\n\n        Can be a fixed string of any length, an integer, or None.\n        '
        if isinstance(term, str) and self.use_encoding:
            term = bytes(term, self.encoding)
        elif isinstance(term, int) and term < 0:
            raise ValueError('the number of received bytes must be positive')
        self.terminator = term

    def get_terminator(self):
        if False:
            i = 10
            return i + 15
        return self.terminator

    def handle_read(self):
        if False:
            while True:
                i = 10
        try:
            data = self.recv(self.ac_in_buffer_size)
        except BlockingIOError:
            return
        except OSError:
            self.handle_error()
            return
        if isinstance(data, str) and self.use_encoding:
            data = bytes(str, self.encoding)
        self.ac_in_buffer = self.ac_in_buffer + data
        while self.ac_in_buffer:
            lb = len(self.ac_in_buffer)
            terminator = self.get_terminator()
            if not terminator:
                self.collect_incoming_data(self.ac_in_buffer)
                self.ac_in_buffer = b''
            elif isinstance(terminator, int):
                n = terminator
                if lb < n:
                    self.collect_incoming_data(self.ac_in_buffer)
                    self.ac_in_buffer = b''
                    self.terminator = self.terminator - lb
                else:
                    self.collect_incoming_data(self.ac_in_buffer[:n])
                    self.ac_in_buffer = self.ac_in_buffer[n:]
                    self.terminator = 0
                    self.found_terminator()
            else:
                terminator_len = len(terminator)
                index = self.ac_in_buffer.find(terminator)
                if index != -1:
                    if index > 0:
                        self.collect_incoming_data(self.ac_in_buffer[:index])
                    self.ac_in_buffer = self.ac_in_buffer[index + terminator_len:]
                    self.found_terminator()
                else:
                    index = find_prefix_at_end(self.ac_in_buffer, terminator)
                    if index:
                        if index != lb:
                            self.collect_incoming_data(self.ac_in_buffer[:-index])
                            self.ac_in_buffer = self.ac_in_buffer[-index:]
                        break
                    else:
                        self.collect_incoming_data(self.ac_in_buffer)
                        self.ac_in_buffer = b''

    def handle_write(self):
        if False:
            print('Hello World!')
        self.initiate_send()

    def handle_close(self):
        if False:
            i = 10
            return i + 15
        self.close()

    def push(self, data):
        if False:
            return 10
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError('data argument must be byte-ish (%r)', type(data))
        sabs = self.ac_out_buffer_size
        if len(data) > sabs:
            for i in range(0, len(data), sabs):
                self.producer_fifo.append(data[i:i + sabs])
        else:
            self.producer_fifo.append(data)
        self.initiate_send()

    def push_with_producer(self, producer):
        if False:
            return 10
        self.producer_fifo.append(producer)
        self.initiate_send()

    def readable(self):
        if False:
            for i in range(10):
                print('nop')
        'predicate for inclusion in the readable for select()'
        return 1

    def writable(self):
        if False:
            return 10
        'predicate for inclusion in the writable for select()'
        return self.producer_fifo or not self.connected

    def close_when_done(self):
        if False:
            i = 10
            return i + 15
        'automatically close this channel once the outgoing queue is empty'
        self.producer_fifo.append(None)

    def initiate_send(self):
        if False:
            return 10
        while self.producer_fifo and self.connected:
            first = self.producer_fifo[0]
            if not first:
                del self.producer_fifo[0]
                if first is None:
                    self.handle_close()
                    return
            obs = self.ac_out_buffer_size
            try:
                data = first[:obs]
            except TypeError:
                data = first.more()
                if data:
                    self.producer_fifo.appendleft(data)
                else:
                    del self.producer_fifo[0]
                continue
            if isinstance(data, str) and self.use_encoding:
                data = bytes(data, self.encoding)
            try:
                num_sent = self.send(data)
            except OSError:
                self.handle_error()
                return
            if num_sent:
                if num_sent < len(data) or obs < len(first):
                    self.producer_fifo[0] = first[num_sent:]
                else:
                    del self.producer_fifo[0]
            return

    def discard_buffers(self):
        if False:
            i = 10
            return i + 15
        self.ac_in_buffer = b''
        del self.incoming[:]
        self.producer_fifo.clear()

class simple_producer:

    def __init__(self, data, buffer_size=512):
        if False:
            for i in range(10):
                print('nop')
        self.data = data
        self.buffer_size = buffer_size

    def more(self):
        if False:
            while True:
                i = 10
        if len(self.data) > self.buffer_size:
            result = self.data[:self.buffer_size]
            self.data = self.data[self.buffer_size:]
            return result
        else:
            result = self.data
            self.data = b''
            return result

def find_prefix_at_end(haystack, needle):
    if False:
        while True:
            i = 10
    l = len(needle) - 1
    while l and (not haystack.endswith(needle[:l])):
        l -= 1
    return l