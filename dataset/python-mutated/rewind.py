import array
import logging
from pyboy.plugins.base_plugin import PyBoyPlugin
from pyboy.utils import IntIOInterface, WindowEvent
logger = logging.getLogger(__name__)
try:
    from cython import compiled
    cythonmode = compiled
except ImportError:
    cythonmode = False
FIXED_BUFFER_SIZE = 8 * 1024 * 1024
FIXED_BUFFER_MIN_ALLOC = 256 * 1024
FILL_VALUE = 123

class Rewind(PyBoyPlugin):
    argv = [('--rewind', {'action': 'store_true', 'help': 'Enable rewind function'})]

    def __init__(self, *args):
        if False:
            print('Hello World!')
        super().__init__(*args)
        self.rewind_speed = 1.0
        self.rewind_buffer = DeltaFixedAllocBuffers()

    def post_tick(self):
        if False:
            while True:
                i = 10
        if not self.pyboy.paused:
            self.mb.save_state(self.rewind_buffer)
            self.rewind_buffer.new()

    def window_title(self):
        if False:
            for i in range(10):
                print('nop')
        return ' Rewind: %0.2fKB/s' % (self.rewind_buffer.avg_section_size * 60 / 1024)

    def handle_events(self, events):
        if False:
            i = 10
            return i + 15
        old_rewind_speed = self.rewind_speed
        for event in events:
            if event == WindowEvent.UNPAUSE:
                self.rewind_buffer.commit()
            elif event == WindowEvent.PAUSE_TOGGLE:
                if self.pyboy.paused:
                    self.rewind_buffer.commit()
            elif event == WindowEvent.RELEASE_REWIND_FORWARD:
                self.rewind_speed = 1
            elif event == WindowEvent.PRESS_REWIND_FORWARD:
                self.pyboy._pause()
                if self.rewind_buffer.seek_frame(1):
                    self.mb.load_state(self.rewind_buffer)
                    events.append(WindowEvent._INTERNAL_RENDERER_FLUSH)
                    self.rewind_speed = min(self.rewind_speed * 1.1, 5)
                else:
                    logger.info('Rewind limit reached')
            elif event == WindowEvent.RELEASE_REWIND_BACK:
                self.rewind_speed = 1
            elif event == WindowEvent.PRESS_REWIND_BACK:
                self.pyboy._pause()
                if self.rewind_buffer.seek_frame(-1):
                    self.mb.load_state(self.rewind_buffer)
                    events.append(WindowEvent._INTERNAL_RENDERER_FLUSH)
                    self.rewind_speed = min(self.rewind_speed * 1.1, 5)
                else:
                    logger.info('Rewind limit reached')
        if old_rewind_speed != self.rewind_speed:
            self.pyboy.set_emulation_speed(int(self.rewind_speed))
        return events

    def enabled(self):
        if False:
            while True:
                i = 10
        return self.pyboy_argv.get('rewind')

class FixedAllocBuffers(IntIOInterface):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.buffer = _malloc(FIXED_BUFFER_SIZE)
        for n in range(FIXED_BUFFER_SIZE):
            self.buffer[n] = FILL_VALUE
        self.sections = [0]
        self.current_section = 0
        self.tail_pointer = 0
        self.section_head = 0
        self.section_tail = 0
        self.section_pointer = 0
        self.avg_section_size = 0.0

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        _free(self.buffer)

    def flush(self):
        if False:
            i = 10
            return i + 15
        pass

    def new(self):
        if False:
            while True:
                i = 10
        self.flush()
        self.sections.append(self.section_pointer)
        self.current_section += 1
        section_size = (self.section_head - self.section_tail + FIXED_BUFFER_SIZE) % FIXED_BUFFER_SIZE
        self.avg_section_size = 0.9 * self.avg_section_size + 0.1 * section_size
        self.section_tail = self.section_pointer

    def write(self, val):
        if False:
            for i in range(10):
                print('nop')
        assert val < 256
        if (self.section_pointer + 1) % FIXED_BUFFER_SIZE == self.tail_pointer:
            self.sections = self.sections[1:]
            self.tail_pointer = self.sections[0]
            self.current_section -= 1
        self.buffer[self.section_pointer] = val
        self.section_pointer = (self.section_pointer + 1) % FIXED_BUFFER_SIZE
        self.section_head = self.section_pointer
        return 1

    def read(self):
        if False:
            i = 10
            return i + 15
        if self.section_pointer == self.section_head:
            raise Exception('Read beyond section')
        data = self.buffer[self.section_pointer]
        self.section_pointer = (self.section_pointer + 1) % FIXED_BUFFER_SIZE
        return data

    def commit(self):
        if False:
            return 10
        if not self.section_head == self.section_pointer:
            raise Exception("Section wasn't read to finish. This would likely be unintentional")
        self.sections = self.sections[:self.current_section + 1]

    def seek_frame(self, frames):
        if False:
            return 10
        for _ in range(abs(frames)):
            if frames < 0:
                if self.current_section < 1:
                    return False
                head = self.sections[self.current_section]
                self.current_section -= 1
                tail = self.sections[self.current_section]
            else:
                if self.current_section == len(self.sections) - 1:
                    return False
                tail = self.sections[self.current_section]
                self.current_section += 1
                head = self.sections[self.current_section]
        (self.section_tail, self.section_head) = (tail, head)
        self.section_pointer = self.section_tail
        return True

class CompressedFixedAllocBuffers(FixedAllocBuffers):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        FixedAllocBuffers.__init__(self)
        self.zeros = 0

    def flush(self):
        if False:
            while True:
                i = 10
        if self.zeros > 0:
            chunks = self.zeros // 255
            rest = self.zeros % 255
            for i in range(chunks):
                FixedAllocBuffers.write(self, 0)
                FixedAllocBuffers.write(self, 255)
            if rest != 0:
                FixedAllocBuffers.write(self, 0)
                FixedAllocBuffers.write(self, rest)
        self.zeros = 0
        FixedAllocBuffers.flush(self)

    def write(self, data):
        if False:
            print('Hello World!')
        if data == 0:
            self.zeros += 1
            return 1
        else:
            self.flush()
            return FixedAllocBuffers.write(self, data)

    def read(self):
        if False:
            i = 10
            return i + 15
        if self.zeros > 0:
            self.zeros -= 1
            return 0
        else:
            byte = FixedAllocBuffers.read(self)
            if byte == 0:
                self.zeros = FixedAllocBuffers.read(self)
                self.zeros -= 1
            return byte

    def new(self):
        if False:
            print('Hello World!')
        FixedAllocBuffers.new(self)

    def commit(self):
        if False:
            return 10
        FixedAllocBuffers.commit(self)

    def seek_frame(self, v):
        if False:
            return 10
        return FixedAllocBuffers.seek_frame(self, v)

class DeltaFixedAllocBuffers(CompressedFixedAllocBuffers):
    """
    I chose to keep the code simple at the expense of some edge cases acting different from the other buffers.
    When seeking, the last frame will be lost. This has no practical effect, and is only noticeble in unittesting.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        CompressedFixedAllocBuffers.__init__(self)
        self.internal_pointer = 0
        self.prev_internal_pointer = 0
        self.internal_buffer = array.array('B', [0] * FIXED_BUFFER_MIN_ALLOC)
        self.internal_buffer_dirty = False
        self.base_frame = 0
        self.injected_zero_frame = 0

    def write(self, data):
        if False:
            i = 10
            return i + 15
        self.internal_buffer_dirty = True
        old_val = self.internal_buffer[self.internal_pointer]
        xor_val = data ^ old_val
        self.internal_buffer[self.internal_pointer] = data
        self.internal_pointer += 1
        return CompressedFixedAllocBuffers.write(self, xor_val)

    def read(self):
        if False:
            print('Hello World!')
        old_val = CompressedFixedAllocBuffers.read(self)
        data = old_val ^ self.internal_buffer[self.internal_pointer]
        self.internal_buffer[self.internal_pointer] = data
        self.internal_pointer += 1
        return data

    def commit(self):
        if False:
            return 10
        self.internal_pointer = 0
        self.injected_zero_frame = 0
        CompressedFixedAllocBuffers.commit(self)

    def new(self):
        if False:
            while True:
                i = 10
        self.prev_internal_pointer = self.internal_pointer
        self.internal_pointer = 0
        CompressedFixedAllocBuffers.new(self)

    def flush_internal_buffer(self):
        if False:
            for i in range(10):
                print('nop')
        for n in range(self.prev_internal_pointer):
            CompressedFixedAllocBuffers.write(self, self.internal_buffer[n])
            self.internal_buffer[n] = 0
        self.internal_buffer_dirty = False
        CompressedFixedAllocBuffers.new(self)
        self.injected_zero_frame = self.current_section

    def seek_frame(self, frames):
        if False:
            while True:
                i = 10
        if frames < 0:
            frames = -1
        else:
            frames = 1
        if self.internal_buffer_dirty:
            self.flush_internal_buffer()
        self.internal_pointer = 0
        if frames > 0 and self.injected_zero_frame - 1 == self.current_section:
            return False
        elif frames < 0 and self.current_section - 1 == self.base_frame:
            return False
        return CompressedFixedAllocBuffers.seek_frame(self, frames)
if not cythonmode:
    exec("\ndef _malloc(n):\n    return array.array('B', [0]*(FIXED_BUFFER_SIZE))\n\ndef _free(_):\n    pass\n", globals(), locals())