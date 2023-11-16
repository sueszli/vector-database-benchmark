import unittest
import pygame

class MidiInputTest(unittest.TestCase):
    __tags__ = ['interactive']

    def setUp(self):
        if False:
            print('Hello World!')
        import pygame.midi
        pygame.midi.init()
        in_id = pygame.midi.get_default_input_id()
        if in_id != -1:
            self.midi_input = pygame.midi.Input(in_id)
        else:
            self.midi_input = None

    def tearDown(self):
        if False:
            print('Hello World!')
        if self.midi_input:
            self.midi_input.close()
        pygame.midi.quit()

    def test_Input(self):
        if False:
            for i in range(10):
                print('nop')
        i = pygame.midi.get_default_input_id()
        if self.midi_input:
            self.assertEqual(self.midi_input.device_id, i)
        i = pygame.midi.get_default_output_id()
        self.assertRaises(pygame.midi.MidiException, pygame.midi.Input, i)
        self.assertRaises(pygame.midi.MidiException, pygame.midi.Input, 9009)
        self.assertRaises(pygame.midi.MidiException, pygame.midi.Input, -1)
        self.assertRaises(TypeError, pygame.midi.Input, '1234')
        self.assertRaises(OverflowError, pygame.midi.Input, pow(2, 99))

    def test_poll(self):
        if False:
            print('Hello World!')
        if not self.midi_input:
            self.skipTest('No midi Input device')
        self.assertFalse(self.midi_input.poll())
        pygame.midi.quit()
        self.assertRaises(RuntimeError, self.midi_input.poll)
        self.midi_input = None

    def test_read(self):
        if False:
            return 10
        if not self.midi_input:
            self.skipTest('No midi Input device')
        read = self.midi_input.read(5)
        self.assertEqual(read, [])
        pygame.midi.quit()
        self.assertRaises(RuntimeError, self.midi_input.read, 52)
        self.midi_input = None

    def test_close(self):
        if False:
            return 10
        if not self.midi_input:
            self.skipTest('No midi Input device')
        self.assertIsNotNone(self.midi_input._input)
        self.midi_input.close()
        self.assertIsNone(self.midi_input._input)

class MidiOutputTest(unittest.TestCase):
    __tags__ = ['interactive']

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        import pygame.midi
        pygame.midi.init()
        m_out_id = pygame.midi.get_default_output_id()
        if m_out_id != -1:
            self.midi_output = pygame.midi.Output(m_out_id)
        else:
            self.midi_output = None

    def tearDown(self):
        if False:
            while True:
                i = 10
        if self.midi_output:
            self.midi_output.close()
        pygame.midi.quit()

    def test_Output(self):
        if False:
            i = 10
            return i + 15
        i = pygame.midi.get_default_output_id()
        if self.midi_output:
            self.assertEqual(self.midi_output.device_id, i)
        i = pygame.midi.get_default_input_id()
        self.assertRaises(pygame.midi.MidiException, pygame.midi.Output, i)
        self.assertRaises(pygame.midi.MidiException, pygame.midi.Output, 9009)
        self.assertRaises(pygame.midi.MidiException, pygame.midi.Output, -1)
        self.assertRaises(TypeError, pygame.midi.Output, '1234')
        self.assertRaises(OverflowError, pygame.midi.Output, pow(2, 99))

    def test_note_off(self):
        if False:
            print('Hello World!')
        if self.midi_output:
            out = self.midi_output
            out.note_on(5, 30, 0)
            out.note_off(5, 30, 0)
            with self.assertRaises(ValueError) as cm:
                out.note_off(5, 30, 25)
            self.assertEqual(str(cm.exception), 'Channel not between 0 and 15.')
            with self.assertRaises(ValueError) as cm:
                out.note_off(5, 30, -1)
            self.assertEqual(str(cm.exception), 'Channel not between 0 and 15.')

    def test_note_on(self):
        if False:
            for i in range(10):
                print('nop')
        if self.midi_output:
            out = self.midi_output
            out.note_on(5, 30, 0)
            out.note_on(5, 42, 10)
            with self.assertRaises(ValueError) as cm:
                out.note_on(5, 30, 25)
            self.assertEqual(str(cm.exception), 'Channel not between 0 and 15.')
            with self.assertRaises(ValueError) as cm:
                out.note_on(5, 30, -1)
            self.assertEqual(str(cm.exception), 'Channel not between 0 and 15.')

    def test_set_instrument(self):
        if False:
            while True:
                i = 10
        if not self.midi_output:
            self.skipTest('No midi device')
        out = self.midi_output
        out.set_instrument(5)
        out.set_instrument(42, channel=2)
        with self.assertRaises(ValueError) as cm:
            out.set_instrument(-6)
        self.assertEqual(str(cm.exception), 'Undefined instrument id: -6')
        with self.assertRaises(ValueError) as cm:
            out.set_instrument(156)
        self.assertEqual(str(cm.exception), 'Undefined instrument id: 156')
        with self.assertRaises(ValueError) as cm:
            out.set_instrument(5, -1)
        self.assertEqual(str(cm.exception), 'Channel not between 0 and 15.')
        with self.assertRaises(ValueError) as cm:
            out.set_instrument(5, 16)
        self.assertEqual(str(cm.exception), 'Channel not between 0 and 15.')

    def test_write(self):
        if False:
            while True:
                i = 10
        if not self.midi_output:
            self.skipTest('No midi device')
        out = self.midi_output
        out.write([[[192, 0, 0], 20000]])
        out.write([[[192], 20000]])
        out.write([[[192, 0, 0], 20000], [[144, 60, 100], 20500]])
        out.write([])
        verrry_long = [[[144, 60, i % 100], 20000 + 100 * i] for i in range(1024)]
        out.write(verrry_long)
        too_long = [[[144, 60, i % 100], 20000 + 100 * i] for i in range(1025)]
        self.assertRaises(IndexError, out.write, too_long)
        with self.assertRaises(TypeError) as cm:
            out.write('Non sens ?')
        error_msg = "unsupported operand type(s) for &: 'str' and 'int'"
        self.assertEqual(str(cm.exception), error_msg)
        with self.assertRaises(TypeError) as cm:
            out.write(["Hey what's that?"])
        self.assertEqual(str(cm.exception), error_msg)

    def test_write_short(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.midi_output:
            self.skipTest('No midi device')
        out = self.midi_output
        out.write_short(192)
        out.write_short(144, 65, 100)
        out.write_short(128, 65, 100)
        out.write_short(144)

    def test_write_sys_ex(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.midi_output:
            self.skipTest('No midi device')
        out = self.midi_output
        out.write_sys_ex(pygame.midi.time(), [240, 125, 16, 17, 18, 19, 247])

    def test_pitch_bend(self):
        if False:
            return 10
        if not self.midi_output:
            self.skipTest('No midi device')
        out = self.midi_output
        with self.assertRaises(ValueError) as cm:
            out.pitch_bend(5, channel=-1)
        self.assertEqual(str(cm.exception), 'Channel not between 0 and 15.')
        with self.assertRaises(ValueError) as cm:
            out.pitch_bend(5, channel=16)
        with self.assertRaises(ValueError) as cm:
            out.pitch_bend(-10001, 1)
        self.assertEqual(str(cm.exception), 'Pitch bend value must be between -8192 and +8191, not -10001.')
        with self.assertRaises(ValueError) as cm:
            out.pitch_bend(10665, 2)

    def test_close(self):
        if False:
            while True:
                i = 10
        if not self.midi_output:
            self.skipTest('No midi device')
        self.assertIsNotNone(self.midi_output._output)
        self.midi_output.close()
        self.assertIsNone(self.midi_output._output)

    def test_abort(self):
        if False:
            i = 10
            return i + 15
        if not self.midi_output:
            self.skipTest('No midi device')
        self.assertEqual(self.midi_output._aborted, 0)
        self.midi_output.abort()
        self.assertEqual(self.midi_output._aborted, 1)

class MidiModuleTest(unittest.TestCase):
    """Midi module tests that require midi hardware or midi.init().

    See MidiModuleNonInteractiveTest for non-interactive module tests.
    """
    __tags__ = ['interactive']

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        import pygame.midi
        pygame.midi.init()

    def tearDown(self):
        if False:
            return 10
        pygame.midi.quit()

    def test_get_count(self):
        if False:
            print('Hello World!')
        c = pygame.midi.get_count()
        self.assertIsInstance(c, int)
        self.assertTrue(c >= 0)

    def test_get_default_input_id(self):
        if False:
            i = 10
            return i + 15
        midin_id = pygame.midi.get_default_input_id()
        self.assertIsInstance(midin_id, int)
        self.assertTrue(midin_id >= -1)
        pygame.midi.quit()
        self.assertRaises(RuntimeError, pygame.midi.get_default_output_id)

    def test_get_default_output_id(self):
        if False:
            for i in range(10):
                print('nop')
        c = pygame.midi.get_default_output_id()
        self.assertIsInstance(c, int)
        self.assertTrue(c >= -1)
        pygame.midi.quit()
        self.assertRaises(RuntimeError, pygame.midi.get_default_output_id)

    def test_get_device_info(self):
        if False:
            while True:
                i = 10
        an_id = pygame.midi.get_default_output_id()
        if an_id != -1:
            (interf, name, input, output, opened) = pygame.midi.get_device_info(an_id)
            self.assertEqual(output, 1)
            self.assertEqual(input, 0)
            self.assertEqual(opened, 0)
        an_in_id = pygame.midi.get_default_input_id()
        if an_in_id != -1:
            r = pygame.midi.get_device_info(an_in_id)
            (interf, name, input, output, opened) = r
            self.assertEqual(output, 0)
            self.assertEqual(input, 1)
            self.assertEqual(opened, 0)
        out_of_range = pygame.midi.get_count()
        for num in range(out_of_range):
            self.assertIsNotNone(pygame.midi.get_device_info(num))
        info = pygame.midi.get_device_info(out_of_range)
        self.assertIsNone(info)

    def test_init(self):
        if False:
            while True:
                i = 10
        pygame.midi.quit()
        self.assertRaises(RuntimeError, pygame.midi.get_count)
        pygame.midi.init()
        pygame.midi.init()
        pygame.midi.init()
        pygame.midi.init()
        self.assertTrue(pygame.midi.get_init())

    def test_quit(self):
        if False:
            for i in range(10):
                print('nop')
        pygame.midi.quit()
        pygame.midi.init()
        pygame.midi.quit()
        pygame.midi.quit()
        pygame.midi.init()
        pygame.midi.init()
        pygame.midi.quit()
        self.assertFalse(pygame.midi.get_init())

    def test_get_init(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(pygame.midi.get_init())

    def test_time(self):
        if False:
            while True:
                i = 10
        mtime = pygame.midi.time()
        self.assertIsInstance(mtime, int)
        self.assertTrue(0 <= mtime < 100)

class MidiModuleNonInteractiveTest(unittest.TestCase):
    """Midi module tests that do not require midi hardware or midi.init().

    See MidiModuleTest for interactive module tests.
    """

    def setUp(self):
        if False:
            return 10
        import pygame.midi

    def test_midiin(self):
        if False:
            while True:
                i = 10
        'Ensures the MIDIIN event id exists in the midi module.\n\n        The MIDIIN event id can be accessed via the midi module for backward\n        compatibility.\n        '
        self.assertEqual(pygame.midi.MIDIIN, pygame.MIDIIN)
        self.assertEqual(pygame.midi.MIDIIN, pygame.locals.MIDIIN)
        self.assertNotEqual(pygame.midi.MIDIIN, pygame.MIDIOUT)
        self.assertNotEqual(pygame.midi.MIDIIN, pygame.locals.MIDIOUT)

    def test_midiout(self):
        if False:
            while True:
                i = 10
        'Ensures the MIDIOUT event id exists in the midi module.\n\n        The MIDIOUT event id can be accessed via the midi module for backward\n        compatibility.\n        '
        self.assertEqual(pygame.midi.MIDIOUT, pygame.MIDIOUT)
        self.assertEqual(pygame.midi.MIDIOUT, pygame.locals.MIDIOUT)
        self.assertNotEqual(pygame.midi.MIDIOUT, pygame.MIDIIN)
        self.assertNotEqual(pygame.midi.MIDIOUT, pygame.locals.MIDIIN)

    def test_MidiException(self):
        if False:
            i = 10
            return i + 15
        'Ensures the MidiException is raised as expected.'

        def raiseit():
            if False:
                while True:
                    i = 10
            raise pygame.midi.MidiException('Hello Midi param')
        with self.assertRaises(pygame.midi.MidiException) as cm:
            raiseit()
        self.assertEqual(cm.exception.parameter, 'Hello Midi param')

    def test_midis2events(self):
        if False:
            print('Hello World!')
        'Ensures midi events are properly converted to pygame events.'
        MIDI_DATA = 0
        MD_STATUS = 0
        MD_DATA1 = 1
        MD_DATA2 = 2
        MD_DATA3 = 3
        TIMESTAMP = 1
        midi_events = (((192, 0, 1, 2), 20000), ((144, 60, 1000, 'string_data'), 20001), (('0', '1', '2', '3'), '4'))
        expected_num_events = len(midi_events)
        for device_id in range(3):
            pg_events = pygame.midi.midis2events(midi_events, device_id)
            self.assertEqual(len(pg_events), expected_num_events)
            for (i, pg_event) in enumerate(pg_events):
                midi_event = midi_events[i]
                midi_event_data = midi_event[MIDI_DATA]
                self.assertEqual(pg_event.__class__.__name__, 'Event')
                self.assertEqual(pg_event.type, pygame.MIDIIN)
                self.assertEqual(pg_event.status, midi_event_data[MD_STATUS])
                self.assertEqual(pg_event.data1, midi_event_data[MD_DATA1])
                self.assertEqual(pg_event.data2, midi_event_data[MD_DATA2])
                self.assertEqual(pg_event.data3, midi_event_data[MD_DATA3])
                self.assertEqual(pg_event.timestamp, midi_event[TIMESTAMP])
                self.assertEqual(pg_event.vice_id, device_id)

    def test_midis2events__missing_event_data(self):
        if False:
            i = 10
            return i + 15
        'Ensures midi events with missing values are handled properly.'
        midi_event_missing_data = ((192, 0, 1), 20000)
        midi_event_missing_timestamp = ((192, 0, 1, 2),)
        for midi_event in (midi_event_missing_data, midi_event_missing_timestamp):
            with self.assertRaises(ValueError):
                events = pygame.midi.midis2events([midi_event], 0)

    def test_midis2events__extra_event_data(self):
        if False:
            return 10
        'Ensures midi events with extra values are handled properly.'
        midi_event_extra_data = ((192, 0, 1, 2, 'extra'), 20000)
        midi_event_extra_timestamp = ((192, 0, 1, 2), 20000, 'extra')
        for midi_event in (midi_event_extra_data, midi_event_extra_timestamp):
            with self.assertRaises(ValueError):
                events = pygame.midi.midis2events([midi_event], 0)

    def test_midis2events__extra_event_data_missing_timestamp(self):
        if False:
            print('Hello World!')
        'Ensures midi events with extra data and no timestamps are handled\n        properly.\n        '
        midi_event_extra_data_no_timestamp = ((192, 0, 1, 2, 'extra'),)
        with self.assertRaises(ValueError):
            events = pygame.midi.midis2events([midi_event_extra_data_no_timestamp], 0)

    def test_conversions(self):
        if False:
            while True:
                i = 10
        'of frequencies to midi note numbers and ansi note names.'
        from pygame.midi import frequency_to_midi, midi_to_frequency, midi_to_ansi_note
        self.assertEqual(frequency_to_midi(27.5), 21)
        self.assertEqual(frequency_to_midi(36.7), 26)
        self.assertEqual(frequency_to_midi(4186.0), 108)
        self.assertEqual(midi_to_frequency(21), 27.5)
        self.assertEqual(midi_to_frequency(26), 36.7)
        self.assertEqual(midi_to_frequency(108), 4186.0)
        self.assertEqual(midi_to_ansi_note(21), 'A0')
        self.assertEqual(midi_to_ansi_note(71), 'B4')
        self.assertEqual(midi_to_ansi_note(82), 'A#5')
        self.assertEqual(midi_to_ansi_note(83), 'B5')
        self.assertEqual(midi_to_ansi_note(93), 'A6')
        self.assertEqual(midi_to_ansi_note(94), 'A#6')
        self.assertEqual(midi_to_ansi_note(95), 'B6')
        self.assertEqual(midi_to_ansi_note(96), 'C7')
        self.assertEqual(midi_to_ansi_note(102), 'F#7')
        self.assertEqual(midi_to_ansi_note(108), 'C8')
if __name__ == '__main__':
    unittest.main()