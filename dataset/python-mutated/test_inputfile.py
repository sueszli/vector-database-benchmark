import contextlib
import subprocess
import sys
from io import BytesIO
import pytest
from telegram import InputFile
from tests.auxil.files import data_file
from tests.auxil.slots import mro_slots

@pytest.fixture(scope='module')
def png_file():
    if False:
        print('Hello World!')
    return data_file('game.png')

class TestInputFileWithoutRequest:

    def test_slot_behaviour(self):
        if False:
            return 10
        inst = InputFile(BytesIO(b'blah'), filename='tg.jpg')
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_subprocess_pipe(self, png_file):
        if False:
            for i in range(10):
                print('nop')
        cmd_str = 'type' if sys.platform == 'win32' else 'cat'
        cmd = [cmd_str, str(png_file)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=sys.platform == 'win32')
        in_file = InputFile(proc.stdout)
        assert in_file.input_file_content == png_file.read_bytes()
        assert in_file.mimetype == 'application/octet-stream'
        assert in_file.filename == 'application.octet-stream'
        with contextlib.suppress(ProcessLookupError):
            proc.kill()

    @pytest.mark.parametrize('attach', [True, False])
    def test_attach(self, attach):
        if False:
            while True:
                i = 10
        input_file = InputFile('contents', attach=attach)
        if attach:
            assert isinstance(input_file.attach_name, str)
            assert input_file.attach_uri == f'attach://{input_file.attach_name}'
        else:
            assert input_file.attach_name is None
            assert input_file.attach_uri is None

    def test_mimetypes(self):
        if False:
            print('Hello World!')
        assert InputFile(data_file('telegram.jpg').open('rb')).mimetype == 'image/jpeg'
        assert InputFile(data_file('telegram.webp').open('rb')).mimetype in ['application/octet-stream', 'image/webp']
        assert InputFile(data_file('telegram.mp3').open('rb')).mimetype == 'audio/mpeg'
        assert InputFile(data_file('telegram.midi').open('rb')).mimetype in ['audio/mid', 'audio/midi']
        assert InputFile(BytesIO(b'blah'), filename='tg.jpg').mimetype == 'image/jpeg'
        assert InputFile(BytesIO(b'blah'), filename='tg.mp3').mimetype == 'audio/mpeg'
        assert InputFile(BytesIO(b'blah'), filename='tg.notaproperext').mimetype == 'application/octet-stream'
        assert InputFile(BytesIO(b'blah')).mimetype == 'application/octet-stream'
        assert InputFile(data_file('text_file.txt').open()).mimetype == 'text/plain'

    def test_filenames(self):
        if False:
            i = 10
            return i + 15
        assert InputFile(data_file('telegram.jpg').open('rb')).filename == 'telegram.jpg'
        assert InputFile(data_file('telegram.jpg').open('rb'), filename='blah').filename == 'blah'
        assert InputFile(data_file('telegram.jpg').open('rb'), filename='blah.jpg').filename == 'blah.jpg'
        assert InputFile(data_file('telegram').open('rb')).filename == 'telegram'
        assert InputFile(data_file('telegram').open('rb'), filename='blah').filename == 'blah'
        assert InputFile(data_file('telegram').open('rb'), filename='blah.jpg').filename == 'blah.jpg'

        class MockedFileobject:

            def __init__(self, f):
                if False:
                    i = 10
                    return i + 15
                self.f = f.open('rb')

            def read(self):
                if False:
                    while True:
                        i = 10
                return self.f.read()
        assert InputFile(MockedFileobject(data_file('telegram.jpg'))).filename == 'application.octet-stream'
        assert InputFile(MockedFileobject(data_file('telegram.jpg')), filename='blah').filename == 'blah'
        assert InputFile(MockedFileobject(data_file('telegram.jpg')), filename='blah.jpg').filename == 'blah.jpg'
        assert InputFile(MockedFileobject(data_file('telegram'))).filename == 'application.octet-stream'
        assert InputFile(MockedFileobject(data_file('telegram')), filename='blah').filename == 'blah'
        assert InputFile(MockedFileobject(data_file('telegram')), filename='blah.jpg').filename == 'blah.jpg'

class TestInputFileWithRequest:

    async def test_send_bytes(self, bot, chat_id):
        message = await bot.send_document(chat_id, data_file('text_file.txt').read_bytes())
        out = BytesIO()
        await (await message.document.get_file()).download_to_memory(out=out)
        out.seek(0)
        assert out.read().decode('utf-8') == 'PTB Rocks! ⅞'

    async def test_send_string(self, bot, chat_id):
        message = await bot.send_document(chat_id, InputFile(data_file('text_file.txt').read_text(encoding='utf-8')))
        out = BytesIO()
        await (await message.document.get_file()).download_to_memory(out=out)
        out.seek(0)
        assert out.read().decode('utf-8') == 'PTB Rocks! ⅞'