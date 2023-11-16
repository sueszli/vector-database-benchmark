import importlib
from collections import namedtuple
from struct import pack
from future.utils import with_metaclass
from routersploit.core.exploit.exploit import BaseExploit, ExploitOptionsAggregator
from routersploit.core.exploit.option import OptIP, OptPort, OptString
from routersploit.core.exploit.exceptions import OptionValidationError
from routersploit.core.exploit.printer import print_status, print_error, print_success, print_info
from routersploit.core.exploit.utils import index_modules, random_text
architectures = namedtuple('ArchitectureType', ['ARMLE', 'MIPSBE', 'MIPSLE', 'X86', 'X64', 'PERL', 'PHP', 'PYTHON'])
Architectures = architectures(ARMLE='armle', MIPSBE='mipsbe', MIPSLE='mipsle', X86='x86', X64='x64', PERL='perl', PHP='php', PYTHON='python')
payload_handlers = namedtuple('PayloadHandlers', ['BIND_TCP', 'REVERSE_TCP'])
PayloadHandlers = payload_handlers(BIND_TCP='bind_tcp', REVERSE_TCP='reverse_tcp')
ARCH_ELF_HEADERS = {Architectures.ARMLE: b'\x7fELF\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00(\x00\x01\x00\x00\x00T\x80\x00\x004\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x004\x00 \x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x80\x00\x00\x00\x80\x00\x00\xef\xbe\xad\xde\xef\xbe\xad\xde\x07\x00\x00\x00\x00\x10\x00\x00', Architectures.MIPSBE: b'\x7fELF\x01\x02\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x08\x00\x00\x00\x01\x00@\x00T\x00\x00\x004\x00\x00\x00\x00\x00\x00\x00\x00\x004\x00 \x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00@\x00\x00\x00@\x00\x00\xde\xad\xbe\xef\xde\xad\xbe\xef\x00\x00\x00\x07\x00\x00\x10\x00', Architectures.MIPSLE: b'\x7fELF\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x08\x00\x01\x00\x00\x00T\x00@\x004\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x004\x00 \x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00@\x00\xef\xbe\xad\xde\xef\xbe\xad\xde\x07\x00\x00\x00\x00\x10\x00\x00', Architectures.X86: b'\x7fELF\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x03\x00\x01\x00\x00\x00T\x80\x04\x084\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x004\x00 \x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x80\x04\x08\x00\x80\x04\x08\xef\xbe\xad\xde\xef\xbe\xad\xde\x07\x00\x00\x00\x00\x10\x00\x00', Architectures.X64: b'\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00>\x00\x01\x00\x00\x00x\x00@\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x008\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x07\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00AAAAAAAABBBBBBBB\x00\x10\x00\x00\x00\x00\x00\x00'}

class ReverseTCPPayloadMixin(with_metaclass(ExploitOptionsAggregator, object)):
    handler = PayloadHandlers.REVERSE_TCP
    lhost = OptIP('', 'Connect-back IP address')
    lport = OptPort(5555, 'Connect-back TCP Port')

class BindTCPPayloadMixin(with_metaclass(ExploitOptionsAggregator, object)):
    handler = PayloadHandlers.BIND_TCP
    rport = OptPort(5555, 'Bind Port')

class BasePayload(BaseExploit):
    architecture = None
    handler = None
    encoder = OptString('', 'Encoder')
    fmt = None

    def __init__(self):
        if False:
            while True:
                i = 10
        if self.handler not in PayloadHandlers:
            raise OptionValidationError('Please use one of valid payload handlers: {}'.format(PayloadHandlers._fields))

    def generate(self):
        if False:
            return 10
        raise NotImplementedError("Please implement 'generate()' method")

    def run(self):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def get_encoders(self):
        if False:
            print('Hello World!')
        encoders = []
        all_encoders = [e for e in index_modules() if 'encoders.{}'.format(self.architecture) in e]
        for e in all_encoders:
            encoder = e.replace('encoders.{}.'.format(self.architecture), '').replace('.', '/')
            module = getattr(importlib.import_module('routersploit.modules.' + e), 'Encoder')
            encoders.append(('{}/{}'.format(self.architecture, encoder), module._Encoder__info__['name'], module._Encoder__info__['description']))
        return encoders

    def get_encoder(self, encoder):
        if False:
            while True:
                i = 10
        module_path = 'routersploit/modules/encoders/{}'.format(encoder).replace('/', '.')
        try:
            module = getattr(importlib.import_module(module_path), 'Encoder')
        except ImportError:
            return None
        return module()

class ArchitectureSpecificPayload(BasePayload):
    output = OptString('python', 'Output type: elf/c/python')
    filepath = OptString('/tmp/{}'.format(random_text(8)), 'Output file to write')

    def __init__(self):
        if False:
            return 10
        super(ArchitectureSpecificPayload, self).__init__()
        if self.architecture not in Architectures:
            raise OptionValidationError('Please use one of valid payload architectures: {}'.format(Architectures._fields))
        self.header = ARCH_ELF_HEADERS[self.architecture]
        self.bigendian = True if self.architecture.endswith('be') else False

    def run(self):
        if False:
            i = 10
            return i + 15
        print_status('Generating payload')
        try:
            data = self.generate()
        except OptionValidationError as e:
            print_error(e)
            return
        if self.output == 'elf':
            with open(self.filepath, 'wb+') as f:
                print_status('Building ELF payload')
                content = self.generate_elf(data)
                print_success('Saving file {}'.format(self.filepath))
                f.write(content)
        elif self.output == 'c':
            print_success('Bulding payload for C')
            content = self.generate_c(data)
            print_info(content)
        elif self.output == 'python':
            print_success('Building payload for python')
            content = self.generate_python(data)
            print_info(content)
        else:
            raise OptionValidationError('No such option as {}'.format(self.output))
        return content

    def generate_elf(self, data):
        if False:
            i = 10
            return i + 15
        elf = self.header + data
        if elf[4] == 1:
            if self.bigendian:
                p_filesz = pack('>L', len(elf))
                p_memsz = pack('>L', len(elf) + len(data))
            else:
                p_filesz = pack('<L', len(elf))
                p_memsz = pack('<L', len(elf) + len(data))
            content = elf[:68] + p_filesz + p_memsz + elf[76:]
        elif elf[4] == 2:
            if self.bigendian:
                p_filesz = pack('>Q', len(elf))
                p_memsz = pack('>Q', len(elf) + len(data))
            else:
                p_filesz = pack('<Q', len(elf))
                p_memsz = pack('<Q', len(elf) + len(data))
            content = elf[:96] + p_filesz + p_memsz + elf[112:]
        return content

    @staticmethod
    def generate_c(data):
        if False:
            i = 10
            return i + 15
        res = 'unsigned char sh[] = {\n    "'
        for (idx, x) in enumerate(data):
            if idx % 15 == 0 and idx != 0:
                res += '"\n    "'
            res += '\\x%02x' % x
        res += '"\n};'
        return res

    @staticmethod
    def generate_python(data):
        if False:
            return 10
        res = 'payload = (\n    "'
        for (idx, x) in enumerate(data):
            if idx % 15 == 0 and idx != 0:
                res += '"\n    "'
            res += '\\x%02x' % x
        res += '"\n)'
        return res

class GenericPayload(BasePayload):

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        print_status('Generating payload')
        payload = self.generate()
        if self.encoder:
            payload = self.encoder.encode(payload)
        if self.fmt:
            payload = self.fmt.format(payload)
        print_info(payload)
        return payload