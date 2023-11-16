from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from gnuradio.ctrlport.GNURadio import ControlPort
from gnuradio.ctrlport import RPCConnection
from gnuradio import gr
import pmt
import sys

class ThriftRadioClient(object):

    def __init__(self, host, port):
        if False:
            while True:
                i = 10
        self.tsocket = TSocket.TSocket(host, port)
        self.transport = TTransport.TBufferedTransport(self.tsocket)
        self.protocol = TBinaryProtocol.TBinaryProtocol(self.transport)
        self.radio = ControlPort.Client(self.protocol)
        self.transport.open()
        self.host = host
        self.port = port

    def __del__(self):
        if False:
            while True:
                i = 10
        try:
            self.transport.close()
            self.radio.shutdown()
        except:
            pass

    def getRadio(self):
        if False:
            print('Hello World!')
        return self.radio
'\nRPC Client interface for the Apache Thrift middle-ware RPC transport.\n\nArgs:\n    port: port number of the connection\n    host: hostname of the connection\n'

class RPCConnectionThrift(RPCConnection.RPCConnection):

    class Knob(object):

        def __init__(self, key, value=None, ktype=0):
            if False:
                print('Hello World!')
            (self.key, self.value, self.ktype) = (key, value, ktype)

        def __repr__(self):
            if False:
                return 10
            return '({0} = {1})'.format(self.key, self.value)

    def __init__(self, host=None, port=None):
        if False:
            for i in range(10):
                print('nop')
        from gnuradio.ctrlport.GNURadio import ttypes
        self.BaseTypes = ttypes.BaseTypes
        self.KnobBase = ttypes.KnobBase
        if port is None:
            p = gr.prefs()
            thrift_config_file = p.get_string('ControlPort', 'config', '')
            if len(thrift_config_file) > 0:
                p.add_config_file(thrift_config_file)
                port = p.get_long('thrift', 'port', 9090)
            else:
                port = 9090
        else:
            port = int(port)
        super(RPCConnectionThrift, self).__init__(method='thrift', port=port, host=host)
        self.newConnection(host, port)
        self.unpack_dict = {self.BaseTypes.BOOL: lambda k, b: self.Knob(k, b.value.a_bool, self.BaseTypes.BOOL), self.BaseTypes.BYTE: lambda k, b: self.Knob(k, b.value.a_byte, self.BaseTypes.BYTE), self.BaseTypes.SHORT: lambda k, b: self.Knob(k, b.value.a_short, self.BaseTypes.SHORT), self.BaseTypes.INT: lambda k, b: self.Knob(k, b.value.a_int, self.BaseTypes.INT), self.BaseTypes.LONG: lambda k, b: self.Knob(k, b.value.a_long, self.BaseTypes.LONG), self.BaseTypes.DOUBLE: lambda k, b: self.Knob(k, b.value.a_double, self.BaseTypes.DOUBLE), self.BaseTypes.STRING: lambda k, b: self.Knob(k, b.value.a_string, self.BaseTypes.STRING), self.BaseTypes.COMPLEX: lambda k, b: self.Knob(k, b.value.a_complex, self.BaseTypes.COMPLEX), self.BaseTypes.F32VECTOR: lambda k, b: self.Knob(k, b.value.a_f32vector, self.BaseTypes.F32VECTOR), self.BaseTypes.F64VECTOR: lambda k, b: self.Knob(k, b.value.a_f64vector, self.BaseTypes.F64VECTOR), self.BaseTypes.S64VECTOR: lambda k, b: self.Knob(k, b.value.a_s64vector, self.BaseTypes.S64VECTOR), self.BaseTypes.S32VECTOR: lambda k, b: self.Knob(k, b.value.a_s32vector, self.BaseTypes.S32VECTOR), self.BaseTypes.S16VECTOR: lambda k, b: self.Knob(k, b.value.a_s16vector, self.BaseTypes.S16VECTOR), self.BaseTypes.S8VECTOR: lambda k, b: self.Knob(k, b.value.a_s8vector, self.BaseTypes.S8VECTOR), self.BaseTypes.C32VECTOR: lambda k, b: self.Knob(k, b.value.a_c32vector, self.BaseTypes.C32VECTOR)}
        self.pack_dict = {self.BaseTypes.BOOL: lambda k: ttypes.Knob(type=k.ktype, value=ttypes.KnobBase(a_bool=k.value)), self.BaseTypes.BYTE: lambda k: ttypes.Knob(type=k.ktype, value=ttypes.KnobBase(a_byte=k.value)), self.BaseTypes.SHORT: lambda k: ttypes.Knob(type=k.ktype, value=ttypes.KnobBase(a_short=k.value)), self.BaseTypes.INT: lambda k: ttypes.Knob(type=k.ktype, value=ttypes.KnobBase(a_int=k.value)), self.BaseTypes.LONG: lambda k: ttypes.Knob(type=k.ktype, value=ttypes.KnobBase(a_long=k.value)), self.BaseTypes.DOUBLE: lambda k: ttypes.Knob(type=k.ktype, value=ttypes.KnobBase(a_double=k.value)), self.BaseTypes.STRING: lambda k: ttypes.Knob(type=k.ktype, value=ttypes.KnobBase(a_string=k.value)), self.BaseTypes.COMPLEX: lambda k: ttypes.Knob(type=k.ktype, value=ttypes.KnobBase(a_complex=k.value)), self.BaseTypes.F32VECTOR: lambda k: ttypes.Knob(type=k.ktype, value=ttypes.KnobBase(a_f32vector=k.value)), self.BaseTypes.F64VECTOR: lambda k: ttypes.Knob(type=k.ktype, value=ttypes.KnobBase(a_f64vector=k.value)), self.BaseTypes.S64VECTOR: lambda k: ttypes.Knob(type=k.ktype, value=ttypes.KnobBase(a_s64vector=k.value)), self.BaseTypes.S32VECTOR: lambda k: ttypes.Knob(type=k.ktype, value=ttypes.KnobBase(a_s32vector=k.value)), self.BaseTypes.S16VECTOR: lambda k: ttypes.Knob(type=k.ktype, value=ttypes.KnobBase(a_s16vector=k.value)), self.BaseTypes.S8VECTOR: lambda k: ttypes.Knob(type=k.ktype, value=ttypes.KnobBase(a_s8vector=k.value)), self.BaseTypes.C32VECTOR: lambda k: ttypes.Knob(type=k.ktype, value=ttypes.KnobBase(a_c32vector=k.value))}

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Apache Thrift connection to {0}:{1}'.format(self.thriftclient.host, self.thriftclient.port)

    def unpackKnob(self, key, knob):
        if False:
            print('Hello World!')
        f = self.unpack_dict.get(knob.type, None)
        if f:
            return f(key, knob)
        else:
            sys.stderr.write('unpackKnobs: Incorrect Knob type: {0}\n'.format(knob.type))
            raise exceptions.ValueError

    def packKnob(self, knob):
        if False:
            print('Hello World!')
        f = self.pack_dict.get(knob.ktype, None)
        if f:
            return f(knob)
        else:
            sys.stderr.write('packKnobs: Incorrect Knob type: {0}\n'.format(knob.type))
            raise exceptions.ValueError

    def newConnection(self, host=None, port=None):
        if False:
            while True:
                i = 10
        self.thriftclient = ThriftRadioClient(host, int(port))

    def properties(self, *args):
        if False:
            while True:
                i = 10
        knobprops = self.thriftclient.radio.properties(*args)
        for (key, knobprop) in list(knobprops.items()):
            knobprops[key].min = self.unpackKnob(key, knobprop.min)
            knobprops[key].max = self.unpackKnob(key, knobprop.max)
            knobprops[key].defaultvalue = self.unpackKnob(key, knobprop.defaultvalue)
        return knobprops

    def getKnobs(self, *args):
        if False:
            print('Hello World!')
        result = {}
        for (key, knob) in list(self.thriftclient.radio.getKnobs(*args).items()):
            result[key] = self.unpackKnob(key, knob)
            if knob.type == self.BaseTypes.C32VECTOR:
                for i in range(len(result[key].value)):
                    result[key].value[i] = complex(result[key].value[i].re, result[key].value[i].im)
        return result

    def getKnobsRaw(self, *args):
        if False:
            i = 10
            return i + 15
        result = {}
        for (key, knob) in list(self.thriftclient.radio.getKnobs(*args).items()):
            result[key] = knob
        return result

    def getRe(self, *args):
        if False:
            i = 10
            return i + 15
        result = {}
        for (key, knob) in list(self.thriftclient.radio.getRe(*args).items()):
            result[key] = self.unpackKnob(key, knob)
        return result

    def setKnobs(self, *args):
        if False:
            for i in range(10):
                print('nop')
        if type(*args) == dict:
            a = dict(*args)
            result = {}
            for (key, knob) in list(a.items()):
                result[key] = self.packKnob(knob)
            self.thriftclient.radio.setKnobs(result)
        elif type(*args) == list or type(*args) == tuple:
            a = list(*args)
            result = {}
            for k in a:
                result[k.key] = self.packKnob(k)
            self.thriftclient.radio.setKnobs(result)
        else:
            sys.stderr.write('setKnobs: Invalid type; must be dict, list, or tuple\n')

    def shutdown(self):
        if False:
            for i in range(10):
                print('nop')
        self.thriftclient.radio.shutdown()

    def postMessage(self, blk_alias, port, msg):
        if False:
            i = 10
            return i + 15
        "\n        blk_alias: the alias of the block we are posting the message\n                   to; must have an open message port named 'port'.\n                   Provide as a string.\n        port: The name of the message port we are sending the message to.\n              Provide as a string.\n        msg: The actual message. Provide this as a PMT of the form\n             right for the message port.\n        The alias and port names are converted to PMT symbols and\n        serialized. The msg is already a PMT and so just serialized.\n        "
        self.thriftclient.radio.postMessage(pmt.serialize_str(pmt.intern(blk_alias)), pmt.serialize_str(pmt.intern(port)), pmt.serialize_str(msg))

    def printProperties(self, props):
        if False:
            i = 10
            return i + 15
        info = ''
        info += 'Item:\t\t{0}\n'.format(props.description)
        info += 'units:\t\t{0}\n'.format(props.units)
        info += 'min:\t\t{0}\n'.format(props.min.value)
        info += 'max:\t\t{0}\n'.format(props.max.value)
        info += 'default:\t\t{0}\n'.format(props.defaultvalue.value)
        info += 'Type Code:\t0x{0:x}\n'.format(props.type)
        info += 'Disp Code:\t0x{0:x}\n'.format(props.display)
        return info