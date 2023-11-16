from robot.output.xmllogger import XmlLogger

class OutputWriter(XmlLogger):

    def __init__(self, output, rpa=False):
        if False:
            return 10
        XmlLogger.__init__(self, output, rpa=rpa, generator='Rebot')

    def start_message(self, msg):
        if False:
            while True:
                i = 10
        self._write_message(msg)

    def close(self):
        if False:
            while True:
                i = 10
        self._writer.end('robot')
        self._writer.close()

    def end_result(self, result):
        if False:
            i = 10
            return i + 15
        self.close()