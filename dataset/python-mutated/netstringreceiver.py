import gc
import math
import os
import sys
import time
from twisted.protocols.test import test_basic
from twisted.python.compat import range, raw_input
from twisted.test import proto_helpers
NETSTRING_POSTFIX = b','
USAGE = 'Usage: %s <number> <filename>\n\nThis script creates up to 2 ** <number> chunks with up to 2 **\n<number> characters and sends them to the NetstringReceiver. The\nsorted performance data for all combination is written to <filename>\nafterwards.\n\nYou might want to start with a small number, maybe 10 or 12, and slowly\nincrease it. Stop when the performance starts to deteriorate ;-).\n'

class PerformanceTester:
    """
    A class for testing the performance of some
    """
    headers = []
    lineFormat = ''
    performanceData = {}

    def __init__(self, filename):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initializes C{self.filename}.\n\n        If a file with this name already exists, asks if it should be\n        overwritten. Terminates with exit status 1, if the user does\n        not accept.\n        '
        if os.path.isfile(filename):
            response = raw_input('A file named %s exists. Overwrite it (y/n)? ' % filename)
            if response.lower() != 'y':
                print('Performance test cancelled.')
                sys.exit(1)
        self.filename = filename

    def testPerformance(self, number):
        if False:
            return 10
        '\n        Drives the execution of C{performTest} with arguments between\n        0 and C{number - 1}.\n\n        @param number: Defines the number of test runs to be performed.\n        @type number: C{int}\n        '
        for iteration in range(number):
            self.performTest(iteration)

    def performTest(self, iteration):
        if False:
            print('Hello World!')
        '\n        Performs one test iteration. Overwrite this.\n\n        @param iteration: The iteration number. Can be used to configure\n            the test.\n        @type iteration: C{int}\n        @raise NotImplementedError: because this method has to be implemented\n            by the subclass.\n        '
        raise NotImplementedError

    def createReport(self):
        if False:
            i = 10
            return i + 15
        '\n        Creates a file and writes a table with performance data.\n\n        The performance data are ordered by the total size of the netstrings.\n        In addition they show the chunk size, the number of chunks and the\n        time (in seconds) that elapsed while the C{NetstringReceiver}\n        received the netstring.\n\n        @param filename: The name of the report file that will be written.\n        @type filename: C{str}\n        '
        self.outputFile = open(self.filename, 'w')
        self.writeHeader()
        self.writePerformanceData()
        self.writeLineSeparator()
        print('The report was written to %s.' % self.filename)

    def writeHeader(self):
        if False:
            while True:
                i = 10
        '\n        Writes the table header for the report.\n        '
        self.writeLineSeparator()
        self.outputFile.write('| {} |\n'.format(' | '.join(self.headers)))
        self.writeLineSeparator()

    def writeLineSeparator(self):
        if False:
            i = 10
            return i + 15
        "\n        Writes a 'line separator' made from '+' and '-' characters.\n        "
        dashes = ('-' * (len(header) + 2) for header in self.headers)
        self.outputFile.write('+%s+\n' % '+'.join(dashes))

    def writePerformanceData(self):
        if False:
            return 10
        '\n        Writes one line for each item in C{self.performanceData}.\n        '
        for (combination, elapsed) in sorted(self.performanceData.items()):
            (totalSize, chunkSize, numberOfChunks) = combination
            self.outputFile.write(self.lineFormat % (totalSize, chunkSize, numberOfChunks, elapsed))

class NetstringPerformanceTester(PerformanceTester):
    """
    A class for determining the C{NetstringReceiver.dataReceived} performance.

    Instantiates a C{NetstringReceiver} and calls its
    C{dataReceived()} method with different chunks sizes and numbers
    of chunks.  Presents a table showing the relation between input
    data and time to process them.
    """
    headers = ['Chunk size', 'Number of chunks', 'Total size', 'Time to receive']
    lineFormat = '| %%%dd | %%%dd | %%%dd | %%%d.4f |\n' % tuple((len(header) for header in headers))

    def __init__(self, filename):
        if False:
            print('Hello World!')
        '\n        Sets up the output file and the netstring receiver that will be\n        used for receiving data.\n\n        @param filename: The name of the file for storing the report.\n        @type filename: C{str}\n        '
        PerformanceTester.__init__(self, filename)
        transport = proto_helpers.StringTransport()
        self.netstringReceiver = test_basic.TestNetstring()
        self.netstringReceiver.makeConnection(transport)

    def performTest(self, number):
        if False:
            while True:
                i = 10
        '\n        Tests the performance of C{NetstringReceiver.dataReceived}.\n\n        Feeds netstrings of various sizes in different chunk sizes\n        to a C{NetstringReceiver} and stores the elapsed time in\n        C{self.performanceData}.\n\n        @param number: The maximal chunks size / number of\n            chunks to be checked.\n        @type number: C{int}\n        '
        chunkSize = 2 ** number
        numberOfChunks = chunkSize
        while numberOfChunks:
            self.testCombination(chunkSize, numberOfChunks)
            numberOfChunks = numberOfChunks // 2

    def testCombination(self, chunkSize, numberOfChunks):
        if False:
            i = 10
            return i + 15
        '\n        Tests one combination of chunk size and number of chunks.\n\n        @param chunkSize: The size of one chunk to be sent to the\n            C{NetstringReceiver}.\n        @type chunkSize: C{int}\n        @param numberOfChunks: The number of C{chunkSize}-sized chunks to\n            be sent to the C{NetstringReceiver}.\n        @type numberOfChunks: C{int}\n        '
        (chunk, dataSize) = self.configureCombination(chunkSize, numberOfChunks)
        elapsed = self.receiveData(chunk, numberOfChunks, dataSize)
        key = (chunkSize, numberOfChunks, dataSize)
        self.performanceData[key] = elapsed

    def configureCombination(self, chunkSize, numberOfChunks):
        if False:
            for i in range(10):
                print('nop')
        "\n        Updates C{MAX_LENGTH} for {self.netstringReceiver} (to avoid\n        C{NetstringParseErrors} that might be raised if the size\n        exceeds the default C{MAX_LENGTH}).\n\n        Calculates and returns one 'chunk' of data and the total size\n        of the netstring.\n\n        @param chunkSize: The size of chunks that will be received.\n        @type chunkSize: C{int}\n        @param numberOfChunks: The number of C{chunkSize}-sized chunks\n            that will be received.\n        @type numberOfChunks: C{int}\n\n        @return: A tuple consisting of string of C{chunkSize} 'a'\n        characters and the size of the netstring data portion.\n        "
        chunk = b'a' * chunkSize
        dataSize = chunkSize * numberOfChunks
        self.netstringReceiver.MAX_LENGTH = dataSize
        numberOfDigits = math.ceil(math.log10(dataSize)) + 1
        return (chunk, dataSize)

    def receiveData(self, chunk, numberOfChunks, dataSize):
        if False:
            for i in range(10):
                print('nop')
        dr = self.netstringReceiver.dataReceived
        now = time.time()
        dr(f'{dataSize}:'.encode('ascii'))
        for idx in range(numberOfChunks):
            dr(chunk)
        dr(NETSTRING_POSTFIX)
        elapsed = time.time() - now
        assert self.netstringReceiver.received, "Didn't receive string!"
        return elapsed

def disableGarbageCollector():
    if False:
        while True:
            i = 10
    gc.disable()
    print('Disabled Garbage Collector.')

def main(number, filename):
    if False:
        return 10
    disableGarbageCollector()
    npt = NetstringPerformanceTester(filename)
    npt.testPerformance(int(number))
    npt.createReport()
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(USAGE % sys.argv[0])
        sys.exit(1)
    main(*sys.argv[1:3])