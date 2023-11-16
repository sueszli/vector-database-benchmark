from robot.utils import file_writer

class LibdocJsonWriter:

    def write(self, libdoc, outfile):
        if False:
            while True:
                i = 10
        with file_writer(outfile) as writer:
            writer.write(libdoc.to_json(indent=2))