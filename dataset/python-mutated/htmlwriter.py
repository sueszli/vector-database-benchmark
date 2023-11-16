from robot.htmldata import HtmlFileWriter, ModelWriter, LIBDOC

class LibdocHtmlWriter:

    def __init__(self, theme=None):
        if False:
            for i in range(10):
                print('nop')
        self.theme = theme

    def write(self, libdoc, output):
        if False:
            while True:
                i = 10
        model_writer = LibdocModelWriter(output, libdoc, self.theme)
        HtmlFileWriter(output, model_writer).write(LIBDOC)

class LibdocModelWriter(ModelWriter):

    def __init__(self, output, libdoc, theme=None):
        if False:
            return 10
        self.output = output
        self.libdoc = libdoc
        self.theme = theme

    def write(self, line):
        if False:
            while True:
                i = 10
        data = self.libdoc.to_json(include_private=False, theme=self.theme)
        self.output.write(f'<script type="text/javascript">\nlibdoc = {data}\n</script>\n')