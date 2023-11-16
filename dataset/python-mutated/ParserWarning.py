class Warning(Exception):

    def __init__(self, Str, File=None, Line=None):
        if False:
            while True:
                i = 10
        self.message = Str
        self.FileName = File
        self.LineNumber = Line
        self.ToolName = 'EOT'