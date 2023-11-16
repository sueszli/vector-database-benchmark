import luigi

class SparkeyExportTask(luigi.Task):
    """
    A luigi task that writes to a local sparkey log file.

    Subclasses should implement the requires and output methods. The output
    must be a luigi.LocalTarget.

    The resulting sparkey log file will contain one entry for every line in
    the input, mapping from the first value to a tab-separated list of the
    rest of the line.

    To generate a simple key-value index, yield "key", "value" pairs from the input(s) to this task.
    """
    separator = '\t'

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(SparkeyExportTask, self).__init__(*args, **kwargs)

    def run(self):
        if False:
            while True:
                i = 10
        self._write_sparkey_file()

    def _write_sparkey_file(self):
        if False:
            i = 10
            return i + 15
        import sparkey
        infile = self.input()
        outfile = self.output()
        if not isinstance(outfile, luigi.LocalTarget):
            raise TypeError('output must be a LocalTarget')
        temp_output = luigi.LocalTarget(is_tmp=True)
        w = sparkey.LogWriter(temp_output.path)
        for line in infile.open('r'):
            (k, v) = line.strip().split(self.separator, 1)
            w[k] = v
        w.close()
        temp_output.move(outfile.path)