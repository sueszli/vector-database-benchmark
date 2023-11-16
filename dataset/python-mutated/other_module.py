import luigi

class OtherModuleTask(luigi.Task):
    p = luigi.Parameter()

    def output(self):
        if False:
            i = 10
            return i + 15
        return luigi.LocalTarget(self.p)

    def run(self):
        if False:
            print('Hello World!')
        with self.output().open('w') as f:
            f.write('Done!')