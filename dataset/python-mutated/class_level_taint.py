class ClassSink:

    async def sink(self, argument):
        pass

class ClassSource:

    async def source(self):
        pass

def test(class_sink: ClassSink, class_source: ClassSource):
    if False:
        return 10
    class_sink.sink(class_source.source())