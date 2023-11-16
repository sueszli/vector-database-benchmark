from lightning.app import LightningWork, LightningFlow, LightningApp

class EmbeddingProcessor(LightningWork):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self.embeddings = None

    def run(self):
        if False:
            while True:
                i = 10
        print('PROCESSOR: Generating embeddings...')
        fake_embeddings = [[1, 2, 3], [2, 3, 4]]
        self.embeddings = storage.Payload(fake_embeddings)

class EmbeddingServer(LightningWork):

    def run(self, payload):
        if False:
            return 10
        print('SERVER: Using embeddings from processor', payload)
        embeddings = payload.value
        print('serving embeddings sent from EmbeddingProcessor: ', embeddings)

class WorkflowOrchestrator(LightningFlow):

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.processor = EmbeddingProcessor()
        self.server = EmbeddingServer()

    def run(self):
        if False:
            i = 10
            return i + 15
        self.processor.run()
        self.server.run(self.processor.embeddings)
app = LightningApp(WorkflowOrchestrator())