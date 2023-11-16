from fastapi import FastAPI
from ray import serve
app = FastAPI(docs_url='/my_docs')

@serve.deployment
@serve.ingress(app)
class FastAPIDeployment:

    @app.get('/hello')
    def incr(self):
        if False:
            return 10
        return 'Hello world!'
node = FastAPIDeployment.bind()