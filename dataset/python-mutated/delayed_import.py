from ray import serve

@serve.deployment
class MyDeployment:

    def __call__(self, model_path):
        if False:
            i = 10
            return i + 15
        from my_module import my_model
        self.model = my_model.load(model_path)