from lightning.app import LightningWork, LightningApp

class YourComponent(LightningWork):

    def run(self):
        if False:
            return 10
        print('RUN ANY PYTHON CODE HERE')
config = BuildConfig(image='gcr.io/google-samples/hello-app:1.0')
component = YourComponent(cloud_build_config=config)
app = LightningApp(component)