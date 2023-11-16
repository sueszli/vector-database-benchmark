from lightning.app import LightningWork, LightningApp, CloudCompute
import os

class YourComponent(LightningWork):

    def run(self):
        if False:
            return 10
        os.listdir('/foo')
mount = Mount(source='s3://lightning-example-public/', mount_path='/foo')
compute = CloudCompute(mounts=mount)
component = YourComponent(cloud_compute=compute)
app = LightningApp(component)