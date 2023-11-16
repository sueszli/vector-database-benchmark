from lightning.app import LightningWork, LightningApp, CloudCompute

class YourComponent(LightningWork):

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        print('RUN ANY PYTHON CODE HERE')
compute = CloudCompute('gpu', wait_timeout=60)
component = YourComponent(cloud_compute=compute)
app = LightningApp(component)