from lightning.app import LightningWork, LightningApp, CloudCompute

class YourComponent(LightningWork):

    def run(self):
        if False:
            i = 10
            return i + 15
        print('RUN ANY PYTHON CODE HERE')
compute = CloudCompute('gpu', disk_size=100)
component = YourComponent(cloud_compute=compute)
app = LightningApp(component)