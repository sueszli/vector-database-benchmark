class PrintVolumesArgs:
    pass

class PrintVolumesList:

    def __init__(self, environ, volumes_listing):
        if False:
            while True:
                i = 10
        self.environ = environ
        self.volumes_listing = volumes_listing

    def exectute(self, args):
        if False:
            print('Hello World!')
        for volume in self.volumes_listing.list_volumes(self.environ):
            print(volume)