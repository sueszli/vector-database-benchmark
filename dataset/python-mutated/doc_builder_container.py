from ci.ray_ci.container import Container

class DocBuilderContainer(Container):

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__('docbuild')
        self.install_ray()

    def run(self) -> None:
        if False:
            while True:
                i = 10
        self.run_script(['cd doc', 'FAST=True make html'])