from metaflow import FlowSpec, step, kubernetes, retry

class HelloCloudFlow(FlowSpec):
    """
    A flow where Metaflow prints 'Metaflow says Hi from the cloud!'

    Run this flow to validate your Kubernetes configuration.

    """

    @step
    def start(self):
        if False:
            return 10
        "\n        The 'start' step is a regular step, so runs locally on the machine from\n        which the flow is executed.\n\n        "
        from metaflow import get_metadata
        print('HelloCloud is starting.')
        print('')
        print('Using metadata provider: %s' % get_metadata())
        print('')
        print('The start step is running locally. Next, the ')
        print("'hello' step will run remotely on Kubernetes. ")
        self.next(self.hello)

    @kubernetes(cpu=1, memory=500)
    @retry
    @step
    def hello(self):
        if False:
            print('Hello World!')
        '\n        This steps runs remotely on Kubernetes using 1 virtual CPU and 500Mb of\n        memory. Since we are now using a remote metadata service and data\n        store, the flow information and artifacts are available from\n        anywhere. The step also uses the retry decorator, so that if something\n        goes wrong, the step will be automatically retried.\n\n        '
        self.message = 'Hi from the cloud!'
        print('Metaflow says: %s' % self.message)
        self.next(self.end)

    @step
    def end(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The 'end' step is a regular step, so runs locally on the machine from\n        which the flow is executed.\n\n        "
        print('HelloCloud is finished.')
if __name__ == '__main__':
    HelloCloudFlow()