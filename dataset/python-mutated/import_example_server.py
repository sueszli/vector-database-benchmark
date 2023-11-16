import example_resource
import hug

@hug.get()
def hello():
    if False:
        return 10
    return example_resource.hi()