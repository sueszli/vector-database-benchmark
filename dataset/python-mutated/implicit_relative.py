from shared_libs.bar import bar
from shared_libs.foo import foo
import prefect

@prefect.flow
def foobar():
    if False:
        while True:
            i = 10
    return foo() + bar()