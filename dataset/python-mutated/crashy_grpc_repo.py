import sys
from dagster import repository

@repository
def crashy_repo():
    if False:
        for i in range(10):
            print('nop')
    sys.exit(123)