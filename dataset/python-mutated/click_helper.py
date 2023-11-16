import click
import pandas as pd

class Date(click.ParamType):

    def __init__(self, tz=None):
        if False:
            return 10
        self.tz = tz

    def convert(self, value, param, ctx):
        if False:
            print('Hello World!')
        return pd.Timestamp(value)

    @property
    def name(self):
        if False:
            print('Hello World!')
        return type(self).__name__.upper()