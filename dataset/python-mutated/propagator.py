import os
import typing
from opentelemetry.context import Context
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.context.context import Context
from opentelemetry.propagators.textmap import DefaultGetter, DefaultSetter, Getter, Setter, TextMapPropagator, CarrierT

class EnvPropagator(TextMapPropagator):

    def __init__(self, formatter):
        if False:
            i = 10
            return i + 15
        if formatter is None:
            self.formatter = TraceContextTextMapPropagator()
        else:
            self.formatter = formatter

    def extract(self, carrier: CarrierT, context: typing.Optional[Context]=None, getter: Getter=DefaultGetter()) -> Context:
        if False:
            for i in range(10):
                print('nop')
        return self.formatter.extract(carrier=carrier, context=context, getter=getter)

    def inject(self, carrier: CarrierT, context: typing.Optional[Context]=None, setter: Setter=DefaultSetter()) -> None:
        if False:
            return 10
        self.formatter.inject(carrier=carrier, context=context, setter=setter)

    def inject_to_carrier(self, context: typing.Optional[Context]=None):
        if False:
            return 10
        env_dict = os.environ.copy()
        self.inject(carrier=env_dict, context=context, setter=DefaultSetter())
        return env_dict

    def extract_context(self) -> Context:
        if False:
            print('Hello World!')
        if self.formatter is None:
            self.formatter = TraceContextTextMapPropagator()
        return self.extract(carrier=os.environ, getter=DefaultGetter())

    @property
    def fields(self) -> typing.Set[str]:
        if False:
            return 10
        return self.formatter.fields