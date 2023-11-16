"""
Contains console consumers for outputting XRay information back to console/terminal
"""
import click
from samcli.lib.observability.observability_info_puller import ObservabilityEventConsumer
from samcli.lib.observability.xray_traces.xray_events import XRayTraceEvent

class XRayTraceConsoleConsumer(ObservabilityEventConsumer[XRayTraceEvent]):
    """
    An XRayTraceEvent consumer which will output incoming XRayTraceEvent and print it back to console
    """

    def consume(self, event: XRayTraceEvent):
        if False:
            while True:
                i = 10
        click.echo(event.message)