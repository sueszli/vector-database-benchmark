"""Celery + :pypi:`cell` integration."""
from celery import bootsteps
from .connection import Connection
__all__ = ('Agent',)

class Agent(bootsteps.StartStopStep):
    """Agent starts :pypi:`cell` actors."""
    conditional = True
    requires = (Connection,)

    def __init__(self, c, **kwargs):
        if False:
            print('Hello World!')
        self.agent_cls = self.enabled = c.app.conf.worker_agent
        super().__init__(c, **kwargs)

    def create(self, c):
        if False:
            for i in range(10):
                print('nop')
        agent = c.agent = self.instantiate(self.agent_cls, c.connection)
        return agent