import inspect
import sys
import traceback
from typing import Any, Dict, Optional
from twisted.conch import manhole_ssh
from twisted.conch.insults import insults
from twisted.conch.manhole import ColoredManhole, ManholeInterpreter
from twisted.conch.ssh.keys import Key
from twisted.cred import checkers, portal
from twisted.internet import defer
from twisted.internet.protocol import ServerFactory
from synapse.config.server import ManholeConfig
PUBLIC_KEY = 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDHhGATaW4KhE23+7nrH4jFx3yLq9OjaEs5XALqeK+7385NlLja3DE/DO9mGhnd9+bAy39EKT3sTV6+WXQ4yD0TvEEyUEMtjWkSEm6U32+CDaS3TW/vPBUMeJQwq+Ydcif1UlnpXrDDTamD0AU9VaEvHq+3HAkipqn0TGpKON6aqk4vauDxoXSsV5TXBVrxP/y7HpMOpU4GUWsaaacBTKKNnUaQB4UflvydaPJUuwdaCUJGTMjbhWrjVfK+jslseSPxU6XvrkZMyCr4znxvuDxjMk1RGIdO7v+rbBMLEgqtSMNqJbYeVCnj2CFgc3fcTcldX2uOJDrJb/WRlHulthCh'
PRIVATE_KEY = '-----BEGIN RSA PRIVATE KEY-----\nMIIEpQIBAAKCAQEAx4RgE2luCoRNt/u56x+Ixcd8i6vTo2hLOVwC6nivu9/OTZS4\n2twxPwzvZhoZ3ffmwMt/RCk97E1evll0OMg9E7xBMlBDLY1pEhJulN9vgg2kt01v\n7zwVDHiUMKvmHXIn9VJZ6V6ww02pg9AFPVWhLx6vtxwJIqap9ExqSjjemqpOL2rg\n8aF0rFeU1wVa8T/8ux6TDqVOBlFrGmmnAUyijZ1GkAeFH5b8nWjyVLsHWglCRkzI\n24Vq41Xyvo7JbHkj8VOl765GTMgq+M58b7g8YzJNURiHTu7/q2wTCxIKrUjDaiW2\nHlQp49ghYHN33E3JXV9rjiQ6yW/1kZR7pbYQoQIDAQABAoIBAQC8KJ0q8Wzzwh5B\nesa1dQHZ8+4DEsL/Amae66VcVwD0X3cCN1W2IZ7X5W0Ij2kBqr8V51RYhcR+S+Ek\nBtzSiBUBvbKGrqcMGKaUgomDIMzai99hd0gvCCyZnEW1OQhFkNkaRNXCfqiZJ27M\nfqvSUiU2eOwh9fCvmxoA6Of8o3FbzcJ+1GMcobWRllDtLmj6lgVbDzuA+0jC5daB\n9Tj1pBzu3wn3ufxiS+gBnJ+7NcXH3E73lqCcPa2ufbZ1haxfiGCnRIhFXuQDgxFX\nvKdEfDgtvas6r1ahGbc+b/q8E8fZT7cABuIU4yfOORK+MhpyWbvoyyzuVGKj3PKt\nKSPJu5CZAoGBAOkoJfAVyYteqKcmGTanGqQnAY43CaYf6GdSPX/jg+JmKZg0zqMC\njWZUtPb93i+jnOInbrnuHOiHAxI8wmhEPed28H2lC/LU8PzlqFkZXKFZ4vLOhhRB\n/HeHCFIDosPFlohWi3b+GAjD7sXgnIuGmnXWe2ea/TS3yersifDEoKKjAoGBANsQ\ngJX2cJv1c3jhdgcs8vAt5zIOKcCLTOr/QPmVf/kxjNgndswcKHwsxE/voTO9q+TF\nv/6yCSTxAdjuKz1oIYWgi/dZo82bBKWxNRpgrGviU3/zwxiHlyIXUhzQu78q3VS/\n7S1XVbc7qMV++XkYKHPVD+nVG/gGzFxumX7MLXfrAoGBAJit9cn2OnjNj9uFE1W6\nr7N254ndeLAUjPe73xH0RtTm2a4WRopwjW/JYIetTuYbWgyujc+robqTTuuOZjAp\nH/CG7o0Ym251CypQqaFO/l2aowclPp/dZhpPjp9GSjuxFBZLtiBB3DNBOwbRQzIK\n/vLTdRQvZkgzYkI4i0vjNt3JAoGBANP8HSKBLymMlShlrSx2b8TB9tc2Y2riohVJ\n2ttqs0M2kt/dGJWdrgOz4mikL+983Olt/0P9juHDoxEEMK2kpcPEv40lnmBpYU7h\ns8yJvnBLvJe2EJYdJ8AipyAhUX1FgpbvfxmASP8eaUxsegeXvBWTGWojAoS6N2o+\n0KSl+l3vAoGAFqm0gO9f/Q1Se60YQd4l2PZeMnJFv0slpgHHUwegmd6wJhOD7zJ1\nCkZcXwiv7Nog7AI9qKJEUXLjoqL+vJskBzSOqU3tcd670YQMi1aXSXJqYE202K7o\nEddTrx3TNpr1D5m/f+6mnXWrc8u9y1+GNx9yz889xMjIBTBI9KqaaOs=\n-----END RSA PRIVATE KEY-----'

def manhole(settings: ManholeConfig, globals: Dict[str, Any]) -> ServerFactory:
    if False:
        return 10
    'Starts a ssh listener with password authentication using\n    the given username and password. Clients connecting to the ssh\n    listener will find themselves in a colored python shell with\n    the supplied globals.\n\n    Args:\n        username: The username ssh clients should auth with.\n        password: The password ssh clients should auth with.\n        globals: The variables to expose in the shell.\n\n    Returns:\n        A factory to pass to ``listenTCP``\n    '
    username = settings.username
    password = settings.password.encode('ascii')
    priv_key = settings.priv_key
    if priv_key is None:
        priv_key = Key.fromString(PRIVATE_KEY)
    pub_key = settings.pub_key
    if pub_key is None:
        pub_key = Key.fromString(PUBLIC_KEY)
    checker = checkers.InMemoryUsernamePasswordDatabaseDontUse(**{username: password})
    rlm = manhole_ssh.TerminalRealm()
    rlm.chainedProtocolFactory = lambda : insults.ServerProtocol(SynapseManhole, dict(globals, __name__='__console__'))
    factory = manhole_ssh.ConchFactory(portal.Portal(rlm, [checker]))
    factory.privateKeys[b'ssh-rsa'] = priv_key
    factory.publicKeys[b'ssh-rsa'] = pub_key
    return factory

class SynapseManhole(ColoredManhole):
    """Overrides connectionMade to create our own ManholeInterpreter"""

    def connectionMade(self) -> None:
        if False:
            while True:
                i = 10
        super().connectionMade()
        self.interpreter = SynapseManholeInterpreter(self, self.namespace)

class SynapseManholeInterpreter(ManholeInterpreter):

    def showsyntaxerror(self, filename: Optional[str]=None) -> None:
        if False:
            while True:
                i = 10
        'Display the syntax error that just occurred.\n\n        Overrides the base implementation, ignoring sys.excepthook. We always want\n        any syntax errors to be sent to the terminal, rather than sentry.\n        '
        (type, value, tb) = sys.exc_info()
        assert value is not None
        sys.last_type = type
        sys.last_value = value
        sys.last_traceback = tb
        if filename and type is SyntaxError:
            try:
                (msg, (dummy_filename, lineno, offset, line)) = value.args
            except ValueError:
                pass
            else:
                value = SyntaxError(msg, (filename, lineno, offset, line))
                sys.last_value = value
        lines = traceback.format_exception_only(type, value)
        self.write(''.join(lines))

    def showtraceback(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Display the exception that just occurred.\n\n        Overrides the base implementation, ignoring sys.excepthook. We always want\n        any syntax errors to be sent to the terminal, rather than sentry.\n        '
        (sys.last_type, sys.last_value, last_tb) = ei = sys.exc_info()
        sys.last_traceback = last_tb
        assert last_tb is not None
        try:
            lines = traceback.format_exception(ei[0], ei[1], last_tb.tb_next)
            self.write(''.join(lines))
        finally:
            last_tb = ei = None

    def displayhook(self, obj: Any) -> None:
        if False:
            return 10
        "\n        We override the displayhook so that we automatically convert coroutines\n        into Deferreds. (Our superclass' displayhook will take care of the rest,\n        by displaying the Deferred if it's ready, or registering a callback\n        if it's not).\n        "
        if inspect.iscoroutine(obj):
            super().displayhook(defer.ensureDeferred(obj))
        else:
            super().displayhook(obj)