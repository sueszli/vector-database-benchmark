import asyncio
import click
from mitmproxy.addons import dumper
from mitmproxy.test import taddons
from mitmproxy.test import tflow

def run_async(coro):
    if False:
        i = 10
        return i + 15
    '\n    Run the given async function in a new event loop.\n    This allows async functions to be called synchronously.\n    '
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

def show(flow_detail, flows):
    if False:
        print('Hello World!')
    d = dumper.Dumper()
    with taddons.context() as ctx:
        ctx.configure(d, flow_detail=flow_detail)
        for f in flows:
            run_async(ctx.cycle(d, f))

@click.group()
def cli():
    if False:
        for i in range(10):
            print('nop')
    pass

@cli.command()
@click.option('--level', default=1, help='Detail level')
def tcp(level):
    if False:
        while True:
            i = 10
    f1 = tflow.ttcpflow()
    show(level, [f1])

@cli.command()
@click.option('--level', default=1, help='Detail level')
def udp(level):
    if False:
        while True:
            i = 10
    f1 = tflow.tudpflow()
    show(level, [f1])

@cli.command()
@click.option('--level', default=1, help='Detail level')
def large(level):
    if False:
        return 10
    f1 = tflow.tflow(resp=True)
    f1.response.headers['content-type'] = 'text/html'
    f1.response.content = b'foo bar voing\n' * 100
    show(level, [f1])

@cli.command()
@click.option('--level', default=1, help='Detail level')
def small(level):
    if False:
        print('Hello World!')
    f1 = tflow.tflow(resp=True)
    f1.response.headers['content-type'] = 'text/html'
    f1.response.content = b'<html><body>Hello!</body></html>'
    f2 = tflow.tflow(err=True)
    show(level, [f1, f2])
if __name__ == '__main__':
    cli()