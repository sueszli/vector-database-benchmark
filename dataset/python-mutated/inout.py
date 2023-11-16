import click

@click.command()
@click.argument('input', type=click.File('rb'), nargs=-1)
@click.argument('output', type=click.File('wb'))
def cli(input, output):
    if False:
        while True:
            i = 10
    'This script works similar to the Unix `cat` command but it writes\n    into a specific file (which could be the standard output as denoted by\n    the ``-`` sign).\n\n    \x08\n    Copy stdin to stdout:\n        inout - -\n\n    \x08\n    Copy foo.txt and bar.txt to stdout:\n        inout foo.txt bar.txt -\n\n    \x08\n    Write stdin into the file foo.txt\n        inout - foo.txt\n    '
    for f in input:
        while True:
            chunk = f.read(1024)
            if not chunk:
                break
            output.write(chunk)
            output.flush()