def foo():
    if False:
        for i in range(10):
            print('nop')
    click.echo(click.style(f'Detected project with {projects}\n', fg='blue', err=True))
    click.echo(f' $ {click.style(cmd, bold=True)}\n', err=True)