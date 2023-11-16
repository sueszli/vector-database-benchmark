import click

def test_command_context_class():
    if False:
        return 10
    'A command with a custom ``context_class`` should produce a\n    context using that type.\n    '

    class CustomContext(click.Context):
        pass

    class CustomCommand(click.Command):
        context_class = CustomContext
    command = CustomCommand('test')
    context = command.make_context('test', [])
    assert isinstance(context, CustomContext)

def test_context_invoke_type(runner):
    if False:
        for i in range(10):
            print('nop')
    'A command invoked from a custom context should have a new\n    context with the same type.\n    '

    class CustomContext(click.Context):
        pass

    class CustomCommand(click.Command):
        context_class = CustomContext

    @click.command()
    @click.argument('first_id', type=int)
    @click.pass_context
    def second(ctx, first_id):
        if False:
            return 10
        assert isinstance(ctx, CustomContext)
        assert id(ctx) != first_id

    @click.command(cls=CustomCommand)
    @click.pass_context
    def first(ctx):
        if False:
            print('Hello World!')
        assert isinstance(ctx, CustomContext)
        ctx.invoke(second, first_id=id(ctx))
    assert not runner.invoke(first).exception

def test_context_formatter_class():
    if False:
        i = 10
        return i + 15
    'A context with a custom ``formatter_class`` should format help\n    using that type.\n    '

    class CustomFormatter(click.HelpFormatter):

        def write_heading(self, heading):
            if False:
                for i in range(10):
                    print('nop')
            heading = click.style(heading, fg='yellow')
            return super().write_heading(heading)

    class CustomContext(click.Context):
        formatter_class = CustomFormatter
    context = CustomContext(click.Command('test', params=[click.Option(['--value'])]), color=True)
    assert '\x1b[33mOptions\x1b[0m:' in context.get_help()

def test_group_command_class(runner):
    if False:
        i = 10
        return i + 15
    'A group with a custom ``command_class`` should create subcommands\n    of that type by default.\n    '

    class CustomCommand(click.Command):
        pass

    class CustomGroup(click.Group):
        command_class = CustomCommand
    group = CustomGroup()
    subcommand = group.command()(lambda : None)
    assert type(subcommand) is CustomCommand
    subcommand = group.command(cls=click.Command)(lambda : None)
    assert type(subcommand) is click.Command

def test_group_group_class(runner):
    if False:
        while True:
            i = 10
    'A group with a custom ``group_class`` should create subgroups\n    of that type by default.\n    '

    class CustomSubGroup(click.Group):
        pass

    class CustomGroup(click.Group):
        group_class = CustomSubGroup
    group = CustomGroup()
    subgroup = group.group()(lambda : None)
    assert type(subgroup) is CustomSubGroup
    subgroup = group.command(cls=click.Group)(lambda : None)
    assert type(subgroup) is click.Group

def test_group_group_class_self(runner):
    if False:
        return 10
    'A group with ``group_class = type`` should create subgroups of\n    the same type as itself.\n    '

    class CustomGroup(click.Group):
        group_class = type
    group = CustomGroup()
    subgroup = group.group()(lambda : None)
    assert type(subgroup) is CustomGroup