import typing as t
import click

class OctaviaCommand(click.Command):

    def make_context(self, info_name: t.Optional[str], args: t.List[str], parent: t.Optional[click.Context]=None, **extra: t.Any) -> click.Context:
        if False:
            return 10
        'Wrap parent make context with telemetry sending in case of failure.\n\n        Args:\n            info_name (t.Optional[str]): The info name for this invocation.\n            args (t.List[str]): The arguments to parse as list of strings.\n            parent (t.Optional[click.Context], optional): The parent context if available.. Defaults to None.\n\n        Raises:\n            e: Raise whatever exception that was caught.\n\n        Returns:\n            click.Context: The built context.\n        '
        try:
            return super().make_context(info_name, args, parent, **extra)
        except Exception as e:
            telemetry_client = parent.obj['TELEMETRY_CLIENT']
            if isinstance(e, click.exceptions.Exit) and e.exit_code == 0:
                telemetry_client.send_command_telemetry(parent, extra_info_name=info_name, is_help=True)
            else:
                telemetry_client.send_command_telemetry(parent, error=e, extra_info_name=info_name)
            raise e

    def invoke(self, ctx: click.Context) -> t.Any:
        if False:
            print('Hello World!')
        'Wrap parent invoke by sending telemetry in case of success or failure.\n\n        Args:\n            ctx (click.Context): The invocation context.\n\n        Raises:\n            e: Raise whatever exception that was caught.\n\n        Returns:\n            t.Any: The invocation return value.\n        '
        telemetry_client = ctx.obj['TELEMETRY_CLIENT']
        try:
            result = super().invoke(ctx)
        except Exception as e:
            telemetry_client.send_command_telemetry(ctx, error=e)
            raise e
        telemetry_client.send_command_telemetry(ctx)
        return result