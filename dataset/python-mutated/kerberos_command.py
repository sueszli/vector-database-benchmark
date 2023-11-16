"""Kerberos command."""
from __future__ import annotations
from airflow import settings
from airflow.cli.commands.daemon_utils import run_command_with_daemon_option
from airflow.security import kerberos as krb
from airflow.security.kerberos import KerberosMode
from airflow.utils import cli as cli_utils
from airflow.utils.providers_configuration_loader import providers_configuration_loaded

@cli_utils.action_cli
@providers_configuration_loaded
def kerberos(args):
    if False:
        while True:
            i = 10
    'Start a kerberos ticket renewer.'
    print(settings.HEADER)
    mode = KerberosMode.STANDARD
    if args.one_time:
        mode = KerberosMode.ONE_TIME
    run_command_with_daemon_option(args=args, process_name='kerberos', callback=lambda : krb.run(principal=args.principal, keytab=args.keytab, mode=mode))