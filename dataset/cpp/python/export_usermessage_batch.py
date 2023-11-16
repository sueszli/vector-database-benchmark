import glob
import logging
import os
from argparse import ArgumentParser
from typing import Any

from django.core.management.base import BaseCommand
from typing_extensions import override

from zerver.lib.export import export_usermessages_batch


class Command(BaseCommand):
    help = """UserMessage fetching helper for export.py"""

    @override
    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--path", help="Path to find messages.json archives")
        parser.add_argument("--thread", help="Thread ID")
        parser.add_argument(
            "--consent-message-id",
            type=int,
            help="ID of the message advertising users to react with thumbs up",
        )

    @override
    def handle(self, *args: Any, **options: Any) -> None:
        logging.info("Starting UserMessage batch thread %s", options["thread"])
        files = set(glob.glob(os.path.join(options["path"], "messages-*.json.partial")))
        for partial_path in files:
            locked_path = partial_path.replace(".json.partial", ".json.locked")
            output_path = partial_path.replace(".json.partial", ".json")
            try:
                os.rename(partial_path, locked_path)
            except FileNotFoundError:
                # Already claimed by another process
                continue
            logging.info("Thread %s processing %s", options["thread"], output_path)
            try:
                export_usermessages_batch(locked_path, output_path, options["consent_message_id"])
            except BaseException:
                # Put the item back in the free pool when we fail
                os.rename(locked_path, partial_path)
                raise
