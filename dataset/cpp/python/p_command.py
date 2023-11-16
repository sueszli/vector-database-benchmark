import json
import pathlib
import time
from typing import cast

import click
import filetype
from tqdm import tqdm
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from ..bg import remove
from ..session_factory import new_session
from ..sessions import sessions_names


@click.command(  # type: ignore
    name="p",
    help="for a folder as input",
)
@click.option(
    "-m",
    "--model",
    default="u2net",
    type=click.Choice(sessions_names),
    show_default=True,
    show_choices=True,
    help="model name",
)
@click.option(
    "-a",
    "--alpha-matting",
    is_flag=True,
    show_default=True,
    help="use alpha matting",
)
@click.option(
    "-af",
    "--alpha-matting-foreground-threshold",
    default=240,
    type=int,
    show_default=True,
    help="trimap fg threshold",
)
@click.option(
    "-ab",
    "--alpha-matting-background-threshold",
    default=10,
    type=int,
    show_default=True,
    help="trimap bg threshold",
)
@click.option(
    "-ae",
    "--alpha-matting-erode-size",
    default=10,
    type=int,
    show_default=True,
    help="erode size",
)
@click.option(
    "-om",
    "--only-mask",
    is_flag=True,
    show_default=True,
    help="output only the mask",
)
@click.option(
    "-ppm",
    "--post-process-mask",
    is_flag=True,
    show_default=True,
    help="post process the mask",
)
@click.option(
    "-w",
    "--watch",
    default=False,
    is_flag=True,
    show_default=True,
    help="watches a folder for changes",
)
@click.option(
    "-bgc",
    "--bgcolor",
    default=None,
    type=(int, int, int, int),
    nargs=4,
    help="Background color (R G B A) to replace the removed background with",
)
@click.option("-x", "--extras", type=str)
@click.argument(
    "input",
    type=click.Path(
        exists=True,
        path_type=pathlib.Path,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
)
@click.argument(
    "output",
    type=click.Path(
        exists=False,
        path_type=pathlib.Path,
        file_okay=False,
        dir_okay=True,
        writable=True,
    ),
)
def p_command(
    model: str,
    extras: str,
    input: pathlib.Path,
    output: pathlib.Path,
    watch: bool,
    **kwargs,
) -> None:
    """
    Command-line interface (CLI) program for performing background removal on images in a folder.

    This program takes a folder as input and uses a specified model to remove the background from the images in the folder.
    It provides various options for configuration, such as choosing the model, enabling alpha matting, setting trimap thresholds, erode size, etc.
    Additional options include outputting only the mask and post-processing the mask.
    The program can also watch the input folder for changes and automatically process new images.
    The resulting images with the background removed are saved in the specified output folder.

    Parameters:
        model (str): The name of the model to use for background removal.
        extras (str): Additional options in JSON format.
        input (pathlib.Path): The path to the input folder.
        output (pathlib.Path): The path to the output folder.
        watch (bool): Whether to watch the input folder for changes.
        **kwargs: Additional keyword arguments.

    Returns:
        None
    """
    try:
        kwargs.update(json.loads(extras))
    except Exception:
        pass

    session = new_session(model, **kwargs)

    def process(each_input: pathlib.Path) -> None:
        try:
            mimetype = filetype.guess(each_input)
            if mimetype is None:
                return
            if mimetype.mime.find("image") < 0:
                return

            each_output = (output / each_input.name).with_suffix(".png")
            each_output.parents[0].mkdir(parents=True, exist_ok=True)

            if not each_output.exists():
                each_output.write_bytes(
                    cast(
                        bytes,
                        remove(each_input.read_bytes(), session=session, **kwargs),
                    )
                )

                if watch:
                    print(
                        f"processed: {each_input.absolute()} -> {each_output.absolute()}"
                    )
        except Exception as e:
            print(e)

    inputs = list(input.glob("**/*"))
    if not watch:
        inputs = tqdm(inputs)

    for each_input in inputs:
        if not each_input.is_dir():
            process(each_input)

    if watch:
        observer = Observer()

        class EventHandler(FileSystemEventHandler):
            def on_any_event(self, event: FileSystemEvent) -> None:
                if not (
                    event.is_directory or event.event_type in ["deleted", "closed"]
                ):
                    process(pathlib.Path(event.src_path))

        event_handler = EventHandler()
        observer.schedule(event_handler, input, recursive=False)
        observer.start()

        try:
            while True:
                time.sleep(1)

        finally:
            observer.stop()
            observer.join()
