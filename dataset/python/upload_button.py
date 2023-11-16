"""gr.UploadButton() component."""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from typing import Any, Callable, List, Literal

from gradio_client.documentation import document, set_documentation_group

from gradio.components.base import Component
from gradio.data_classes import FileData, GradioRootModel
from gradio.events import Events
from gradio.utils import NamedString

set_documentation_group("component")


class ListFiles(GradioRootModel):
    root: List[FileData]

    def __getitem__(self, index):
        return self.root[index]

    def __iter__(self):
        return iter(self.root)


@document()
class UploadButton(Component):
    """
    Used to create an upload button, when clicked allows a user to upload files that satisfy the specified file type or generic files (if file_type not set).
    Preprocessing: passes the uploaded file as a {file-object} or {List[file-object]} depending on `file_count` (or a {bytes}/{List[bytes]} depending on `type`)
    Postprocessing: expects function to return a {str} path to a file, or {List[str]} consisting of paths to files.
    Examples-format: a {str} path to a local file that populates the component.
    Demos: upload_button
    """

    EVENTS = [Events.click, Events.upload]

    def __init__(
        self,
        label: str = "Upload a File",
        value: str | list[str] | Callable | None = None,
        *,
        every: float | None = None,
        variant: Literal["primary", "secondary", "stop"] = "secondary",
        visible: bool = True,
        size: Literal["sm", "lg"] | None = None,
        scale: int | None = None,
        min_width: int | None = None,
        interactive: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        render: bool = True,
        type: Literal["filepath", "bytes"] = "filepath",
        file_count: Literal["single", "multiple", "directory"] = "single",
        file_types: list[str] | None = None,
    ):
        """
        Parameters:
            label: Text to display on the button. Defaults to "Upload a File".
            value: File or list of files to upload by default.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            variant: 'primary' for main call-to-action, 'secondary' for a more subdued style, 'stop' for a stop button.
            visible: If False, component will be hidden.
            size: Size of the button. Can be "sm" or "lg".
            scale: relative width compared to adjacent Components in a Row. For example, if Component A has scale=2, and Component B has scale=1, A will be twice as wide as B. Should be an integer.
            min_width: minimum pixel width, will wrap if not sufficient screen space to satisfy this value. If a certain scale value results in this Component being narrower than min_width, the min_width parameter will be respected first.
            interactive: If False, the UploadButton will be in a disabled state.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
            render: If False, component will not render be rendered in the Blocks context. Should be used if the intention is to assign event listeners now but render the component later.
            type: Type of value to be returned by component. "file" returns a temporary file object with the same base name as the uploaded file, whose full path can be retrieved by file_obj.name, "binary" returns an bytes object.
            file_count: if single, allows user to upload one file. If "multiple", user uploads multiple files. If "directory", user uploads all files in selected directory. Return type will be list for each file in case of "multiple" or "directory".
            file_types: List of type of files to be uploaded. "file" allows any file to be uploaded, "image" allows only image files to be uploaded, "audio" allows only audio files to be uploaded, "video" allows only video files to be uploaded, "text" allows only text files to be uploaded.
        """
        valid_types = [
            "filepath",
            "binary",
        ]
        if type not in valid_types:
            raise ValueError(
                f"Invalid value for parameter `type`: {type}. Please choose from one of: {valid_types}"
            )
        self.type = type
        self.file_count = file_count
        if file_count == "directory" and file_types is not None:
            warnings.warn(
                "The `file_types` parameter is ignored when `file_count` is 'directory'."
            )
        if file_types is not None and not isinstance(file_types, list):
            raise ValueError(
                f"Parameter file_types must be a list. Received {file_types.__class__.__name__}"
            )
        if self.file_count == "multiple":
            self.data_model = ListFiles
        else:
            self.data_model = FileData
        self.size = size
        self.file_types = file_types
        self.label = label
        self.variant = variant
        super().__init__(
            label=label,
            every=every,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            render=render,
            value=value,
            scale=scale,
            min_width=min_width,
            interactive=interactive,
        )

    def api_info(self) -> dict[str, list[str]]:
        if self.file_count == "single":
            return FileData.model_json_schema()
        else:
            return ListFiles.model_json_schema()

    def example_inputs(self) -> Any:
        if self.file_count == "single":
            return "https://github.com/gradio-app/gradio/raw/main/test/test_files/sample_file.pdf"
        else:
            return [
                "https://github.com/gradio-app/gradio/raw/main/test/test_files/sample_file.pdf"
            ]

    def _process_single_file(self, f: FileData) -> bytes | NamedString:
        file_name = f.path
        if self.type == "filepath":
            file = tempfile.NamedTemporaryFile(delete=False, dir=self.GRADIO_CACHE)
            file.name = file_name
            return NamedString(file_name)
        elif self.type == "binary":
            with open(file_name, "rb") as file_data:
                return file_data.read()
        else:
            raise ValueError(
                "Unknown type: "
                + str(type)
                + ". Please choose from: 'filepath', 'binary'."
            )

    def preprocess(
        self, payload: ListFiles | FileData | None
    ) -> bytes | NamedString | list[bytes | NamedString] | None:
        if payload is None:
            return None

        if self.file_count == "single":
            if isinstance(payload, ListFiles):
                return self._process_single_file(payload[0])
            else:
                return self._process_single_file(payload)
        else:
            if isinstance(payload, ListFiles):
                return [self._process_single_file(f) for f in payload]
            else:
                return [self._process_single_file(payload)]

    def postprocess(self, value: str | list[str] | None) -> ListFiles | FileData | None:
        if value is None:
            return None
        if isinstance(value, list):
            return ListFiles(
                root=[
                    FileData(
                        path=file,
                        orig_name=Path(file).name,
                        size=Path(file).stat().st_size,
                    )
                    for file in value
                ]
            )
        else:
            return FileData(
                path=value,
                orig_name=Path(value).name,
                size=Path(value).stat().st_size,
            )

    @property
    def skip_api(self):
        return False
