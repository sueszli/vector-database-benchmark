import reflex as rx
from ..state import State

class FormState(State):
    """Form state."""
    form_data: dict = {}

    def handle_submit(self, form_data: dict):
        if False:
            i = 10
            return i + 15
        'Handle the form submit.\n\n        Args:\n            form_data: The form data.\n        '
        self.form_data = form_data

class UploadState(State):
    """The app state."""
    img: list[str]

    async def handle_upload(self, files: list[rx.UploadFile]):
        """Handle the upload of file(s).

        Args:
            files: The uploaded files.
        """
        for file in files:
            upload_data = await file.read()
            outfile = rx.get_asset_path(file.filename)
            with open(outfile, 'wb') as file_object:
                file_object.write(upload_data)
            self.img.append(f'/{file.filename}')