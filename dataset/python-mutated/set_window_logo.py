import os
from base64 import standard_b64decode, standard_b64encode
from io import BytesIO
from typing import TYPE_CHECKING, Optional
from kitty.types import AsyncResponse
from kitty.utils import is_png
from .base import MATCH_WINDOW_OPTION, ArgsType, Boss, CmdGenerator, ImageCompletion, PayloadGetType, PayloadType, RCOptions, RemoteCommand, ResponseType, Window
if TYPE_CHECKING:
    from kitty.cli_stub import SetWindowLogoRCOptions as CLIOptions

class SetWindowLogo(RemoteCommand):
    protocol_spec = __doc__ = '\n    data+/str: Chunk of PNG data, base64 encoded no more than 2048 bytes. Must send an empty chunk to indicate end of image.     Or the special value :code:`-` to indicate image must be removed.\n    position/str: The logo position as a string, empty string means default\n    alpha/float: The logo alpha between :code:`0` and :code:`1`. :code:`-1` means use default\n    match/str: Which window to change the logo in\n    self/bool: Boolean indicating whether to act on the window the command is run in\n    '
    short_desc = 'Set the window logo'
    desc = 'Set the logo image for the specified windows. You must specify the path to a PNG image that will be used as the logo. If you specify the special value :code:`none` then any existing logo will be removed.'
    options_spec = MATCH_WINDOW_OPTION + "\n\n--self\ntype=bool-set\nAct on the window this command is run in, rather than the active window.\n\n\n--position\nThe position for the window logo. See :opt:`window_logo_position`.\n\n\n--alpha\ntype=float\ndefault=-1\nThe amount the window logo should be faded into the background.\nSee :opt:`window_logo_position`.\n\n\n--no-response\ntype=bool-set\ndefault=false\nDon't wait for a response from kitty. This means that even if setting the image\nfailed, the command will exit with a success code.\n"
    args = RemoteCommand.Args(spec='PATH_TO_PNG_IMAGE', count=1, json_field='data', special_parse='!read_window_logo(io_data, args[0])', completion=ImageCompletion)
    reads_streaming_data = True

    def message_to_kitty(self, global_opts: RCOptions, opts: 'CLIOptions', args: ArgsType) -> PayloadType:
        if False:
            return 10
        if len(args) != 1:
            self.fatal('Must specify path to exactly one PNG image')
        path = os.path.expanduser(args[0])
        import secrets
        ret = {'match': opts.match, 'self': opts.self, 'alpha': opts.alpha, 'position': opts.position, 'stream_id': secrets.token_urlsafe()}
        if path.lower() == 'none':
            ret['data'] = '-'
            return ret
        if not is_png(path):
            self.fatal(f'{path} is not a PNG image')

        def file_pipe(path: str) -> CmdGenerator:
            if False:
                i = 10
                return i + 15
            with open(path, 'rb') as f:
                while True:
                    data = f.read(512)
                    if not data:
                        break
                    ret['data'] = standard_b64encode(data).decode('ascii')
                    yield ret
            ret['data'] = ''
            yield ret
        return file_pipe(path)

    def response_from_kitty(self, boss: Boss, window: Optional[Window], payload_get: PayloadGetType) -> ResponseType:
        if False:
            print('Hello World!')
        data = payload_get('data')
        alpha = float(payload_get('alpha', '-1'))
        position = payload_get('position') or ''
        if data == '-':
            path = ''
            tfile = BytesIO()
        else:
            q = self.handle_streamed_data(standard_b64decode(data) if data else b'', payload_get)
            if isinstance(q, AsyncResponse):
                return q
            import hashlib
            path = '/from/remote/control/' + hashlib.sha1(q.getvalue()).hexdigest()
            tfile = q
        for window in self.windows_for_match_payload(boss, window, payload_get):
            if window:
                window.set_logo(path, position, alpha, tfile.getvalue())
        return None
set_window_logo = SetWindowLogo()