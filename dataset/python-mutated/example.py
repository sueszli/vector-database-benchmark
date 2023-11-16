import volatility.timefmt as timefmt
import volatility.obj as obj
import volatility.utils as utils
import volatility.commands as commands

class DateTime(commands.Command):
    """A simple example plugin that gets the date/time information from a Windows image"""

    def calculate(self):
        if False:
            for i in range(10):
                print('nop')
        'Calculate and carry out any processing that may take time upon the image'
        addr_space = utils.load_as(self._config)
        return self.get_image_time(addr_space)

    def get_image_time(self, addr_space):
        if False:
            for i in range(10):
                print('nop')
        'Extracts the time and date from the KUSER_SHARED_DATA area'
        result = {}
        KUSER_SHARED_DATA = obj.VolMagic(addr_space).KUSER_SHARED_DATA.v()
        k = obj.Object('_KUSER_SHARED_DATA', offset=KUSER_SHARED_DATA, vm=addr_space)
        result['ImageDatetime'] = k.SystemTime
        result['ImageTz'] = timefmt.OffsetTzInfo(-k.TimeZoneBias.as_windows_timestamp() / 10000000)
        return result

    def render_text(self, outfd, data):
        if False:
            while True:
                i = 10
        'Renders the calculated data as text to outfd'
        dt = data['ImageDatetime'].as_datetime()
        outfd.write('Image date and time       : {0}\n'.format(data['ImageDatetime']))
        outfd.write('Image local date and time : {0}\n'.format(timefmt.display_datetime(dt, data['ImageTz'])))