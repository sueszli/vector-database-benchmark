"""Battery plugin."""
import psutil
from glances.logger import logger
from glances.plugins.plugin.model import GlancesPluginModel
batinfo_tag = True
try:
    import batinfo
except ImportError:
    logger.debug('batinfo library not found. Fallback to psutil.')
    batinfo_tag = False
psutil_tag = True
try:
    psutil.sensors_battery()
except Exception as e:
    logger.error('Cannot grab battery status {}.'.format(e))
    psutil_tag = False

class PluginModel(GlancesPluginModel):
    """Glances battery capacity plugin.

    stats is a list
    """

    def __init__(self, args=None, config=None):
        if False:
            return 10
        'Init the plugin.'
        super(PluginModel, self).__init__(args=args, config=config, stats_init_value=[])
        try:
            self.glances_grab_bat = GlancesGrabBat()
        except Exception as e:
            logger.error('Can not init battery class ({})'.format(e))
            global batinfo_tag
            global psutil_tag
            batinfo_tag = False
            psutil_tag = False
        self.display_curse = False

    @GlancesPluginModel._log_result_decorator
    def update(self):
        if False:
            print('Hello World!')
        'Update battery capacity stats using the input method.'
        stats = self.get_init_value()
        if self.input_method == 'local':
            self.glances_grab_bat.update()
            stats = self.glances_grab_bat.get()
        elif self.input_method == 'snmp':
            pass
        self.stats = stats
        return self.stats

class GlancesGrabBat(object):
    """Get batteries stats using the batinfo library."""

    def __init__(self):
        if False:
            print('Hello World!')
        'Init batteries stats.'
        self.bat_list = []
        if batinfo_tag:
            self.bat = batinfo.batteries()
        elif psutil_tag:
            self.bat = psutil
        else:
            self.bat = None

    def update(self):
        if False:
            for i in range(10):
                print('nop')
        'Update the stats.'
        self.bat_list = []
        if batinfo_tag:
            self.bat.update()
            for b in self.bat.stat:
                self.bat_list.append({'label': 'BAT {}'.format(b.path.split('/')[-1]), 'value': b.capacity, 'unit': '%', 'status': b.status})
        elif psutil_tag and hasattr(self.bat.sensors_battery(), 'percent'):
            self.bat_list = [{'label': 'Battery', 'value': int(self.bat.sensors_battery().percent), 'unit': '%', 'status': 'Charging' if self.bat.sensors_battery().power_plugged else 'Discharging'}]

    def get(self):
        if False:
            i = 10
            return i + 15
        'Get the stats.'
        return self.bat_list

    @property
    def battery_percent(self):
        if False:
            i = 10
            return i + 15
        'Get batteries capacity percent.'
        if not batinfo_tag or not self.bat.stat:
            return []
        b_sum = 0
        for b in self.bat.stat:
            try:
                b_sum += int(b.capacity)
            except ValueError:
                return []
        return int(b_sum / len(self.bat.stat))