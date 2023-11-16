__license__ = 'GPL 3'
__copyright__ = '2009, John Schember <john@nachtimwald.com>'
__docformat__ = 'restructuredtext en'
from calibre.utils.config_base import Config, ConfigProxy

class DeviceConfig:
    HELP_MESSAGE = _('Configure device')
    EXTRA_CUSTOMIZATION_MESSAGE = None
    EXTRA_CUSTOMIZATION_DEFAULT = None
    EXTRA_CUSTOMIZATION_CHOICES = None
    SUPPORTS_SUB_DIRS = False
    SUPPORTS_SUB_DIRS_FOR_SCAN = False
    SUPPORTS_SUB_DIRS_DEFAULT = True
    MUST_READ_METADATA = False
    SUPPORTS_USE_AUTHOR_SORT = False
    SAVE_TEMPLATE = None
    USER_CAN_ADD_NEW_FORMATS = True

    @classmethod
    def _default_save_template(cls):
        if False:
            i = 10
            return i + 15
        from calibre.library.save_to_disk import config
        return cls.SAVE_TEMPLATE if cls.SAVE_TEMPLATE else config().parse().send_template

    @classmethod
    def _config_base_name(cls):
        if False:
            for i in range(10):
                print('nop')
        klass = cls if isinstance(cls, type) else cls.__class__
        return klass.__name__

    @classmethod
    def _config(cls):
        if False:
            while True:
                i = 10
        name = cls._config_base_name()
        c = Config('device_drivers_%s' % name, _('settings for device drivers'))
        c.add_opt('format_map', default=cls.FORMATS, help=_('Ordered list of formats the device will accept'))
        c.add_opt('use_subdirs', default=cls.SUPPORTS_SUB_DIRS_DEFAULT, help=_('Place files in sub-folders if the device supports them'))
        c.add_opt('read_metadata', default=True, help=_('Read metadata from files on device'))
        c.add_opt('use_author_sort', default=False, help=_('Use author sort instead of author'))
        c.add_opt('save_template', default=cls._default_save_template(), help=_('Template to control how books are saved'))
        c.add_opt('extra_customization', default=cls.EXTRA_CUSTOMIZATION_DEFAULT, help=_('Extra customization'))
        return c

    @classmethod
    def _configProxy(cls):
        if False:
            print('Hello World!')
        return ConfigProxy(cls._config())

    @classmethod
    def config_widget(cls):
        if False:
            i = 10
            return i + 15
        from calibre.gui2.device_drivers.configwidget import ConfigWidget
        cw = ConfigWidget(cls.settings(), cls.FORMATS, cls.SUPPORTS_SUB_DIRS, cls.MUST_READ_METADATA, cls.SUPPORTS_USE_AUTHOR_SORT, cls.EXTRA_CUSTOMIZATION_MESSAGE, cls, extra_customization_choices=cls.EXTRA_CUSTOMIZATION_CHOICES)
        return cw

    @classmethod
    def save_settings(cls, config_widget):
        if False:
            print('Hello World!')
        proxy = cls._configProxy()
        proxy['format_map'] = config_widget.format_map()
        if cls.SUPPORTS_SUB_DIRS:
            proxy['use_subdirs'] = config_widget.use_subdirs()
        if not cls.MUST_READ_METADATA:
            proxy['read_metadata'] = config_widget.read_metadata()
        if cls.SUPPORTS_USE_AUTHOR_SORT:
            proxy['use_author_sort'] = config_widget.use_author_sort()
        if cls.EXTRA_CUSTOMIZATION_MESSAGE:
            if isinstance(cls.EXTRA_CUSTOMIZATION_MESSAGE, list):
                ec = []
                for i in range(0, len(cls.EXTRA_CUSTOMIZATION_MESSAGE)):
                    if config_widget.opt_extra_customization[i] is None:
                        ec.append(None)
                        continue
                    if hasattr(config_widget.opt_extra_customization[i], 'isChecked'):
                        ec.append(config_widget.opt_extra_customization[i].isChecked())
                    elif hasattr(config_widget.opt_extra_customization[i], 'currentText'):
                        ec.append(str(config_widget.opt_extra_customization[i].currentText()).strip())
                    else:
                        ec.append(str(config_widget.opt_extra_customization[i].text()).strip())
            else:
                ec = str(config_widget.opt_extra_customization.text()).strip()
                if not ec:
                    ec = None
            proxy['extra_customization'] = ec
        st = str(config_widget.opt_save_template.text())
        proxy['save_template'] = st

    @classmethod
    def migrate_extra_customization(cls, vals):
        if False:
            for i in range(10):
                print('nop')
        return vals

    @classmethod
    def settings(cls):
        if False:
            for i in range(10):
                print('nop')
        opts = cls._config().parse()
        if isinstance(cls.EXTRA_CUSTOMIZATION_DEFAULT, list):
            if opts.extra_customization is None:
                opts.extra_customization = []
            if not isinstance(opts.extra_customization, list):
                opts.extra_customization = [opts.extra_customization]
            for (i, d) in enumerate(cls.EXTRA_CUSTOMIZATION_DEFAULT):
                if i >= len(opts.extra_customization):
                    opts.extra_customization.append(d)
            opts.extra_customization = cls.migrate_extra_customization(opts.extra_customization)
        return opts

    @classmethod
    def save_template(cls):
        if False:
            while True:
                i = 10
        st = cls.settings().save_template
        if st:
            return st
        else:
            return cls._default_save_template()

    @classmethod
    def customization_help(cls, gui=False):
        if False:
            while True:
                i = 10
        return cls.HELP_MESSAGE