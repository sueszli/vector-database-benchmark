"""
    :codeauthor: Thomas Stoner <tmstoner@cisco.com>
"""
import re
from string import Template

class NXOSPlatform:
    """Cisco Systems Base Platform Unit Test Object"""
    chassis = 'Unknown NXOS Chassis'
    upgrade_required = False
    show_install_all_impact_no_module_data = '\nInstaller will perform impact only check. Please wait.\n\nVerifying image bootflash:/$IMAGE for boot variable "nxos".\n[####################] 100% -- SUCCESS\n\nVerifying image type.\n[####################] 100% -- SUCCESS\n\nPreparing "nxos" version info using image bootflash:/$IMAGE.\n[####################] 100% -- SUCCESS\n\nPreparing "bios" version info using image bootflash:/$IMAGE.\n[####################] 100% -- SUCCESS\n\nPerforming module support checks.\n[####################] 100% -- SUCCESS\n\nNotifying services about system upgrade.\n[####################] 100% -- SUCCESS\n\n\n\nCompatibility check is done:\nModule  bootable          Impact  Install-type  Reason\n------  --------  --------------  ------------  ------\n     1       yes      disruptive         reset  default upgrade is not hitless\n\n\n\nImages will be upgraded according to following table:\nModule       Image                  Running-Version(pri:alt)           New-Version  Upg-Required\n------  ----------  ----------------------------------------  --------------------  ------------\n    '
    internal_server_error_500 = '\n    Code: 500\n    '
    invalid_command = "\n    % Invalid command at '^' marker.\n    "
    internal_server_error_500_dict = {'code': '500', 'cli_error': internal_server_error_500}
    bad_request_client_error_400_invalid_command_dict = {'code': '400', 'cli_error': invalid_command}
    backend_processing_error_500 = internal_server_error_500_dict
    show_install_all_impact_in_progress = '\n    Installer will perform impact only check. Please wait.\n    Another install procedure may be in progress. (0x401E0007)\n    '
    bad_request_client_error_400_in_progress_dict = {'code': '400', 'cli_error': show_install_all_impact_in_progress}
    show_install_all_impact = None
    install_all_disruptive_success = None
    show_install_all_impact_non_disruptive = None
    install_all_non_disruptive_success = None

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        '\n        ckimage - current kickstart image\n        cimage - current system image\n        nkimage - new kickstart image\n        nimage - new system image\n        '
        self.ckimage = kwargs.get('ckimage', None)
        self.cimage = kwargs.get('cimage', None)
        self.nkimage = kwargs.get('nkimage', None)
        self.nimage = kwargs.get('nimage', None)
        self.ckversion = self.version_from_image(self.ckimage)
        self.cversion = self.version_from_image(self.cimage)
        self.nkversion = self.version_from_image(self.nkimage)
        self.nversion = self.version_from_image(self.nimage)
        self.upgrade_required = self.cversion != self.nversion
        values = {'KIMAGE': self.nkimage, 'IMAGE': self.nimage, 'CKVER': self.ckversion, 'CVER': self.cversion, 'NKVER': self.nkversion, 'NVER': self.nversion, 'REQ': 'no' if self.cversion == self.nversion else 'yes', 'KREQ': 'no' if self.ckversion == self.nkversion else 'yes'}
        if self.show_install_all_impact_no_module_data:
            self.show_install_all_impact_no_module_data = self.templatize(self.show_install_all_impact_no_module_data, values)
        if self.show_install_all_impact:
            self.show_install_all_impact = self.templatize(self.show_install_all_impact, values)
        if self.show_install_all_impact_non_disruptive:
            self.show_install_all_impact_non_disruptive = self.templatize(self.show_install_all_impact_non_disruptive, values)
        if self.install_all_non_disruptive_success:
            self.install_all_non_disruptive_success = self.templatize(self.install_all_non_disruptive_success, values)
        if self.install_all_disruptive_success:
            self.install_all_disruptive_success = self.templatize(self.install_all_disruptive_success, values)

    @staticmethod
    def templatize(template, values):
        if False:
            i = 10
            return i + 15
        'Substitute variables in template with their corresponding values'
        return Template(template).substitute(values)

    @staticmethod
    def version_from_image(image):
        if False:
            while True:
                i = 10
        'Given a NXOS image named image decompose to appropriate image version'
        ver = None
        if image:
            match_object = re.search('^.*\\.(\\d+)\\.(\\d+)\\.(\\d+)\\.(\\d+|[A-Z][0-9])\\.(?:bin)?(\\d+)?.*', image)
            try:
                ver = match_object.group(1)
                ver += '.' + match_object.group(2)
                if match_object.groups()[-1]:
                    ver += '(' + match_object.group(3) + ')'
                    ver += match_object.group(4)
                    ver += '(' + match_object.group(5) + ')'
                else:
                    ver += '(' + match_object.group(3) + '.' + match_object.group(4) + ')'
            except IndexError:
                return None
        return ver