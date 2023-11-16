"""
All logic regarding the Operation Modes (opmodes).

The opmode is defined based on the user's arguments and the available
resources of the host system
"""
import argparse
import logging
import os
import sys
import pyric
import wifiphisher.common.constants as constants
import wifiphisher.common.interfaces as interfaces
import wifiphisher.extensions.handshakeverify as handshakeverify
logger = logging.getLogger(__name__)

class OpMode(object):
    """
    Manager of the operation mode
    """

    def __init__(self):
        if False:
            return 10
        '\n        Construct the class\n        :param self: An OpMode object\n        :type self: OpMode\n        :return: None\n        :rtype: None\n        '
        self.op_mode = 0
        self._use_one_phy = False
        self._perfect_card = None

    def initialize(self, args):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize the opmode manager\n        :param self: An OpMode object\n        :param args: An argparse.Namespace object\n        :type self: OpMode\n        :type args: argparse.Namespace\n        :return: None\n        :rtype: None\n        '
        (self._perfect_card, self._use_one_phy) = interfaces.is_add_vif_required(args.interface, args.internetinterface, args.wpspbc_assoc_interface)
        self._check_args(args)

    def _check_args(self, args):
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks the given arguments for logic errors.\n        :param self: An OpMode object\n        :param args: An argparse.Namespace object\n        :type self: OpMode\n        :type args: argparse.Namespace\n        :return: None\n        :rtype: None\n        '
        if args.presharedkey and (len(args.presharedkey) < 8 or len(args.presharedkey) > 64):
            sys.exit('[' + constants.R + '-' + constants.W + '] Pre-shared key must be between 8 and 63 printablecharacters.')
        if args.handshake_capture and (not os.path.isfile(args.handshake_capture)):
            sys.exit('[' + constants.R + '-' + constants.W + '] Handshake capture does not exist.')
        elif args.handshake_capture and (not handshakeverify.is_valid_handshake_capture(args.handshake_capture)):
            sys.exit('[' + constants.R + '-' + constants.W + '] Handshake capture does not contain valid handshake')
        if (args.extensionsinterface and (not args.apinterface) or (not args.extensionsinterface and args.apinterface)) and (not (args.noextensions and args.apinterface)):
            sys.exit('[' + constants.R + '-' + constants.W + '] --apinterface (-aI) and --extensionsinterface (-eI)(or --noextensions (-nE)) are used in conjuction.')
        if args.noextensions and args.extensionsinterface:
            sys.exit('[' + constants.R + '-' + constants.W + '] --noextensions (-nE) and --extensionsinterface (-eI)cannot work together.')
        if args.lure10_exploit and args.noextensions:
            sys.exit('[' + constants.R + '-' + constants.W + '] --lure10-exploit (-lE) and --noextensions (-eJ)cannot work together.')
        if args.lure10_exploit and (not os.path.isfile(constants.LOCS_DIR + args.lure10_exploit)):
            sys.exit('[' + constants.R + '-' + constants.W + '] Lure10 capture does not exist. Listing directoryof captures: ' + str(os.listdir(constants.LOCS_DIR)))
        if args.mac_ap_interface and args.no_mac_randomization or (args.mac_extensions_interface and args.no_mac_randomization):
            sys.exit('[' + constants.R + '-' + constants.W + '] --no-mac-randomization (-iNM) cannot work together with--mac-ap-interface or --mac-extensions-interface (-iDM)')
        if args.deauth_essid and args.noextensions:
            sys.exit('[' + constants.R + '-' + constants.W + '] --deauth-essid (-dE) cannot work together with--noextension (-nE)')
        if args.deauth_essid and self._use_one_phy:
            print('[' + constants.R + '!' + constants.W + '] Only one card was found. Wifiphisher will deauth only on the target AP channel')
        if args.wpspbc_assoc_interface and (not args.wps_pbc):
            sys.exit('[' + constants.R + '!' + constants.W + '] --wpspbc-assoc-interface (-wAI) requires --wps-pbc (-wP) option.')
        if args.logpath and (not args.logging):
            sys.exit('[' + constants.R + '!' + constants.W + '] --logpath (-lP) requires --logging option.')
        if args.credential_log_path and (not args.logging):
            sys.exit('[' + constants.R + '!' + constants.W + '] --credential-log-path (-cP) requires --logging option.')
        if args.deauth_channels:
            for channel in args.deauth_channels:
                if channel > 14 or channel < 0:
                    sys.exit('[' + constants.R + '!' + constants.W + '] --deauth-channels (-dC) requires channels in range 1-14.')
        if args.mitminterface and args.mitminterface != 'handledAsInternetInterface':
            print('[' + constants.O + '!' + constants.W + '] Using  both --mitminterface (-mI) and --internetinterface (-iI) is redundant. Ignoring --internetinterface (-iI).')

    def set_opmode(self, args, network_manager):
        if False:
            print('Hello World!')
        '\n        Sets the operation mode.\n\n        :param self: An OpMode object\n        :param args: An argparse.Namespace object\n        :param network_manager: A NetworkManager object\n        :type self: OpMode\n        :type args: argparse.Namespace\n        :type network_manager: NetworkManager\n        :return: None\n        :rtype: None\n\n        ..note: An operation mode resembles how the tool will best leverage\n        the given resources.\n\n        Modes of operation\n        1) AP and Extensions 0x1\n          2 cards, 2 interfaces\n          i) AP, ii) EM\n          Channel hopping: Enabled\n        2) AP, Extensions and Internet 0x2\n          3 cards, 3 interfaces\n          i) AP, ii) EM iii) Internet\n          Channel hopping: Enabled\n        3) AP-only and Internet 0x3\n          2 cards, 2 interfaces\n          i) AP, ii) Internet\n        4) AP-only 0x4\n          1 card, 1 interface\n          i) AP\n        5) AP and Extensions 0x5\n          1 card, 2 interfaces\n          (1 card w/ vif support AP/Monitor)\n          i) AP, ii) Extensions\n          Channel hopping: Disabled\n          !!Most common mode!!\n        6) AP and Extensions and Internet 0x6\n          2 cards, 3 interfaces\n          Channel hopping: Disabled\n          (Internet and 1 card w/ 1 vif support AP/Monitor)\n          i) AP, ii) Extensions, iii) Internet\n        7) Advanced and WPS association 0x7\n          3 cards, 3 interfaces\n          i) AP, ii) Extensions (Monitor), iii) Extensions (Managed)\n        8) Advanced and WPS association w/ 1 vif support AP/Monitor 0x8\n          2 cards, 3 interfaces\n          i) AP, ii) Extensions (Monitor), iii) Extensions (Managed)\n        '
        if not args.internetinterface and (not args.noextensions):
            if not self._use_one_phy:
                if args.wpspbc_assoc_interface:
                    self.op_mode = constants.OP_MODE7
                    logger.info('Starting OP_MODE7 (0x7)')
                else:
                    self.op_mode = constants.OP_MODE1
                    logger.info('Starting OP_MODE1 (0x1)')
            else:
                if self._perfect_card is not None:
                    network_manager.add_virtual_interface(self._perfect_card)
                if args.wpspbc_assoc_interface:
                    self.op_mode = constants.OP_MODE8
                    logger.info('Starting OP_MODE8 (0x8)')
                else:
                    self.op_mode = constants.OP_MODE5
                    logger.info('Starting OP_MODE5 (0x5)')
        if args.internetinterface and (not args.noextensions):
            if not self._use_one_phy:
                self.op_mode = constants.OP_MODE2
                logger.info('Starting OP_MODE2 (0x2)')
            else:
                if self._perfect_card is not None:
                    network_manager.add_virtual_interface(self._perfect_card)
                self.op_mode = constants.OP_MODE6
                logger.info('Starting OP_MODE6 (0x6)')
        if args.internetinterface and args.noextensions:
            self.op_mode = constants.OP_MODE3
            logger.info('Starting OP_MODE3 (0x3)')
        if args.noextensions and (not args.internetinterface):
            self.op_mode = constants.OP_MODE4
            logger.info('Starting OP_MODE4 (0x4)')

    def internet_sharing_enabled(self):
        if False:
            print('Hello World!')
        '\n        :param self: An OpMode object\n        :type self: OpMode\n        :return: True if we are operating in a mode that shares Internet\n        access.\n        :rtype: bool\n        '
        return self.op_mode in [constants.OP_MODE2, constants.OP_MODE3, constants.OP_MODE6]

    def extensions_enabled(self):
        if False:
            return 10
        '\n        :param self: An OpModeManager object\n        :type self: OpModeManager\n        :return: True if we are loading extensions\n        :rtype: bool\n        '
        return self.op_mode in [constants.OP_MODE1, constants.OP_MODE2, constants.OP_MODE5, constants.OP_MODE6, constants.OP_MODE7, constants.OP_MODE8]

    def freq_hopping_enabled(self):
        if False:
            print('Hello World!')
        '\n        :param self: An OpMode object\n        :type self: OpMode\n        :return: True if we are separating the wireless cards\n        for extensions and launching AP.\n        :rtype: bool\n        ..note: MODE5 and MODE6 only use one card to do deauth and\n        lunch ap so it is not allowed to do frequency hopping.\n        '
        return self.op_mode in [constants.OP_MODE1, constants.OP_MODE2, constants.OP_MODE7]

    def assoc_enabled(self):
        if False:
            return 10
        '\n        :param self: An OpMode object\n        :type self: OpMode\n        :return: True if we are using managed Extensions(that associate to WLANs)\n        :rtype: bool\n        '
        return self.op_mode in [constants.OP_MODE7, constants.OP_MODE8]

def validate_ap_interface(interface):
    if False:
        i = 10
        return i + 15
    '\n    Validate the given interface\n\n    :param interface: Name of an interface\n    :type interface: str\n    :return: the ap interface\n    :rtype: str\n    :raises: argparse.ArgumentTypeError in case of invalid interface\n    '
    if not (pyric.pyw.iswireless(interface) and pyric.pyw.isinterface(interface) and interfaces.does_have_mode(interface, 'AP')):
        raise argparse.ArgumentTypeError('Provided interface ({}) either does not exist or does not support AP mode'.format(interface))
    return interface