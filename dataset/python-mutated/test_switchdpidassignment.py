"""Package: mininet
   Regression tests for switch dpid assignment."""
import unittest
import sys
from mininet.net import Mininet
from mininet.node import Host, Controller
from mininet.node import UserSwitch, OVSSwitch, IVSSwitch
from mininet.topo import Topo
from mininet.log import setLogLevel
from mininet.util import quietRun
from mininet.clean import cleanup

class TestSwitchDpidAssignmentOVS(unittest.TestCase):
    """Verify Switch dpid assignment."""
    switchClass = OVSSwitch

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        'Clean up if necessary'
        assert self
        if sys.exc_info() != (None, None, None):
            cleanup()

    def testDefaultDpid(self):
        if False:
            return 10
        'Verify that the default dpid is assigned using a valid provided\n        canonical switchname if no dpid is passed in switch creation.'
        net = Mininet(Topo(), self.switchClass, Host, Controller)
        switch = net.addSwitch('s1')
        self.assertEqual(switch.defaultDpid(), switch.dpid)
        net.stop()

    def dpidFrom(self, num):
        if False:
            i = 10
            return i + 15
        'Compute default dpid from number'
        fmt = '%0' + str(self.switchClass.dpidLen) + 'x'
        return fmt % num

    def testActualDpidAssignment(self):
        if False:
            return 10
        'Verify that Switch dpid is the actual dpid assigned if dpid is\n        passed in switch creation.'
        dpid = self.dpidFrom(43981)
        net = Mininet(Topo(), self.switchClass, Host, Controller)
        switch = net.addSwitch('s1', dpid=dpid)
        self.assertEqual(switch.dpid, dpid)
        net.stop()

    def testDefaultDpidAssignmentFailure(self):
        if False:
            return 10
        'Verify that Default dpid assignment raises an Exception if the\n        name of the switch does not contain a digit. Also verify the\n        exception message.'
        net = Mininet(Topo(), self.switchClass, Host, Controller)
        with self.assertRaises(Exception) as raises_cm:
            net.addSwitch('A')
        self.assertTrue('Unable to derive default datapath ID - please either specify a dpid or use a canonical switch name such as s23.' in str(raises_cm.exception))
        net.stop()

    def testDefaultDpidLen(self):
        if False:
            while True:
                i = 10
        "Verify that Default dpid length is 16 characters consisting of\n        16 - len(hex of first string of contiguous digits passed in switch\n        name) 0's followed by hex of first string of contiguous digits passed\n        in switch name."
        net = Mininet(Topo(), self.switchClass, Host, Controller)
        switch = net.addSwitch('s123')
        self.assertEqual(switch.dpid, self.dpidFrom(123))
        net.stop()

class OVSUser(OVSSwitch):
    """OVS User Switch convenience class"""

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        kwargs.update(datapath='user')
        OVSSwitch.__init__(self, *args, **kwargs)

class testSwitchOVSUser(TestSwitchDpidAssignmentOVS):
    """Test dpid assignment of OVS User Switch."""
    switchClass = OVSUser

@unittest.skipUnless(quietRun('which ivs-ctl'), 'IVS switch is not installed')
class testSwitchIVS(TestSwitchDpidAssignmentOVS):
    """Test dpid assignment of IVS switch."""
    switchClass = IVSSwitch

@unittest.skipUnless(quietRun('which ofprotocol'), 'Reference user switch is not installed')
class testSwitchUserspace(TestSwitchDpidAssignmentOVS):
    """Test dpid assignment of Userspace switch."""
    switchClass = UserSwitch
if __name__ == '__main__':
    setLogLevel('warning')
    unittest.main()
    cleanup()