from __future__ import division
from __future__ import print_function
import zlib
import base64
import pytest
import unittest
from tests import RemoteTestCase
from impacket.dcerpc.v5.dcom import wmi
from impacket.dcerpc.v5.dtypes import NULL
from impacket.dcerpc.v5.dcomrt import DCOMConnection

@pytest.mark.remote
class WMITests(RemoteTestCase, unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        super(WMITests, self).setUp()
        self.set_transport_config()

    @pytest.mark.xfail
    def test_activation(self):
        if False:
            print('Hello World!')
        dcom = DCOMConnection(self.machine, self.username, self.password, self.domain, self.lmhash, self.nthash)
        dcom.CoCreateInstanceEx(wmi.CLSID_WbemLevel1Login, wmi.IID_IWbemLoginClientID)
        dcom.disconnect()

    def test_IWbemLevel1Login_EstablishPosition(self):
        if False:
            while True:
                i = 10
        dcom = DCOMConnection(self.machine, self.username, self.password, self.domain, self.lmhash, self.nthash)
        iInterface = dcom.CoCreateInstanceEx(wmi.CLSID_WbemLevel1Login, wmi.IID_IWbemLevel1Login)
        iWbemLevel1Login = wmi.IWbemLevel1Login(iInterface)
        resp = iWbemLevel1Login.EstablishPosition()
        print(resp)
        dcom.disconnect()

    def test_IWbemLevel1Login_RequestChallenge(self):
        if False:
            print('Hello World!')
        dcom = DCOMConnection(self.machine, self.username, self.password, self.domain, self.lmhash, self.nthash)
        iInterface = dcom.CoCreateInstanceEx(wmi.CLSID_WbemLevel1Login, wmi.IID_IWbemLevel1Login)
        iWbemLevel1Login = wmi.IWbemLevel1Login(iInterface)
        try:
            resp = iWbemLevel1Login.RequestChallenge()
            print(resp)
        except Exception as e:
            if str(e).find('WBEM_E_NOT_SUPPORTED') < 0:
                dcom.disconnect()
                raise
        dcom.disconnect()

    def test_IWbemLevel1Login_WBEMLogin(self):
        if False:
            return 10
        dcom = DCOMConnection(self.machine, self.username, self.password, self.domain, self.lmhash, self.nthash)
        iInterface = dcom.CoCreateInstanceEx(wmi.CLSID_WbemLevel1Login, wmi.IID_IWbemLevel1Login)
        iWbemLevel1Login = wmi.IWbemLevel1Login(iInterface)
        try:
            resp = iWbemLevel1Login.WBEMLogin()
            print(resp)
        except Exception as e:
            if str(e).find('E_NOTIMPL') < 0:
                dcom.disconnect()
                raise
        dcom.disconnect()

    def test_IWbemLevel1Login_NTLMLogin(self):
        if False:
            print('Hello World!')
        dcom = DCOMConnection(self.machine, self.username, self.password, self.domain, self.lmhash, self.nthash)
        iInterface = dcom.CoCreateInstanceEx(wmi.CLSID_WbemLevel1Login, wmi.IID_IWbemLevel1Login)
        iWbemLevel1Login = wmi.IWbemLevel1Login(iInterface)
        resp = iWbemLevel1Login.NTLMLogin('\\\\%s\\root\\cimv2' % self.machine, NULL, NULL)
        print(resp)
        dcom.disconnect()

    @pytest.mark.xfail
    def test_IWbemServices_OpenNamespace(self):
        if False:
            print('Hello World!')
        dcom = DCOMConnection(self.machine, self.username, self.password, self.domain, self.lmhash, self.nthash)
        iInterface = dcom.CoCreateInstanceEx(wmi.CLSID_WbemLevel1Login, wmi.IID_IWbemLevel1Login)
        iWbemLevel1Login = wmi.IWbemLevel1Login(iInterface)
        iWbemServices = iWbemLevel1Login.NTLMLogin('//./ROOT', NULL, NULL)
        try:
            resp = iWbemServices.OpenNamespace('__Namespace')
            print(resp)
        except Exception:
            dcom.disconnect()
            raise
        dcom.disconnect()

    def test_IWbemServices_GetObject(self):
        if False:
            print('Hello World!')
        dcom = DCOMConnection(self.machine, self.username, self.password, self.domain, self.lmhash, self.nthash)
        iInterface = dcom.CoCreateInstanceEx(wmi.CLSID_WbemLevel1Login, wmi.IID_IWbemLevel1Login)
        iWbemLevel1Login = wmi.IWbemLevel1Login(iInterface)
        iWbemServices = iWbemLevel1Login.NTLMLogin('\\\\%s\\root\\cimv2' % self.machine, NULL, NULL)
        iWbemLevel1Login.RemRelease()
        (classObject, _) = iWbemServices.GetObject('Win32_Process')
        dcom.disconnect()

    def test_IWbemServices_ExecQuery(self):
        if False:
            for i in range(10):
                print('nop')
        dcom = DCOMConnection(self.machine, self.username, self.password, self.domain, self.lmhash, self.nthash)
        iInterface = dcom.CoCreateInstanceEx(wmi.CLSID_WbemLevel1Login, wmi.IID_IWbemLevel1Login)
        iWbemLevel1Login = wmi.IWbemLevel1Login(iInterface)
        iWbemServices = iWbemLevel1Login.NTLMLogin('\\\\%s\\root\\cimv2' % self.machine, NULL, NULL)
        classes = ['Win32_Service']
        for classn in classes:
            print('Reading %s ' % classn)
            try:
                iEnumWbemClassObject = iWbemServices.ExecQuery('SELECT * from %s' % classn)
                done = False
                while done is False:
                    try:
                        iEnumWbemClassObject.Next(4294967295, 1)
                    except Exception as e:
                        if str(e).find('S_FALSE') < 0:
                            print(e)
                        else:
                            done = True
                            pass
            except Exception as e:
                if str(e).find('S_FALSE') < 0:
                    print(e)
        dcom.disconnect()

    def test_IWbemServices_ExecMethod(self):
        if False:
            while True:
                i = 10
        dcom = DCOMConnection(self.machine, self.username, self.password, self.domain, self.lmhash, self.nthash)
        iInterface = dcom.CoCreateInstanceEx(wmi.CLSID_WbemLevel1Login, wmi.IID_IWbemLevel1Login)
        iWbemLevel1Login = wmi.IWbemLevel1Login(iInterface)
        iWbemServices = iWbemLevel1Login.NTLMLogin('\\\\%s\\root\\cimv2' % self.machine, NULL, NULL)
        (classObject, _) = iWbemServices.GetObject('Win32_Process')
        obj = classObject.Create('notepad.exe', 'c:\\', None)
        handle = obj.getProperties()['ProcessId']['value']
        iEnumWbemClassObject = iWbemServices.ExecQuery('SELECT * from Win32_Process where handle = %s' % handle)
        oooo = iEnumWbemClassObject.Next(4294967295, 1)[0]
        oooo.Terminate(1)
        dcom.disconnect()

class WMIOfflineTests(unittest.TestCase):

    @staticmethod
    def createIWbemClassObject(b64_compressed_obj_ref):
        if False:
            print('Hello World!')
        obj_ref = zlib.decompress(base64.b64decode(b64_compressed_obj_ref))
        interface = wmi.INTERFACE(objRef=obj_ref, target='')
        return wmi.IWbemClassObject(interface, interface)

    def assertIWbemClassObjectAttr(self, _object, attribute_name, expected_value):
        if False:
            i = 10
            return i + 15
        actual_value = getattr(_object, attribute_name)
        self.assertEqual(expected_value, actual_value, '{}.{} is {!r}, but was expecting {!r}'.format(_object.getClassName(), attribute_name, actual_value, expected_value))

    def test_win32_current_time_class_parsing(self):
        if False:
            print('Hello World!')
        "\n        https://docs.microsoft.com/en-us/previous-versions/windows/desktop/wmitimepprov/win32-currenttime\n        Parse a Win32_CurrentTime instance object, response for the 'Select * from Win32_UTCTime' WMI query\n\n        The data was obtained by running the following command while patching impacket.dcerpc.v5.dcomrt.INTERFACE:\n        echo 'Select * from Win32_UTCTime' | wmiquery.py username:password@x.x.x.x -file -\n\n        The following lines were added in the impacket.dcerpc.v5.dcomrt.INTERFACE class constructor:\n        https://github.com/fortra/impacket/blob/impacket_0_9_22/impacket/dcerpc/v5/dcomrt.py#L1111-L1112\n        if objRef and b'Win32_CurrentTime' in objRef:\n            import base64, textwrap, zlib\n            print('\n'.join(textwrap.wrap(base64.b64encode(zlib.compress(objRef)), 96)))\n\n        Target's time had previously been set to 00:00 UTC\n        "
        current_time_obj = self.createIWbemClassObject('\n        eJzNks8vA0EUx7/bVutXQotEQkPSJg4Sh1YcnCTVhFC/WopIpKmhGzWbbHeFOCAcSBzEwR/g4ODmJP4CN5x6EiccHcWNN7Ok\n        I/bg6CWfnTcvn7d5szup5HjWB2D3PPSwXboLHqRwgZGeaOj9ONkfvg8edjh7UlD2AhszvaFbWv1IJ2eSY7N9vYBpGNZCXl9b\n        j6HghRPdRJtIsjqPxxYTtmkybmX0NYYmqnYRAWBHq6Pk48NPKaopbSAiRNyp19KzR2yJeYIRR8QJcU3cEK/EGxHWgCgxQmSI\n        LWKPuCCuiEfiSRNv/XOcemgs5wDTmYQc3tkmikZ+dcI01vUlZgJpna8UmWVwYDC3iaYBwCOPIyJI0Dl2IqIwJSq2zq14TLrj\n        y1nGVmWHF/VuHftqx5Bhm1L2o9VNvlTllF4s6iWWN/hSSTbVIOrW9PKzidsWk3oA7W56i6bqBrcK0tbgc7MTqj1p50yLOSeo\n        QrObX1L9tBxe6tXodNPPVF18ymFeGcmHRreestozx3LOPJX4IXs9ivx/YrSS1j8HxH2AvHAehe+IfK3i/2gN+H2lgU8VG67O\n        ')
        self.assertIWbemClassObjectAttr(current_time_obj, 'Year', 2021)
        self.assertIWbemClassObjectAttr(current_time_obj, 'Month', 6)
        self.assertIWbemClassObjectAttr(current_time_obj, 'Day', 8)
        self.assertIWbemClassObjectAttr(current_time_obj, 'DayOfWeek', 2)
        self.assertIWbemClassObjectAttr(current_time_obj, 'Hour', 0)
        self.assertIWbemClassObjectAttr(current_time_obj, 'Minute', 0)
        self.assertIWbemClassObjectAttr(current_time_obj, 'Second', 35)
        self.assertIWbemClassObjectAttr(current_time_obj, 'Milliseconds', None)

    def test_wmi_persist_classes_parsing(self):
        if False:
            i = 10
            return i + 15
        "\n        Parse several objects created thorough SpawnInstance in wmipersist.py\n\n        The data was obtained by running the following command while patching IWbemClassObject.SpawnInstance():\n        wmipersist.py username:password@x.x.x.x -debug install -name ASEC -timer 1000 -vbs toexec.vbs\n\n        The following lines were added in the impacket.dcerpc.v5.dcom.wmi.IWbemClassObject.SpawnInstance():\n        https://github.com/fortra/impacket/blob/impacket_0_9_22/impacket/dcerpc/v5/dcom/wmi.py#L2557\n        import base64, textwrap, zlib\n        print('\n'.join(textwrap.wrap(base64.b64encode(zlib.compress(objRefCustomIn.getData())), 96)))\n        "
        default_creator_sid = '[1, 1, 0, 0, 0, 0, 0, 5, 18, 0, 0, 0]'
        false = 'False'
        asec_obj = self.createIWbemClassObject('\n        eJy1k89r1EAUx7/Z7db6AzS7FpRaq1bpSQuuoHhSdrNLKetis1gKwpqmwzqQTCSZKVsvxpvevIlexIsH8eS/4UHBqxcPIp68\n        6qm+mezWrARvHfgwM2/yvu87L0nH6a5PAXj8pvrlUfLRftLBW6xeOl/99cy5Pv/JfrqQ7ekR/CgDwztXql9pnkaz1W623Lbb\n        bDvAWrfbu5uozcSP+QPJI4FBGdk4Rzh60e8720zIRiQSFbIYdhZcEVvc93TOGgs8ybYwmx24O4lkYSPwkgRHKaKfn9NHWvkg\n        mZ6heZFYJu4RIfGKeEd8Jr4RSxZwkegTEfGaeE/8Jiolis3ujsZV2u3+M3S1lyWkuOlLvs1cc7vJWyCIfC9gQCNmnoxid6UJ\n        m9yUjD3SxCniEJAu6sBG5j+tH4YuB8WFvAas8iDo8ZBFSqJGJwdMVjmXXdbZzwm6eIpcdv0y0PH8+1ywW17IMHNj1HhT3M4X\n        /6CjiYy5GOicIQ9VeFsxxVz+kKFGiZaWN4mn84k2deskUdebM7TYq6wElwmwuSMZTZkB4/xIkf91S5dAWhv5H3vJGtviARNj\n        hWmcKFJ4YU12YFKBlo4YUCeMxBSOjSSW8hLfSeKnvkMmUaQmItkXKgjGuj02lEayguNFrpZLha50DyzLvItKFfsyNv4u53Lh\n        eWKBOIvsF4FpB7m5gP9/zXvjDwGExJk=\n        ')
        self.assertIWbemClassObjectAttr(asec_obj, 'CreatorSID', default_creator_sid)
        self.assertIWbemClassObjectAttr(asec_obj, 'KillTimeout', 0)
        self.assertIWbemClassObjectAttr(asec_obj, 'MachineName', '')
        self.assertIWbemClassObjectAttr(asec_obj, 'MaximumQueueSize', 0)
        self.assertIWbemClassObjectAttr(asec_obj, 'Name', '')
        self.assertIWbemClassObjectAttr(asec_obj, 'ScriptingEngine', '')
        self.assertIWbemClassObjectAttr(asec_obj, 'ScriptFilename', '')
        self.assertIWbemClassObjectAttr(asec_obj, 'ScriptText', '')
        iti_obj = self.createIWbemClassObject('\n        eJy9UbFOAkEQfYAxRBINh1aiFtrYWIiVNkY5LsQQDEe0MZLjWM3isUdu91ATEzE22ukX2Fn4EbaW8gF+iLHBWQ6vsbBzkrc3\n        Ozv79r25ilk9nABw82x8XMv37F0FL9hbWzE+H8zNhUH2finaUwuCJHBxsGG06TuJYskqlmzLLlomUKtW60cybEo34F3FfYHp\n        JKLIEI510mjUeYcFZSFVELqjptmobvaYUBYTLHCUH8CIqmXR4q6j+2rMcxRrYS46sC+lYp1dz5ESM1TR+lOEPKFAeCQ8Ed4I\n        A8L8kALQq0qgP6JWLOg53i9B+DnZYeqcMTFSJpEjliR5jt5aJUwB/VSakrYeDCGnfQ6HVzrV9VutNeRCFdYB4auGCD2PKoIT\n        Hzrc87hkri9atLPPeLd8sk9+yGRmG0jEnrLjt5Y156vmbPq+xxySGolvIb09nnQ8hPjCFzQX+oVIHMgoF6f4/9iKs3ycaYuJ\n        RfzxQ/AN/wuDfg==\n        ')
        self.assertIWbemClassObjectAttr(iti_obj, 'IntervalBetweenEvents', 0)
        self.assertIWbemClassObjectAttr(iti_obj, 'SkipIfPassed', false)
        self.assertIWbemClassObjectAttr(iti_obj, 'TimerId', '')
        ef_obj = self.createIWbemClassObject('\n        eJydkr9Lw0AUx1/6C1GhJrWIP4oOjlIcHMRNmrSUtpY2oghCONOjBGpachdpJ3XTrYObzg4ujk7+DfoHuDs6KW71XRJtaDv5\n        4JN7d/e+975HrqJVD2MAcHmvvJ2zF/mqAg9Qyq4r331tJ/MqX6/6cyyBuwhA92BLucExAWq+oOb1gq4WNIB6tbp/zNwTZjpW\n        h1ttG7IR8GMJ2RSJYRTthmUSsV2nLcJpA9L+ht5jnJ7mWoQxSOKKcJRARL6MlJEj5ALpI0/IM/KBfCEZCWADqX0OMHTMBiNx\n        K6HYMLQzavO81eLUAcg5lPC2oxdVkNcAJK+tsL2CTGO36BQmeDuI4iQ1A+JYcC2bb+OyOGnXNClaFmVxmA/UcljdFRdk3LHs\n        ZqDZI6eUdYhJPVkM5ibJHsMyofCK/Rhz+A7C/NDhr67mUqfnCaMwO6lLWgp18arLxG66pOm3iwTfEVUppBLHSpJnK67A/0Mb\n        pslgFD1TwiSyAP6bkBZh/Df+xQ8W2n+V\n        ')
        self.assertIWbemClassObjectAttr(ef_obj, 'CreatorSID', None)
        self.assertIWbemClassObjectAttr(ef_obj, 'EventAccess', '')
        self.assertIWbemClassObjectAttr(ef_obj, 'EventNamespace', '')
        self.assertIWbemClassObjectAttr(ef_obj, 'Name', '')
        self.assertIWbemClassObjectAttr(ef_obj, 'Query', '')
        self.assertIWbemClassObjectAttr(ef_obj, 'QueryLanguage', '')
        ftcb_obj = self.createIWbemClassObject('\n        eJydks9rE0EUx79pqi22VLOlINhSDx68KOIPKIUWNJuEUEO0G/RSWLabaR2YzrQzs2lXEONNb0L/BwUPHnr14sGz+gd48q5H\n        9Rbf7MY00OjBB5/Z2Z33vvPe29eoNB+OA3j2yvvy1HwqPW/gDdauXvJ+vawsL3wuvVjM38kFrSJw8OCmd5eep+FXa341qAV+\n        rQKsN5utDZNsmljzXcuVxNcx5HaeuOY2YViXbR5H7nidiciyNubygyA1lu2URWQMztIXxwVidoqWXm+C1svEdWKPeEK8JT4Q\n        hQIwQ9wiVojHxCHxjvhIzL3vDVme0g/6vjqGLt1c5cIy3VJlJU2yw/QdTjnKbeC2MSrmWbLAn1NsuRvhGubKmyfOAN3iJG02\n        siN0+zlDs63lMKx0mLSDcJQ1i6zSQd1H6aLrojdK6jVRHJJKuLRLgM8E7zAdpDJ+pJVUiREppjLf6VEy34mfx13EplKCRVSO\n        VDaUiRADyfS+CjCbaUz2lUrDSleosCyLG/QL8pZlrcjtxMXbhb+2oh+MRkRqRMDiRHObUossO7BZOeOYGZXEkUtiUEMg1L6v\n        9uU9rTq8zbTJQk/h3KjQb8Oh0/2pce4e/t/qx1s3qwsnHNyYFBbx7zkj+w1KorBh\n        ')
        self.assertIWbemClassObjectAttr(ftcb_obj, 'Consumer', '')
        self.assertIWbemClassObjectAttr(ftcb_obj, 'CreatorSID', None)
        self.assertIWbemClassObjectAttr(ftcb_obj, 'DeliverSynchronously', false)
        self.assertIWbemClassObjectAttr(ftcb_obj, 'DeliveryQoS', 0)
        self.assertIWbemClassObjectAttr(ftcb_obj, 'Filter', '')
        self.assertIWbemClassObjectAttr(ftcb_obj, 'MaintainSecurityContext', false)
        self.assertIWbemClassObjectAttr(ftcb_obj, 'SlowDownProviders', false)
if __name__ == '__main__':
    unittest.main(verbosity=1)