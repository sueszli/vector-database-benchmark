import time
from jeepney import DBusAddress, DBusErrorResponse, MessageType, Properties, new_method_call
from jeepney.io.blocking import open_dbus_connection
from calibre.constants import DEBUG

class HAL:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.bus = open_dbus_connection('SYSTEM')

    def send(self, msg):
        if False:
            for i in range(10):
                print('nop')
        reply = self.bus.send_and_get_reply(msg)
        if reply.header.message_type is MessageType.error:
            raise DBusErrorResponse(reply)
        return reply.body[0]

    def call(self, addr, method, sig='', *args):
        if False:
            for i in range(10):
                print('nop')
        if sig:
            return self.send(new_method_call(addr, method, sig, args))
        return self.send(new_method_call(addr, method))

    def prop(self, addr, name):
        if False:
            for i in range(10):
                print('nop')
        return self.send(Properties(addr).get(name))

    def addr(self, path, interface):
        if False:
            for i in range(10):
                print('nop')
        return DBusAddress(path, bus_name='org.freedesktop.Hal', interface=f'org.freedesktop.Hal.{interface}')

    def get_volume(self, vpath):
        if False:
            print('Hello World!')
        vdevif = self.addr(vpath, 'Device')
        if not self.prop(vdevif, 'block.is_volume') or self.prop(vdevif, 'volume.fsusage') != 'filesystem':
            return
        volif = self.addr(vpath, 'Volume')
        pdevif = self.addr(self.prop(volif, 'info.parent'), 'Device')
        return {'node': self.prop(pdevif, 'block.device'), 'dev': vdevif, 'vol': volif, 'label': self.prop(vdevif, 'volume.label')}

    def get_volumes(self, d):
        if False:
            print('Hello World!')
        vols = []
        manager = self.addr('/org/freedesktop/Hal/Manager', 'Manager')
        paths = self.call(manager, 'FindDeviceStringMatch', 'ss', 'usb.serial', d.serial)
        for path in paths:
            objif = self.addr(path, 'Device')
            try:
                if d.idVendor == self.prop(objif, 'usb.vendor_id') and d.idProduct == self.prop(objif, 'usb.product_id') and (d.manufacturer == self.prop(objif, 'usb.vendor')) and (d.product == self.prop(objif, 'usb.product')) and (d.serial == self.prop(objif, 'usb.serial')):
                    midpath = self.call(manager, 'FindDeviceStringMatch', 'ss', 'info.parent', path)
                    dpaths = self.call(manager, 'FindDeviceStringMatch', 'ss', 'storage.originating_device', path) + self.call(manager, 'FindDeviceStringMatch', 'ss', 'storage.originating_device', midpath[0])
                    for dpath in dpaths:
                        try:
                            vpaths = self.call(manager, 'FindDeviceStringMatch', 'block.storage_device', dpath)
                            for vpath in vpaths:
                                try:
                                    vol = self.get_volume(vpath)
                                    if vol is not None:
                                        vols.append(vol)
                                except DBusErrorResponse as e:
                                    print(e)
                                    continue
                        except DBusErrorResponse as e:
                            print(e)
                            continue
            except DBusErrorResponse:
                continue
        vols.sort(key=lambda x: x['node'])
        return vols

    def get_mount_point(self, vol):
        if False:
            i = 10
            return i + 15
        if not self.prop(vol['dev'], 'volume.is_mounted'):
            fstype = self.prop(vol['dev'], 'volume.fstype')
            self.call(vol['vol'], 'Mount', 'ssas', 'Calibre-' + vol['label'], fstype, [])
            loops = 0
            while not self.prop(vol['dev'], 'volume.is_mounted'):
                time.sleep(1)
                loops += 1
                if loops > 100:
                    raise Exception('ERROR: Timeout waiting for mount to complete')
        return self.prop(vol['dev'], 'volume.mount_point')

    def mount_volumes(self, volumes):
        if False:
            for i in range(10):
                print('nop')
        mtd = 0
        ans = {'_main_prefix': None, '_main_vol': None, '_card_a_prefix': None, '_card_a_vol': None, '_card_b_prefix': None, '_card_b_vol': None}
        for vol in volumes:
            try:
                mp = self.get_mount_point(vol)
            except Exception as e:
                print("Failed to mount: {vol['label']}", e)
                continue
            mp += '/'
            if DEBUG:
                print('FBSD:\t  mounted', vol['label'], 'on', mp)
            if mtd == 0:
                (ans['_main_prefix'], ans['_main_vol']) = (mp, vol['vol'])
                if DEBUG:
                    print('FBSD:\tmain = ', mp)
            elif mtd == 1:
                (ans['_card_a_prefix'], ans['_card_a_vol']) = (mp, vol['vol'])
                if DEBUG:
                    print('FBSD:\tcard a = ', mp)
            elif mtd == 2:
                (ans['_card_b_prefix'], ans['_card_b_vol']) = (mp, vol['vol'])
                if DEBUG:
                    print('FBSD:\tcard b = ', mp)
                break
            mtd += 1
        return (mtd > 0, ans)

    def unmount(self, vol):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.call(vol, 'Unmount', 'as', [])
        except DBusErrorResponse as e:
            print('Unable to eject ', e)

def get_hal():
    if False:
        return 10
    if not hasattr(get_hal, 'ans'):
        get_hal.ans = HAL()
    return get_hal.ans