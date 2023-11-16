from board import LED
from machine import RTCounter, Temp
from ubluepy import Service, Characteristic, UUID, Peripheral, constants

def event_handler(id, handle, data):
    if False:
        for i in range(10):
            print('nop')
    global rtc
    global periph
    global serv_env_sense
    global notif_enabled
    if id == constants.EVT_GAP_CONNECTED:
        LED(1).on()
    elif id == constants.EVT_GAP_DISCONNECTED:
        rtc.stop()
        LED(1).off()
        periph.advertise(device_name='micr_temp', services=[serv_env_sense])
    elif id == constants.EVT_GATTS_WRITE:
        if int(data[0]) == 1:
            notif_enabled = True
            rtc.start()
        else:
            notif_enabled = False
            rtc.stop()

def send_temp(timer_id):
    if False:
        while True:
            i = 10
    global notif_enabled
    global char_temp
    if notif_enabled:
        temp = Temp.read()
        temp = temp * 100
        char_temp.write(bytearray([temp & 255, temp >> 8]))
LED(1).off()
rtc = RTCounter(1, period=50, mode=RTCounter.PERIODIC, callback=send_temp)
notif_enabled = False
uuid_env_sense = UUID('0x181A')
uuid_temp = UUID('0x2A6E')
serv_env_sense = Service(uuid_env_sense)
temp_props = Characteristic.PROP_NOTIFY | Characteristic.PROP_READ
temp_attrs = Characteristic.ATTR_CCCD
char_temp = Characteristic(uuid_temp, props=temp_props, attrs=temp_attrs)
serv_env_sense.addCharacteristic(char_temp)
periph = Peripheral()
periph.addService(serv_env_sense)
periph.setConnectionHandler(event_handler)
periph.advertise(device_name='micr_temp', services=[serv_env_sense])