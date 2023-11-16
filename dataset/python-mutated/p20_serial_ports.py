"""
Topic: 串行端口数据交换
Desc : 
"""
import serial

def serial_posts():
    if False:
        while True:
            i = 10
    ser = serial.Serial('/dev/tty.usbmodem641', baudrate=9600, bytesize=8, parity='N', stopbits=1)
if __name__ == '__main__':
    serial_posts()