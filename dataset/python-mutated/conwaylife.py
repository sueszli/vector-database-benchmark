import pyb
lcd = pyb.LCD('x')
lcd.light(1)

def conway_step():
    if False:
        return 10
    for x in range(128):
        for y in range(32):
            num_neighbours = lcd.get(x - 1, y - 1) + lcd.get(x, y - 1) + lcd.get(x + 1, y - 1) + lcd.get(x - 1, y) + lcd.get(x + 1, y) + lcd.get(x + 1, y + 1) + lcd.get(x, y + 1) + lcd.get(x - 1, y + 1)
            self = lcd.get(x, y)
            if self and (not 2 <= num_neighbours <= 3):
                lcd.pixel(x, y, 0)
            elif not self and num_neighbours == 3:
                lcd.pixel(x, y, 1)

def conway_rand():
    if False:
        while True:
            i = 10
    lcd.fill(0)
    for x in range(128):
        for y in range(32):
            lcd.pixel(x, y, pyb.rng() & 1)

def conway_go(num_frames):
    if False:
        print('Hello World!')
    for i in range(num_frames):
        conway_step()
        lcd.show()
        pyb.delay(50)
conway_rand()
conway_go(100)