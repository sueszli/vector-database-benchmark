import csv
import sys

def evaluate_periph(inper, inlist, periph, subtype, altfn, pin):
    if False:
        for i in range(10):
            print('nop')
    if not inper.find('/') == -1:
        inper = inper[:inper.find('/')]
    if inper[:len(periph)] == periph and inper[-len(subtype):] == subtype:
        inlist.append([inper[len(periph):len(periph) + 1], altfn, pin])

def evaluate_tim(inper, inlist, altfn, pin):
    if False:
        while True:
            i = 10
    if not inper.find('/') == -1:
        inper = inper[:inper.find('/')]
    if inper[:3] == 'TIM' and inper[5:7] == 'CH' and (inper[-1:] != 'N'):
        inlist.append([inper[3:4], altfn, inper[-1:], pin])
    elif inper[:3] == 'TIM' and inper[6:8] == 'CH' and (inper[-1:] != 'N'):
        inlist.append([inper[3:5], altfn, inper[-1:], pin])
with open(sys.argv[1]) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    if sys.argv[2] != '-pins-only':
        todo = [['I2C', 'SDA'], ['I2C', 'SCL'], ['SPI', 'SCK'], ['SPI', 'MOSI'], ['SPI', 'MISO'], ['SPI', 'NSS'], ['UART', 'TX'], ['UART', 'RX']]
        outlist = []
        for items in todo:
            empty = []
            outlist.append(empty)
        empty = []
        outlist.append(empty)
        for row in csv_reader:
            altfn = 0
            pin = row[0]
            if len(pin) < 4:
                pin = pin[:2] + '0' + pin[2:]
            for col in row:
                array_index = 0
                for item in todo:
                    evaluate_periph(col, outlist[array_index], item[0], item[1], altfn - 1, pin)
                    if item[0] == 'UART':
                        evaluate_periph(col, outlist[array_index], 'USART', item[1], altfn - 1, pin)
                    array_index += 1
                evaluate_tim(col, outlist[-1], altfn - 1, pin)
                altfn += 1
            line_count += 1
        for i in range(len(todo)):
            ins = todo[i][0].lower() + '_' + todo[i][1].lower() + '_'
            print('const mcu_periph_obj_t mcu_' + ins + 'list[' + str(len(outlist[i])) + '] = {')
            for row in outlist[i]:
                print('    PERIPH(' + row[0] + ', ' + str(row[1]) + ', &pin_' + row[2] + '),')
            print('};')
        print('const mcu_tim_pin_obj_t mcu_tim_pin_list[' + str(len(outlist[-1])) + '] = {')
        for row in outlist[-1]:
            print('    TIM(' + row[0] + ', ' + str(row[1]) + ', ' + str(row[2]) + ', &pin_' + row[3] + '),')
        print('};')
    else:
        outlist = []
        for row in csv_reader:
            altfn = 0
            pin = row[0]
            if len(pin) < 4:
                pin = pin[:2] + '0' + pin[2:]
            outlist.append([pin, str(ord(pin[1:2]) - 65), pin[2:4]])
            line_count += 1
        for line in outlist:
            print('const mcu_pin_obj_t pin_' + line[0] + ' = PIN(' + line[1] + ', ' + line[2] + ', NO_ADC);')
        for line in outlist:
            print('extern const mcu_pin_obj_t pin_' + line[0] + ';')
    print('Processed %d lines.' % line_count)