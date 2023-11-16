from typing import List
from qrcodegen import QrCode, QrSegment

def main() -> None:
    if False:
        i = 10
        return i + 15
    'The main application program.'
    do_basic_demo()
    do_variety_demo()
    do_segment_demo()
    do_mask_demo()

def do_basic_demo() -> None:
    if False:
        print('Hello World!')
    'Creates a single QR Code, then prints it to the console.'
    text = 'Hello, world!'
    errcorlvl = QrCode.Ecc.LOW
    qr = QrCode.encode_text(text, errcorlvl)
    print_qr(qr)
    print(to_svg_str(qr, 4))

def do_variety_demo() -> None:
    if False:
        return 10
    'Creates a variety of QR Codes that exercise different features of the library, and prints each one to the console.'
    qr = QrCode.encode_text('314159265358979323846264338327950288419716939937510', QrCode.Ecc.MEDIUM)
    print_qr(qr)
    qr = QrCode.encode_text('DOLLAR-AMOUNT:$39.87 PERCENTAGE:100.00% OPERATIONS:+-*/', QrCode.Ecc.HIGH)
    print_qr(qr)
    qr = QrCode.encode_text('こんにちwa、世界！ αβγδ', QrCode.Ecc.QUARTILE)
    print_qr(qr)
    qr = QrCode.encode_text("Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversations?' So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.", QrCode.Ecc.HIGH)
    print_qr(qr)

def do_segment_demo() -> None:
    if False:
        i = 10
        return i + 15
    'Creates QR Codes with manually specified segments for better compactness.'
    silver0 = 'THE SQUARE ROOT OF 2 IS 1.'
    silver1 = '41421356237309504880168872420969807856967187537694807317667973799'
    qr = QrCode.encode_text(silver0 + silver1, QrCode.Ecc.LOW)
    print_qr(qr)
    segs = [QrSegment.make_alphanumeric(silver0), QrSegment.make_numeric(silver1)]
    qr = QrCode.encode_segments(segs, QrCode.Ecc.LOW)
    print_qr(qr)
    golden0 = 'Golden ratio φ = 1.'
    golden1 = '6180339887498948482045868343656381177203091798057628621354486227052604628189024497072072041893911374'
    golden2 = '......'
    qr = QrCode.encode_text(golden0 + golden1 + golden2, QrCode.Ecc.LOW)
    print_qr(qr)
    segs = [QrSegment.make_bytes(golden0.encode('UTF-8')), QrSegment.make_numeric(golden1), QrSegment.make_alphanumeric(golden2)]
    qr = QrCode.encode_segments(segs, QrCode.Ecc.LOW)
    print_qr(qr)
    madoka = '「魔法少女まどか☆マギカ」って、\u3000ИАИ\u3000ｄｅｓｕ\u3000κα？'
    qr = QrCode.encode_text(madoka, QrCode.Ecc.LOW)
    print_qr(qr)
    kanjicharbits = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    segs = [QrSegment(QrSegment.Mode.KANJI, len(kanjicharbits) // 13, kanjicharbits)]
    qr = QrCode.encode_segments(segs, QrCode.Ecc.LOW)
    print_qr(qr)

def do_mask_demo() -> None:
    if False:
        print('Hello World!')
    'Creates QR Codes with the same size and contents but different mask patterns.'
    segs = QrSegment.make_segments('https://www.nayuki.io/')
    print_qr(QrCode.encode_segments(segs, QrCode.Ecc.HIGH, mask=-1))
    print_qr(QrCode.encode_segments(segs, QrCode.Ecc.HIGH, mask=3))
    segs = QrSegment.make_segments('維基百科（Wikipedia，聆聽i/ˌwɪkᵻˈpiːdi.ə/）是一個自由內容、公開編輯且多語言的網路百科全書協作計畫')
    print_qr(QrCode.encode_segments(segs, QrCode.Ecc.MEDIUM, mask=0))
    print_qr(QrCode.encode_segments(segs, QrCode.Ecc.MEDIUM, mask=1))
    print_qr(QrCode.encode_segments(segs, QrCode.Ecc.MEDIUM, mask=5))
    print_qr(QrCode.encode_segments(segs, QrCode.Ecc.MEDIUM, mask=7))

def to_svg_str(qr: QrCode, border: int) -> str:
    if False:
        print('Hello World!')
    'Returns a string of SVG code for an image depicting the given QR Code, with the given number\n\tof border modules. The string always uses Unix newlines (\n), regardless of the platform.'
    if border < 0:
        raise ValueError('Border must be non-negative')
    parts: List[str] = []
    for y in range(qr.get_size()):
        for x in range(qr.get_size()):
            if qr.get_module(x, y):
                parts.append(f'M{x + border},{y + border}h1v1h-1z')
    return f'''<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n<svg xmlns="http://www.w3.org/2000/svg" version="1.1" viewBox="0 0 {qr.get_size() + border * 2} {qr.get_size() + border * 2}" stroke="none">\n\t<rect width="100%" height="100%" fill="#FFFFFF"/>\n\t<path d="{' '.join(parts)}" fill="#000000"/>\n</svg>\n'''

def print_qr(qrcode: QrCode) -> None:
    if False:
        while True:
            i = 10
    'Prints the given QrCode object to the console.'
    border = 4
    for y in range(-border, qrcode.get_size() + border):
        for x in range(-border, qrcode.get_size() + border):
            print('█ '[1 if qrcode.get_module(x, y) else 0] * 2, end='')
        print()
    print()
if __name__ == '__main__':
    main()