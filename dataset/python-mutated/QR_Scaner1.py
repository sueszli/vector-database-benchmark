"""
QR_Scaner1.py:
"""
from pyzbar.pyzbar import decode
import cv2

def main():
    if False:
        print('Hello World!')
    fp = 'macbookPro.jpg'
    image = cv2.imread(fp)
    barcodes = decode(image)
    decoded = barcodes[0]
    print(decoded)
    url: bytes = decoded.data
    url = url.decode()
    print(url)
    rect = decoded.rect
    print(rect)
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        barcodeData = barcode.data.decode('utf-8')
        barcodeType = barcode.type
        text = '{} ({})'.format(barcodeData, barcodeType)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        print('[INFO] Found {} barcode: {}'.format(barcodeType, barcodeData))
    cv2.imshow('Image', image)
    cv2.waitKey(0)
if __name__ == '__main__':
    main()