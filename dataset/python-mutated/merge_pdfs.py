from PyPDF2 import PdfFileMerger

def by_appending():
    if False:
        while True:
            i = 10
    merger = PdfFileMerger()
    f1 = open('samplePdf1.pdf', 'rb')
    merger.append(f1)
    merger.append('samplePdf2.pdf')
    merger.write('mergedPdf.pdf')

def by_inserting():
    if False:
        for i in range(10):
            print('nop')
    merger = PdfFileMerger()
    merger.append('samplePdf1.pdf')
    merger.merge(0, 'samplePdf2.pdf')
    merger.write('mergedPdf1.pdf')
if __name__ == '__main__':
    by_appending()
    by_inserting()