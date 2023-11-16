import pdfplumber
path = 'D:\\Photo\\s素材\\da83e97d-29aa-49ea-9783-e25abe402012.pdf'

def pdf_convert_txt(filename):
    if False:
        i = 10
        return i + 15
    with pdfplumber.open(filename) as pdf:
        content = ''
        for i in range(len(pdf.pages)):
            page = pdf.pages[i]
            page_content = '\n'.join(page.extract_text().split('\n')[:-1])
            content = content + page_content
        print(content)
        return content