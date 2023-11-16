import os
import glob
from pypdf import PdfReader

def get_pdf_files(file_path):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get all PDF files from the specified file path.\n\n    Args:\n        file_path (str): The directory path containing the PDF files.\n\n    Returns:\n        list: A list containing the paths of all the PDF files in the directory.\n    '
    if os.path.exists(file_path):
        return glob.glob(os.path.join(file_path, '*.pdf'))
    else:
        return []

def read_multiple_pdf(file_path: str) -> list:
    if False:
        i = 10
        return i + 15
    '\n    Read multiple PDF files from the specified file path and extract the text from each page.\n\n    Args:\n        file_path (str): The directory path containing the PDF files.\n\n    Returns:\n        list: A list containing the extracted text from each page of the PDF files.\n    '
    pdf_files = get_pdf_files(file_path)
    output = []
    for file in pdf_files:
        try:
            with open(file, 'rb') as f:
                pdf_reader = PdfReader(f)
                count = pdf_reader.getNumPages()
                for i in range(count):
                    page = pdf_reader.getPage(i)
                    output.append(page.extractText())
        except Exception as e:
            print(f"Error reading file '{file}': {str(e)}")
    return output

def read_single_pdf(file_path: str) -> str:
    if False:
        return 10
    '\n    Read a single PDF file and extract the text from each page.\n\n    Args:\n        file_path (str): The path of the PDF file.\n\n    Returns:\n        list: A list containing the extracted text from each page of the PDF file.\n    '
    output = []
    try:
        with open(file_path, 'rb') as f:
            pdf_reader = PdfReader(f)
            count = len(pdf_reader.pages)
            for i in range(count):
                page = pdf_reader.pages[i]
                output.append(page.extract_text())
    except Exception as e:
        print(f"Error reading file '{file_path}': {str(e)}")
    return str(' '.join(output))

def get_pdf_files(file_path: str) -> list:
    if False:
        while True:
            i = 10
    '\n    Get a list of PDF files from the specified directory path.\n\n    Args:\n        file_path (str): The directory path containing the PDF files.\n\n    Returns:\n        list: A list of PDF file paths.\n    '
    pdf_files = []
    try:
        pdf_files = glob.glob(os.path.join(file_path, '*.pdf'))
    except Exception as e:
        print(f"Error getting PDF files from '{file_path}': {str(e)}")
    return pdf_files