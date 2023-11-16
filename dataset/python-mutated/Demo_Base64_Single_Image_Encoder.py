import PySimpleGUI as sg
import base64
"\n    Make base64 image from a file\n    This is usually done in order to create a Base64 image for use as an Ucon or a Button image\n    To use, either copy and paste the full path to the file or use the browse button to locate the file.\n    Once chosen, the conversion will happen automatically with the result placed on the clipboard.\n    When complete, a popup window is shown that tells you to paste the image before closing the window.  This is because of a\n    tkinter problem on Linux.  On Windows you can close the Window, but on Linux, you'll need to keep it open until the paste completes\n\n    NOTE - if you're replacing your ICO file for your window with a base64 image, you will first need to convert your icon from\n    an ICO file into a PNG file.  Encode the PNG file and then you'll be able to pass that value in your call to Window:\n\n    window = sg.Window('Window Title', layout, icon=icon)\n\n    Where icon is a variable you created using the contents of the clipboard folowing running this program.\n\n    Input: a single image file\n    Output: clipboard will contain the Base64 Byte String of the source image\n    \n    Copyright 2021 PySimpleGUI\n"

def convert_file_to_base64(filename):
    if False:
        for i in range(10):
            print('nop')
    try:
        contents = open(filename, 'rb').read()
        encoded = base64.b64encode(contents)
        sg.clipboard_set(encoded)
        sg.popup('Copied to your clipboard!', 'Keep window open until you have pasted the base64 bytestring')
    except Exception as error:
        sg.popup_error('Cancelled - An error occurred', error)
if __name__ == '__main__':
    filename = sg.popup_get_file('Source Image will be encoded and results placed on clipboard', title='Base64 Encoder')
    if filename:
        convert_file_to_base64(filename)
    else:
        sg.popup_cancel('Cancelled - No valid file entered')