import pythoncom
import win32clipboard
from win32com.adsi import adsi
from win32com.adsi.adsicon import *
cf_objectpicker = win32clipboard.RegisterClipboardFormat(CFSTR_DSOP_DS_SELECTION_LIST)

def main():
    if False:
        for i in range(10):
            print('nop')
    hwnd = 0
    picker = pythoncom.CoCreateInstance(adsi.CLSID_DsObjectPicker, None, pythoncom.CLSCTX_INPROC_SERVER, adsi.IID_IDsObjectPicker)
    siis = adsi.DSOP_SCOPE_INIT_INFOs(1)
    sii = siis[0]
    sii.type = DSOP_SCOPE_TYPE_UPLEVEL_JOINED_DOMAIN | DSOP_SCOPE_TYPE_DOWNLEVEL_JOINED_DOMAIN
    sii.filterFlags.uplevel.bothModes = DSOP_FILTER_COMPUTERS
    sii.filterFlags.downlevel = DSOP_DOWNLEVEL_FILTER_COMPUTERS
    picker.Initialize(None, siis, DSOP_FLAG_MULTISELECT, ('objectGUID', 'displayName'))
    do = picker.InvokeDialog(hwnd)
    format_etc = (cf_objectpicker, None, pythoncom.DVASPECT_CONTENT, -1, pythoncom.TYMED_HGLOBAL)
    medium = do.GetData(format_etc)
    data = adsi.StringAsDS_SELECTION_LIST(medium.data)
    for item in data:
        (name, klass, adspath, upn, attrs, flags) = item
        print('Item', name)
        print(' Class:', klass)
        print(' AdsPath:', adspath)
        print(' UPN:', upn)
        print(' Attrs:', attrs)
        print(' Flags:', flags)
if __name__ == '__main__':
    main()