/*
 * PROJECT:     shell32
 * LICENSE:     LGPL-2.1-or-later (https://spdx.org/licenses/LGPL-2.1-or-later)
 * PURPOSE:     SHGetAttributesFromDataObject implementation
 * COPYRIGHT:   Copyright 2021 Mark Jansen <mark.jansen@reactos.org>
 */


#include "precomp.h"

WINE_DEFAULT_DEBUG_CHANNEL(shell);


static CLIPFORMAT g_DataObjectAttributes = 0;
static const DWORD dwDefaultAttributeMask = SFGAO_CANCOPY | SFGAO_CANMOVE | SFGAO_STORAGE | SFGAO_CANRENAME |
                                            SFGAO_CANDELETE | SFGAO_READONLY | SFGAO_STREAM | SFGAO_FOLDER;

struct DataObjectAttributes
{
    DWORD dwMask;
    DWORD dwAttributes;
    UINT cItems;
};

static_assert(sizeof(DataObjectAttributes) == 0xc, "Unexpected struct size!");


static
HRESULT _BindToObject(PCUIDLIST_ABSOLUTE pidl, CComPtr<IShellFolder>& spFolder)
{
    CComPtr<IShellFolder> spDesktop;
    HRESULT hr = SHGetDesktopFolder(&spDesktop);
    if (FAILED(hr))
        return hr;

    return spDesktop->BindToObject(pidl, NULL, IID_PPV_ARG(IShellFolder, &spFolder));
}

EXTERN_C
HRESULT WINAPI SHGetAttributesFromDataObject(IDataObject* pDataObject, DWORD dwAttributeMask, DWORD* pdwAttributes, UINT* pcItems)
{
    DWORD dwAttributes = 0;
    DWORD cItems = 0;
    HRESULT hr = S_OK;

    TRACE("(%p, 0x%x, %p, %p)\n", pDataObject, dwAttributeMask, pdwAttributes, pcItems);

    if (!g_DataObjectAttributes)
        g_DataObjectAttributes = (CLIPFORMAT)RegisterClipboardFormatW(L"DataObjectAttributes");

    if (pDataObject)
    {
        DataObjectAttributes data = {};
        if (FAILED(DataObject_GetData(pDataObject, g_DataObjectAttributes, &data, sizeof(data))))
        {
            TRACE("No attributes yet, creating new\n");
            memset(&data, 0, sizeof(data));
        }

        DWORD dwQueryAttributes = dwAttributeMask | dwDefaultAttributeMask;

        if ((data.dwMask & dwQueryAttributes) != dwQueryAttributes)
        {
            CDataObjectHIDA hida(pDataObject);
            CComPtr<IShellFolder> spFolder;

            if (!FAILED_UNEXPECTEDLY(hr = hida.hr()) &&
                !FAILED_UNEXPECTEDLY(hr = _BindToObject(HIDA_GetPIDLFolder(hida), spFolder)))
            {
                CSimpleArray<PCUIDLIST_RELATIVE> apidl;
                for (UINT n = 0; n < hida->cidl; ++n)
                {
                    apidl.Add(HIDA_GetPIDLItem(hida, n));
                }

                SFGAOF rgfInOut = dwQueryAttributes;
                hr = spFolder->GetAttributesOf(apidl.GetSize(), apidl.GetData(), &rgfInOut);
                if (!FAILED_UNEXPECTEDLY(hr))
                {
                    data.dwMask = dwQueryAttributes;
                    // Only store what we asked for
                    data.dwAttributes = rgfInOut & dwQueryAttributes;
                    data.cItems = apidl.GetSize();

                    hr = DataObject_SetData(pDataObject, g_DataObjectAttributes, &data, sizeof(data));
                    FAILED_UNEXPECTEDLY(hr);
                }
            }
        }

        // Only give the user what they asked for, not everything else we have!
        dwAttributes = data.dwAttributes & dwAttributeMask;
        cItems = data.cItems;
    }

    if (pdwAttributes)
        *pdwAttributes = dwAttributes;

    if (pcItems)
        *pcItems = cItems;

    return hr;
}
