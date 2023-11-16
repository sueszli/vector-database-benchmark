/*
 * PROJECT:         ReactOS win32 kernel mode subsystem
 * LICENSE:         GPL - See COPYING in the top level directory
 * FILE:            win32ss/gdi/ntgdi/stockobj.c
 * PURPOSE:         Stock objects functions
 * PROGRAMMERS:     Colin Finck <colin@reactos.org>
 *                  Katayama Hirofumi MZ <katayama.hirofumi.mz@gmail.com>
 */

#include <win32k.h>

#define NDEBUG
#include <debug.h>


static const COLORREF SysColors[] =
{
    RGB(212, 208, 200), /* COLOR_SCROLLBAR  */
    RGB(58, 110, 165),  /* COLOR_BACKGROUND  */
    RGB(10, 36, 106),   /* COLOR_ACTIVECAPTION  */
    RGB(128, 128, 128), /* COLOR_INACTIVECAPTION  */
    RGB(212, 208, 200), /* COLOR_MENU  */
    RGB(255, 255, 255), /* COLOR_WINDOW  */
    RGB(0, 0, 0),       /* COLOR_WINDOWFRAME  */
    RGB(0, 0, 0),       /* COLOR_MENUTEXT  */
    RGB(0, 0, 0),       /* COLOR_WINDOWTEXT  */
    RGB(255, 255, 255), /* COLOR_CAPTIONTEXT  */
    RGB(212, 208, 200), /* COLOR_ACTIVEBORDER  */
    RGB(212, 208, 200), /* COLOR_INACTIVEBORDER  */
    RGB(128, 128, 128), /* COLOR_APPWORKSPACE  */
    RGB(10, 36, 106),   /* COLOR_HIGHLIGHT  */
    RGB(255, 255, 255), /* COLOR_HIGHLIGHTTEXT  */
    RGB(212, 208, 200), /* COLOR_BTNFACE  */
    RGB(128, 128, 128), /* COLOR_BTNSHADOW  */
    RGB(128, 128, 128), /* COLOR_GRAYTEXT  */
    RGB(0, 0, 0),       /* COLOR_BTNTEXT  */
    RGB(212, 208, 200), /* COLOR_INACTIVECAPTIONTEXT  */
    RGB(255, 255, 255), /* COLOR_BTNHIGHLIGHT  */
    RGB(64, 64, 64),    /* COLOR_3DDKSHADOW  */
    RGB(212, 208, 200), /* COLOR_3DLIGHT  */
    RGB(0, 0, 0),       /* COLOR_INFOTEXT  */
    RGB(255, 255, 225), /* COLOR_INFOBK  */
    RGB(181, 181, 181), /* COLOR_UNKNOWN  */
    RGB(0, 0, 128),     /* COLOR_HOTLIGHT  */
    RGB(166, 202, 240), /* COLOR_GRADIENTACTIVECAPTION  */
    RGB(192, 192, 192), /* COLOR_GRADIENTINACTIVECAPTION  */
    RGB(49, 106, 197),  /* COLOR_MENUHILIGHT  */
    RGB(236, 233, 216)  /* COLOR_MENUBAR  */
};

// System Bitmap DC
HDC hSystemBM;

/*  GDI stock objects */

static LOGPEN WhitePen =
    { PS_SOLID, { 0, 0 }, RGB(255,255,255) };

static LOGPEN BlackPen =
    { PS_SOLID, { 0, 0 }, RGB(0,0,0) };

static LOGPEN NullPen =
    { PS_NULL, { 0, 0 }, 0 };

static LOGFONTW OEMFixedFont =
    { 12, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, OEM_CHARSET,
      OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, FF_DONTCARE | FIXED_PITCH, L"Terminal"
    };

static LOGFONTW AnsiFixedFont =
    { 12, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, ANSI_CHARSET,
      OUT_DEFAULT_PRECIS, /*CLIP_DEFAULT_PRECIS*/ CLIP_STROKE_PRECIS, /*DEFAULT_QUALITY*/ PROOF_QUALITY, FF_DONTCARE | FIXED_PITCH, L"Courier"
    };

static LOGFONTW AnsiVarFont =
    { 12, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, ANSI_CHARSET,
      OUT_DEFAULT_PRECIS, /*CLIP_DEFAULT_PRECIS*/ CLIP_STROKE_PRECIS, /*DEFAULT_QUALITY*/ PROOF_QUALITY, FF_DONTCARE | VARIABLE_PITCH, L"MS Sans Serif"
    };

static LOGFONTW SystemFont =
    { 16, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
      OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, FF_DONTCARE | VARIABLE_PITCH, L"System"
    };

static LOGFONTW DeviceDefaultFont =
    { 16, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, ANSI_CHARSET,
      OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, FF_SWISS | VARIABLE_PITCH, L"System"
    };

static LOGFONTW SystemFixedFont =
    { 15, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
      OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, FF_DONTCARE | FIXED_PITCH, L"Fixedsys"
    };

static LOGFONTW DefaultGuiFont =
    { -11, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
      OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, /*DEFAULT_QUALITY*/ PROOF_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"MS Shell Dlg"
    };

HGDIOBJ StockObjects[NB_STOCK_OBJECTS];

static
HPEN
FASTCALL
IntCreateStockPen(DWORD dwPenStyle,
                  DWORD dwWidth,
                  ULONG ulBrushStyle,
                  ULONG ulColor)
{
    HPEN hPen;
    PBRUSH pbrushPen;

    pbrushPen = PEN_AllocPenWithHandle();
    if (pbrushPen == NULL) return NULL;

    if ((dwPenStyle & PS_STYLE_MASK) == PS_NULL) dwWidth = 1;

    pbrushPen->iHatch = 0;
    pbrushPen->lWidth = abs(dwWidth);
    FLOATOBJ_SetLong(&pbrushPen->eWidth, pbrushPen->lWidth);
    pbrushPen->ulPenStyle = dwPenStyle;
    pbrushPen->BrushAttr.lbColor = ulColor;
    pbrushPen->iBrushStyle = ulBrushStyle;
    pbrushPen->hbmClient = (HANDLE)NULL;
    pbrushPen->dwStyleCount = 0;
    pbrushPen->pStyle = 0;
    pbrushPen->flAttrs = BR_IS_OLDSTYLEPEN;

    switch (dwPenStyle & PS_STYLE_MASK)
    {
        case PS_NULL:
            pbrushPen->flAttrs |= BR_IS_NULL;
            break;

        case PS_SOLID:
            pbrushPen->flAttrs |= BR_IS_SOLID;
            break;
    }
    hPen = pbrushPen->BaseObject.hHmgr;
    PEN_UnlockPen(pbrushPen);
    return hPen;
}

static VOID FASTCALL
CreateStockFonts(void)
{
    USHORT ActiveCodePage, OemCodePage;
    BYTE bActiveCharSet, bOemCharSet;
    BOOL bIsCJK;
    static const WCHAR SimSun[] = { 0x5B8B, 0x4F53, 0 };
    static const WCHAR MingLiU[] = { 0x7D30, 0x660E, 0x9AD4, 0 };
    static const WCHAR Batang[] = { 0xBC14, 0xD0D5, 0 };

    RtlGetDefaultCodePage(&ActiveCodePage, &OemCodePage);
    bActiveCharSet = IntCharSetFromCodePage(ActiveCodePage);
    bOemCharSet = IntCharSetFromCodePage(OemCodePage);

    if (bOemCharSet == DEFAULT_CHARSET)
        bOemCharSet = OEM_CHARSET;

    switch (ActiveCodePage)
    {
        case 936:
            /* Simplified Chinese */
            bIsCJK = TRUE;
            wcscpy(DefaultGuiFont.lfFaceName, SimSun);
            break;

        case 950:
            /* Traditional Chinese */
            bIsCJK = TRUE;
            wcscpy(DefaultGuiFont.lfFaceName, MingLiU);
            break;

        case 932:
            /* Japanese */
            bIsCJK = TRUE;
            wcscpy(DefaultGuiFont.lfFaceName, L"MS UI Gothic");
            break;

        case 949:
        case 1361:
            /* Korean */
            bIsCJK = TRUE;
            wcscpy(DefaultGuiFont.lfFaceName, Batang);
            break;

        default:
            /* Otherwise */
            bIsCJK = FALSE;
            wcscpy(DefaultGuiFont.lfFaceName, L"MS Shell Dlg");
            break;
    }

    if (bIsCJK)
    {
        OEMFixedFont.lfHeight = 18;
        OEMFixedFont.lfPitchAndFamily = FF_DONTCARE | FIXED_PITCH;
        SystemFont.lfHeight = 18;
        SystemFont.lfPitchAndFamily = FF_DONTCARE | VARIABLE_PITCH;
        DeviceDefaultFont.lfHeight = 18;
        DeviceDefaultFont.lfPitchAndFamily = FF_DONTCARE | VARIABLE_PITCH;
        SystemFixedFont.lfHeight = 18;
        SystemFixedFont.lfPitchAndFamily = FF_DONTCARE | FIXED_PITCH;
        DefaultGuiFont.lfHeight = -12;
    }
    else
    {
        OEMFixedFont.lfHeight = 12;
        OEMFixedFont.lfPitchAndFamily = FF_MODERN | FIXED_PITCH;
        SystemFont.lfHeight = 16;
        SystemFont.lfPitchAndFamily = FF_SWISS | VARIABLE_PITCH;
        DeviceDefaultFont.lfHeight = 16;
        DeviceDefaultFont.lfPitchAndFamily = FF_SWISS | VARIABLE_PITCH;
        if (bActiveCharSet == RUSSIAN_CHARSET)
        {
            SystemFixedFont.lfHeight = 16;
            SystemFixedFont.lfPitchAndFamily = FF_SWISS | FIXED_PITCH;
        }
        else
        {
            SystemFixedFont.lfHeight = 15;
            SystemFixedFont.lfPitchAndFamily = FF_MODERN | FIXED_PITCH;
        }
        DefaultGuiFont.lfHeight = -11;
    }

    OEMFixedFont.lfCharSet = bOemCharSet;
    SystemFont.lfCharSet = bActiveCharSet;
    DeviceDefaultFont.lfCharSet = bActiveCharSet;
    SystemFixedFont.lfCharSet = bActiveCharSet;
    DefaultGuiFont.lfCharSet = bActiveCharSet;

    TextIntCreateFontIndirect(&OEMFixedFont, (HFONT*)&StockObjects[OEM_FIXED_FONT]);
    TextIntCreateFontIndirect(&AnsiFixedFont, (HFONT*)&StockObjects[ANSI_FIXED_FONT]);
    TextIntCreateFontIndirect(&AnsiVarFont, (HFONT*)&StockObjects[ANSI_VAR_FONT]);
    TextIntCreateFontIndirect(&SystemFont, (HFONT*)&StockObjects[SYSTEM_FONT]);
    TextIntCreateFontIndirect(&DeviceDefaultFont, (HFONT*)&StockObjects[DEVICE_DEFAULT_FONT]);
    TextIntCreateFontIndirect(&SystemFixedFont, (HFONT*)&StockObjects[SYSTEM_FIXED_FONT]);
    TextIntCreateFontIndirect(&DefaultGuiFont, (HFONT*)&StockObjects[DEFAULT_GUI_FONT]);
}

/*!
 * Creates a bunch of stock objects: brushes, pens, fonts.
*/
VOID FASTCALL
CreateStockObjects(void)
{
    UINT Object;

    DPRINT("Beginning creation of stock objects\n");

    /* Create GDI Stock Objects from the logical structures we've defined */

    StockObjects[WHITE_BRUSH] =  IntGdiCreateSolidBrush(RGB(255,255,255));
    StockObjects[DC_BRUSH]    =  IntGdiCreateSolidBrush(RGB(255,255,255));
    StockObjects[LTGRAY_BRUSH] = IntGdiCreateSolidBrush(RGB(192,192,192));
    StockObjects[GRAY_BRUSH] =   IntGdiCreateSolidBrush(RGB(128,128,128));
    StockObjects[DKGRAY_BRUSH] = IntGdiCreateSolidBrush(RGB(64,64,64));
    StockObjects[BLACK_BRUSH] =  IntGdiCreateSolidBrush(RGB(0,0,0));
    StockObjects[NULL_BRUSH] =   IntGdiCreateNullBrush();

    StockObjects[WHITE_PEN] = IntCreateStockPen(WhitePen.lopnStyle, WhitePen.lopnWidth.x, BS_SOLID, WhitePen.lopnColor);
    StockObjects[BLACK_PEN] = IntCreateStockPen(BlackPen.lopnStyle, BlackPen.lopnWidth.x, BS_SOLID, BlackPen.lopnColor);
    StockObjects[DC_PEN]    = IntCreateStockPen(BlackPen.lopnStyle, BlackPen.lopnWidth.x, BS_SOLID, BlackPen.lopnColor);
    StockObjects[NULL_PEN]  = IntCreateStockPen(NullPen.lopnStyle, NullPen.lopnWidth.x, BS_SOLID, NullPen.lopnColor);

    StockObjects[20] = NULL; /* TODO: Unknown internal stock object */
    StockObjects[DEFAULT_BITMAP] = GreCreateBitmap(1, 1, 1, 1, NULL);

    CreateStockFonts();

    StockObjects[DEFAULT_PALETTE] = (HGDIOBJ)gppalDefault->BaseObject.hHmgr;

    for (Object = 0; Object < NB_STOCK_OBJECTS; Object++)
    {
        if (NULL != StockObjects[Object])
        {
            GDIOBJ_ConvertToStockObj(&StockObjects[Object]);
        }
    }

    DPRINT("Completed creation of stock objects\n");
}

/*!
 * Return stock object.
 * \param	Object - stock object id.
 * \return	Handle to the object.
*/
HGDIOBJ APIENTRY
NtGdiGetStockObject(INT Object)
{
    DPRINT("NtGdiGetStockObject index %d\n", Object);

    return ((Object < 0) || (NB_STOCK_OBJECTS <= Object)) ? NULL : StockObjects[Object];
}

VOID FASTCALL
IntSetSysColors(UINT nColors, CONST INT *Elements, CONST COLORREF *Colors)
{
    UINT i;

    for (i = 0; i < nColors; i++)
    {
        if ((UINT)(*Elements) < NUM_SYSCOLORS)
        {
            gpsi->argbSystem[*Elements] = *Colors;
            IntGdiSetSolidBrushColor(gpsi->ahbrSystem[*Elements], *Colors);
        }
        Elements++;
        Colors++;
    }
}

HGDIOBJ FASTCALL
IntGetSysColorBrush(INT Object)
{
    return ((Object < 0) || (NUM_SYSCOLORS <= Object)) ? NULL : gpsi->ahbrSystem[Object];
}

DWORD FASTCALL
IntGetSysColor(INT nIndex)
{
    return (NUM_SYSCOLORS <= (UINT)nIndex) ? 0 : gpsi->argbSystem[nIndex];
}

VOID FASTCALL
CreateSysColorObjects(VOID)
{
    UINT i;

    for (i = 0; i < NUM_SYSCOLORS; i++)
    {
        gpsi->argbSystem[i] = SysColors[i];
    }

    /* Create the syscolor brushes */
    for (i = 0; i < NUM_SYSCOLORS; i++)
    {
        if (gpsi->ahbrSystem[i] == NULL)
        {
            gpsi->ahbrSystem[i] = IntGdiCreateSolidBrush(SysColors[i]);
            if (gpsi->ahbrSystem[i] != NULL)
            {
                GDIOBJ_ConvertToStockObj((HGDIOBJ*)&gpsi->ahbrSystem[i]);
            }
        }
    }
}

/* EOF */
