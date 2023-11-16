/*
 * NAppGUI Cross-platform C SDK
 * 2015-2023 Francisco Garcia Collado
 * MIT Licence
 * https://nappgui.com/en/legal/license.html
 *
 * File: oswindow.c
 *
 */

/* Operating System native window */

#include "oswindow.h"
#include "oswindow.inl"
#include "osgui.inl"
#include "osgui_win.inl"
#include "osbutton.inl"
#include "osedit.inl"
#include "oscontrol.inl"
#include "oscombo.inl"
#include "osmenuitem.inl"
#include "ospanel.inl"
#include "ospopup.inl"
#include "osslider.inl"
#include "arrpt.h"
#include "arrst.h"
#include "cassert.h"
#include "event.h"
#include "heap.h"

#if !defined(__WINDOWS__)
#error This file is only for Windows
#endif

typedef struct _hotkey_t HotKey;

typedef enum _wstate_t
{
    ekNORMAL,
    i_ekSTATE_MANAGED
} wstate_t;

struct _hotkey_t
{
    vkey_t key;
    uint32_t modifiers;
    Listener *listener;
};

DeclSt(HotKey);

struct _oswindow_t
{
    OSControl control;
    DWORD dwStyle;
    DWORD dwExStyle;
    BOOL bMenu;
    HMENU current_popup_menu;
    bool_t in_user_resizing;
    bool_t in_internal_resize;
    bool_t abort_resize;
    bool_t launch_resize_event;
    uint32_t resize_strategy;
    uint32_t flags;
    wstate_t state;
    gui_role_t role;
    OSPanel *main_panel;
    Listener *OnMoved;
	Listener *OnResize;
    Listener *OnClose;
    ArrSt(HotKey) *hotkeys;
    ArrPt(OSControl) *tabstops;
    HWND ctabstop;
    HCURSOR cursor;
    OSButton *defbutton;
    bool_t destroy_main_view;
};

/*---------------------------------------------------------------------------*/

#define i_WM_MODAL_STOP     0x444
static HWND i_CURRENT_ACTIVE_WINDOW = NULL;
static int i_SCROLL_OFFSET = 10;

/*---------------------------------------------------------------------------*/

static void i_resizing(OSWindow *window, WPARAM edge, RECT *wrect)
{
    cassert_no_null(window);
    cassert_no_null(wrect);
    if (window->launch_resize_event == TRUE)
    {
        if (window->OnResize != NULL)
        {
            RECT rc;
            BOOL ok;
            EvSize params;
            EvSize result;
			SetRectEmpty(&rc);
            ok = AdjustWindowRectEx(&rc, window->dwStyle, window->bMenu, window->dwExStyle);
            cassert_unref(ok != 0, ok);
            params.width = (real32_t)((wrect->right - wrect->left) - (rc.right - rc.left));
            params.height = (real32_t)((wrect->bottom - wrect->top) - (rc.bottom - rc.top));
            listener_event(window->OnResize, ekGUI_EVENT_WND_SIZING, window, &params, &result, OSWindow, EvSize, EvSize);

			rc.left = 0;
			rc.top = 0;
			rc.right = (LONG)result.width;
			rc.bottom = (LONG)result.height;
			ok = AdjustWindowRectEx(&rc, window->dwStyle, window->bMenu, window->dwExStyle);
			cassert_unref(ok != 0, ok);

			switch (edge) {
			case WMSZ_RIGHT:
				wrect->right = wrect->left + (rc.right - rc.left);
				break;
			case WMSZ_BOTTOM:
				wrect->bottom = wrect->top + (rc.bottom - rc.top);
				break;
			case WMSZ_LEFT:
				wrect->left = wrect->right - (rc.right - rc.left);
				break;
			case WMSZ_TOP:
				wrect->top = wrect->bottom - (rc.bottom - rc.top);
				break;
			case WMSZ_BOTTOMLEFT:
				wrect->bottom = wrect->top + (rc.bottom - rc.top);
				wrect->left = wrect->right - (rc.right - rc.left);
				break;
			case WMSZ_BOTTOMRIGHT:
				wrect->bottom = wrect->top + (rc.bottom - rc.top);
				wrect->right = wrect->left + (rc.right - rc.left);
				break;
			case WMSZ_TOPLEFT:
				wrect->top = wrect->bottom - (rc.bottom - rc.top);
				wrect->left = wrect->right - (rc.right - rc.left);
				break;
			case WMSZ_TOPRIGHT:
				wrect->top = wrect->bottom - (rc.bottom - rc.top);
				wrect->right = wrect->left + (rc.right - rc.left);
				break;
			default:
				break;
			}
        }
    }
}

/*---------------------------------------------------------------------------*/

static void i_resize(OSWindow* window, LONG width, LONG height)
{
	cassert_no_null(window);
	if (window->launch_resize_event == TRUE)
	{
		if (window->OnResize != NULL)
		{
			RECT rect;
			BOOL ok;
			EvSize params;
			rect.left = 0;
			rect.top = 0;
			rect.right = width;
			rect.bottom = height;
			ok = AdjustWindowRectEx(&rect, window->dwStyle, window->bMenu, window->dwExStyle);
			cassert_unref(ok != 0, ok);
			params.width = (real32_t)width;
			params.height = (real32_t)height;
			listener_event(window->OnResize, ekGUI_EVENT_WND_SIZE, window, &params, NULL, OSWindow, EvSize, void);
		}
	}
}

/*---------------------------------------------------------------------------*/

static void i_moved(OSWindow *window, const int16_t content_x, const int16_t content_y)
{
    cassert_no_null(window);
    if (window->OnMoved != NULL)
    {
        RECT rect;
        BOOL ok;
        EvPos params;
        rect.left = content_x;
        rect.top = content_y;
        rect.right = content_x + 100;
        rect.bottom = content_y + 100;
        ok = AdjustWindowRectEx(&rect, window->dwStyle, window->bMenu, window->dwExStyle);
        cassert_unref(ok != 0, ok);
        params.x = (real32_t)(rect.left);
        params.y = (real32_t)(rect.top);
        listener_event(window->OnMoved, ekGUI_EVENT_WND_MOVED, window, &params, NULL, OSWindow, EvPos, void);
    }
}

/*---------------------------------------------------------------------------*/

static __INLINE HWND i_focus_hwnd(const OSControl *control)
{
    cassert_no_null(control);
    switch (control->type) {
    case ekGUI_TYPE_LABEL:
    case ekGUI_TYPE_PROGRESS:
        return NULL;

    case ekGUI_TYPE_BUTTON:
    case ekGUI_TYPE_EDITBOX:
    case ekGUI_TYPE_SLIDER:
    case ekGUI_TYPE_TEXTVIEW:
    case ekGUI_TYPE_UPDOWN:
    case ekGUI_TYPE_CUSTOMVIEW:
        return control->hwnd;

    case ekGUI_TYPE_POPUP:
        return _ospopup_focus((OSPopUp*)control);

    case ekGUI_TYPE_COMBOBOX:
        return _oscombo_focus((OSCombo*)control);

    case ekGUI_TYPE_TABLEVIEW:
    case ekGUI_TYPE_TREEVIEW:
    case ekGUI_TYPE_BOXVIEW:
    case ekGUI_TYPE_SPLITVIEW:
    case ekGUI_TYPE_PANEL:
    case ekGUI_TYPE_LINE:
    case ekGUI_TYPE_HEADER:
    case ekGUI_TYPE_WINDOW:
    case ekGUI_TYPE_TOOLBAR:
    cassert_default();
    }

    return NULL;
}

/*---------------------------------------------------------------------------*/

static uint32_t i_search_tabstop(const OSControl **tabstop, const uint32_t size, HWND hwnd)
{
    register uint32_t i;
    if (hwnd == NULL)
        return UINT32_MAX;

    for (i = 0; i < size; ++i)
        if (i_focus_hwnd(tabstop[i]) == hwnd)
            return i;

    return UINT32_MAX;
}

/*---------------------------------------------------------------------------*/

static bool_t i_close(OSWindow *window, const gui_close_t close_origin)
{
    bool_t closed = TRUE;
    cassert_no_null(window);

    /* Before close, finish a possible text editing */
    if (close_origin == ekGUI_CLOSE_INTRO)
    {
        HWND hwnd = GetFocus();
        register const OSControl **tabstop = arrpt_all(window->tabstops, OSControl);
        register uint32_t size = arrpt_size(window->tabstops, OSControl);
        register uint32_t tabindex = i_search_tabstop(tabstop, size, hwnd);
        if (tabindex != UINT32_MAX)
        {
            if (tabstop[tabindex]->type == ekGUI_TYPE_EDITBOX)
                _osedit_kill_focus((const OSEdit*)tabstop[tabindex]);
        }
    }

    if (window->OnClose != NULL)
    {
        EvWinClose params;
        params.origin = close_origin;
        listener_event(window->OnClose, ekGUI_EVENT_WND_CLOSE, window, &params, &closed, OSWindow, EvWinClose, bool_t);
    }

    return closed;
}

/*---------------------------------------------------------------------------*/

static void i_menu_command(HWND hwnd, HMENU popup_hmenu, WORD command_id)
{
    MENUITEMINFO info = {0};
    HMENU hmenu = NULL;
    BOOL ok = FALSE;
    info.cbSize = sizeof(MENUITEMINFO);
    info.fMask = MIIM_DATA | MIIM_ID | MIIM_FTYPE | MIIM_STATE;

    hmenu = GetMenu(hwnd);
    if (hmenu != NULL)
        ok = GetMenuItemInfo(hmenu, command_id, MF_BYCOMMAND, &info);

    if (ok == FALSE && popup_hmenu != NULL)
        ok = GetMenuItemInfo(popup_hmenu, command_id, MF_BYCOMMAND, &info);

    /* Command from accelerator without active menu doesn't send event */
    if (ok == TRUE)
    {
        cassert(info.wID == command_id);
        _osmenuitem_click((OSMenuItem*)info.dwItemData, info.wID, info.fType, info.fState);
    }
}

/*---------------------------------------------------------------------------*/

/*#include "log.h"
static void i_log_control(HWND hwnd, uint32_t taborder)
{
    OSControl *control = (OSControl*)GetWindowLongPtr(hwnd, GWLP_USERDATA);
    if (control != NULL)
    {
        switch (control->type)
        {
            case ekGUI_TYPE_BOXVIEW:
                log_printf("%d: BoxView", taborder);
                break;
            case ekGUI_TYPE_BUTTON:
                log_printf("%d: Button", taborder);
                break;
            case ekGUI_TYPE_EDITBOX:
                log_printf("%d: EditBox", taborder);
                break;
            case ekGUI_TYPE_COMBOBOX:
                log_printf("%d: ComboBox", taborder);
                break;
            case ekGUI_TYPE_CUSTOMVIEW:
                log_printf("%d: CView", taborder);
                break;
            case ekGUI_TYPE_LABEL:
                log_printf("%d: Label", taborder);
                break;
            case ekGUI_TYPE_LINE:
                log_printf("%d: Line", taborder);
                break;
            case ekGUI_TYPE_PANEL:
                log_printf("%d: Panel", taborder);
                break;
            case ekGUI_TYPE_POPUP:
                log_printf("%d: PopUp", taborder);
                break;
            case ekGUI_TYPE_PROGRESS:
                log_printf("%d: Progress", taborder);
                break;
            case ekGUI_TYPE_SLIDER:
                log_printf("%d: Slider", taborder);
                break;
            case ekGUI_TYPE_TABLEVIEW:
                log_printf("%d: TableView", taborder);
                break;
            case ekGUI_TYPE_TEXTVIEW:
                log_printf("%d: VText", taborder);
                break;
            case ekGUI_TYPE_TREEVIEW:
                log_printf("%d: TreeView", taborder);
                break;
            case ekGUI_TYPE_UPDOWN:
                log_printf("%d: UpDown", taborder);
                break;
            case ekGUI_TYPE_WINDOW:
                log_printf("%d: Window", taborder);
                break;
            case ekGUI_TYPE_HEADER:
                log_printf("%d: Header", taborder);
                break;
            cassert_default();
        }
    }
    else
    {
        log_printf("%d: Child Window", taborder);
    }
}*/

/*---------------------------------------------------------------------------*/

static void i_set_tabstop(const OSControl **tabstop, const uint32_t size, const uint32_t index, const bool_t reverse, HWND *ctabstop)
{
    register uint32_t idx = index, i;
    cassert(index < size);
    cassert_no_null(ctabstop);
    for (i = 0 ; i < size; ++i)
    {
        register HWND hwnd = i_focus_hwnd(tabstop[idx]);
        if (hwnd && IsWindowEnabled(hwnd) && IsWindowVisible(hwnd))
        {
            OSControl *parent = (OSControl*)GetWindowLongPtr(GetParent(hwnd), GWLP_USERDATA);

            *ctabstop = hwnd;
            SetFocus(hwnd);

            if (parent != NULL && parent->type == ekGUI_TYPE_PANEL)
            {
                OSPanel *panel = (OSPanel*)parent;
                /* Automatic panel scrolling if control is not completely visible */
                if (_ospanel_with_scroll(panel) == TRUE)
                {
                    const OSControl *control = tabstop[idx];
                    RECT prect, crect;
                    int scroll_x = INT_MAX, scroll_y = INT_MAX;
                    _ospanel_scroll_frame((OSPanel*)parent, &prect);
                    GetWindowRect(hwnd, &crect);
                    crect.right = (crect.right - crect.left);
                    crect.bottom = (crect.bottom - crect.top);
                    crect.left = control->x;
                    crect.right += control->x;
                    crect.top = control->y;
                    crect.bottom += control->y;

                    if (prect.left > crect.left)
                        scroll_x = (crect.left - i_SCROLL_OFFSET);
                    else if (prect.right < crect.right)
                        scroll_x = (crect.right + i_SCROLL_OFFSET) - (prect.right - prect.left);

                    if (prect.top > crect.top)
                        scroll_y = (crect.top - i_SCROLL_OFFSET);
                    else if (prect.bottom < crect.bottom)
                        scroll_y = (crect.bottom + i_SCROLL_OFFSET) - (prect.bottom - prect.top);

                    if (scroll_x != INT_MAX || scroll_y != INT_MAX)
                        _ospanel_scroll(panel, scroll_x, scroll_y);
                }
            }

            return;
        }

        if (reverse == TRUE)
        {
            if (idx == 0)
                idx = size - 1;
            else
                idx -= 1;
        }
        else
        {
            if (idx == size - 1)
                idx = 0;
            else
                idx += 1;
        }
    }
}

/*---------------------------------------------------------------------------*/

static void i_set_next_tabstop(const ArrPt(OSControl) *tabstops, HWND hwnd, HWND *ctabstop)
{
    register uint32_t size = arrpt_size(tabstops, OSControl);
    if (size > 0)
    {
        register const OSControl **tabstop = arrpt_all_const(tabstops, OSControl);
        register uint32_t tabindex = i_search_tabstop(tabstop, size, hwnd);
        if (tabindex == UINT32_MAX)
            tabindex = 0;
        if (tabindex == size - 1)
            tabindex = 0;
        else
            tabindex += 1;
        i_set_tabstop(tabstop, size, tabindex, FALSE, ctabstop);
    }
}

/*---------------------------------------------------------------------------*/

static void i_set_previous_tabstop(const ArrPt(OSControl) *tabstops, HWND hwnd, HWND *ctabstop)
{
    register uint32_t size = arrpt_size(tabstops, OSControl);
    if (size > 0)
    {
        register const OSControl **tabstop = arrpt_all_const(tabstops, OSControl);
        register uint32_t tabindex = i_search_tabstop(tabstop, size, hwnd);
        if (tabindex == UINT32_MAX)
            tabindex = 0;
        if (tabindex == 0)
            tabindex = size - 1;
        else
            tabindex -= 1;
        i_set_tabstop(tabstop, size, tabindex, TRUE, ctabstop);
    }
}

/*---------------------------------------------------------------------------*/

static void i_set_ctabstop(const ArrPt(OSControl) *tabstops, HWND *ctabstop)
{
    register uint32_t size = arrpt_size(tabstops, OSControl);
    cassert_no_null(ctabstop);
    if (size > 0)
    {
        register const OSControl **tabstop = arrpt_all_const(tabstops, OSControl);
        register uint32_t tabindex = 0;

        if (*ctabstop == NULL)
            *ctabstop = tabstop[0]->hwnd;

        tabindex = i_search_tabstop(tabstop, size, *ctabstop);
        if (tabindex == UINT32_MAX)
            tabindex = 0;

        i_set_tabstop(tabstop, size, tabindex, FALSE, ctabstop);
    }
}

/*---------------------------------------------------------------------------*/

static void i_press_defbutton(OSWindow *window)
{
    if (window->defbutton != NULL)
    {
        HWND focus_hwnd = GetFocus();
        SetFocus(((OSControl*)window->defbutton)->hwnd);
        _osbutton_command(window->defbutton, (WPARAM)MAKELONG(0, BN_CLICKED));
        SetFocus(focus_hwnd);
    }
}

/*---------------------------------------------------------------------------*/

static LRESULT CALLBACK i_WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    OSWindow *window = (OSWindow*)GetWindowLongPtr(hwnd, GWLP_USERDATA);
    cassert_no_null(window);

	switch(msg)
	{
        case WM_ACTIVATE:
            if (wParam == WA_ACTIVE || wParam == WA_CLICKACTIVE)
            {
                i_CURRENT_ACTIVE_WINDOW = hwnd;
                i_set_ctabstop(window->tabstops, &window->ctabstop);
            }
            else
            {
                if (window->role == ekGUI_ROLE_OVERLAY)
                {
                    if (i_close(window, ekGUI_CLOSE_DEACT) == TRUE)
                        window->role = ENUM_MAX(gui_role_t);
                }

                i_CURRENT_ACTIVE_WINDOW = NULL;
            }

            return 0;

        case WM_COMMAND:

            /* COMMAND by Control */
            if (lParam != 0)
            {
                /* Press enter when a button has de focus */
                OSControl *control;
                control = (OSControl*)GetWindowLongPtr((HWND)lParam, GWLP_USERDATA);
                cassert_no_null(control);
                cassert(control->type == ekGUI_TYPE_BUTTON);
                _osbutton_command((OSButton*)control, wParam);
                return 0;
            }

            /* COMMAND by Menu */
            if (HIWORD(wParam) == 0)
            {
                /* DialogBox "Menu" Commands �? */
                switch (LOWORD(wParam))
                {
                    case IDCANCEL:
                    {
                        i_close(window, ekGUI_CLOSE_ESC);
                        return TRUE;
                    }

                    case IDOK:
                        i_press_defbutton(window);
                        i_close(window, ekGUI_CLOSE_INTRO);
                        return 0;
                }

                /* 'Real' Menu Command */
                i_menu_command(hwnd, window->current_popup_menu, LOWORD(wParam));
                return 0;
            }
            /* COMMAND by Accelerator */
            else
            {
                cassert(HIWORD(wParam) == 1);
                i_menu_command(hwnd, window->current_popup_menu, LOWORD(wParam));
                return 0;
            }

        case WM_PAINT:
            if (window->in_internal_resize == TRUE)
                return 0;

            break;

		case WM_SIZING:
            if (window->resize_strategy > 0 && window->in_user_resizing == TRUE)
            {
				cassert(FALSE);
                if (window->in_internal_resize == FALSE)
                {
                    window->in_internal_resize = TRUE;
                    ShowWindow(((OSControl*)window->main_panel)->hwnd, SW_HIDE);
                }
            }

            i_resizing(window, wParam, (RECT*)lParam);
		    return TRUE;

        case WM_ENTERSIZEMOVE:
            window->in_user_resizing = TRUE;
            break;

        case WM_EXITSIZEMOVE:
            if (window->in_internal_resize == TRUE)
                ShowWindow(((OSControl*)window->main_panel)->hwnd, SW_SHOW);
            window->in_user_resizing = FALSE;
            window->in_internal_resize = FALSE;
            break;

        case WM_SIZE:

            if (IsWindowVisible(window->control.hwnd) == FALSE)
                return 0;

            if (window->abort_resize == TRUE)
            {
                window->abort_resize = FALSE;
                return 0;
            }

            if (wParam == SIZE_MINIMIZED)
                return 0;

            if (window->in_internal_resize == TRUE)
            {
                HDC memHdc = NULL;
                LONG width, height;
                HDC hdc;
				cassert(FALSE);
                _ospanel_resize_double_buffer(window->main_panel, LOWORD(lParam), HIWORD(lParam));
                memHdc = _ospanel_paint_double_buffer(window->main_panel, window->resize_strategy, &width, &height);
                hdc = GetDC(window->control.hwnd);
                BitBlt(hdc, 0, 0, (int)width, (int)height, memHdc, 0, 0, SRCCOPY);
                ReleaseDC(window->control.hwnd, hdc);
            }

            {
                RECT rect;
                BOOL ok;
                rect.left = 0;
                rect.top = 0;
                rect.right = LOWORD(lParam);
                rect.bottom = HIWORD(lParam);
                ok = AdjustWindowRectEx(&rect, window->dwStyle, window->bMenu, window->dwExStyle);
                cassert(ok != 0);
                i_resizing(window, 1, &rect);
            }

            if (window->resize_strategy == 0)
                window->in_internal_resize = TRUE;

            i_resize(window, LOWORD(lParam), HIWORD(lParam));

            if (window->resize_strategy == 0)
                window->in_internal_resize = FALSE;

			return 0;

		case WM_MOVE:
            i_moved(window, (int16_t)LOWORD(lParam), (int16_t)HIWORD(lParam));
            return 0;

		case WM_CLOSE:
            i_close(window, ekGUI_CLOSE_BUTTON);
		    return 0;

        case WM_MEASUREITEM:
        {
            MEASUREITEMSTRUCT *mi = (MEASUREITEMSTRUCT*)lParam;
            /* Sent by menu */
            cassert((UINT)wParam == 0);
            cassert_no_null(mi);
            cassert(mi->CtlType == ODT_MENU);
            _osmenuitem_image_size((OSMenuItem*)mi->itemData, mi->itemID, &mi->itemWidth, &mi->itemHeight);
            return TRUE;
        }

        case WM_DRAWITEM:
        {
            DRAWITEMSTRUCT *di = (DRAWITEMSTRUCT*)lParam;
            /* Sent by menu */
            cassert((UINT)wParam == 0);
            cassert_no_null(di);
            cassert(di->CtlType == ODT_MENU);
            _osmenuitem_draw_image((OSMenuItem*)di->itemData, di->itemID, di->itemState, di->hDC, &di->rcItem);
            return TRUE;
        }

        case WM_SETCURSOR:
            if (window->cursor != NULL)
            {
                SetCursor(window->cursor);
                return TRUE;
            }
            break;

        case i_WM_MODAL_STOP:
            PostQuitMessage((int)wParam);
		    return 0;
	}

    return CallWindowProc(window->control.def_wnd_proc, hwnd, msg, wParam, lParam);
}

/*---------------------------------------------------------------------------*/

static void i_window_style(const window_flag_t flags, DWORD *dwStyle, DWORD *dwExStyle)
{
    cassert_no_null(dwStyle);
    cassert_no_null(dwExStyle);

    *dwStyle = 0 | WS_CLIPCHILDREN | WS_CLIPSIBLINGS;
    *dwExStyle = 0;

    if (flags & ekWINDOW_EDGE)
        *dwStyle |= WS_BORDER;

    if (flags & ekWINDOW_TITLE)
    {
        *dwStyle |= WS_CAPTION | WS_DLGFRAME | WS_OVERLAPPED;
        *dwExStyle |= WS_EX_DLGMODALFRAME;
    }

    if (flags & ekWINDOW_CLOSE)
        *dwStyle |= WS_SYSMENU;

    if (flags & ekWINDOW_MAX)
        *dwStyle |= WS_MAXIMIZEBOX | WS_SYSMENU;

    if (flags & ekWINDOW_MIN)
        *dwStyle |= WS_MINIMIZEBOX | WS_SYSMENU;

    if (flags & ekWINDOW_RESIZE)
        *dwStyle |= WS_THICKFRAME;
}

/*---------------------------------------------------------------------------*/

OSWindow *oswindow_create(const uint32_t flags)
{
    OSWindow *window = heap_new0(OSWindow);
    window->control.type = ekGUI_TYPE_WINDOW;
    i_window_style(flags, &window->dwStyle, &window->dwExStyle);
    _oscontrol_init_hidden((OSControl*)window, window->dwExStyle, window->dwStyle | WS_POPUP, kWINDOW_CLASS, 0, 0, i_WndProc, GetDesktopWindow());
    window->launch_resize_event = TRUE;
    window->flags = flags;
    window->state = ekNORMAL;
    window->role = ENUM_MAX(gui_role_t);
    window->tabstops = arrpt_create(OSControl);
    window->destroy_main_view = TRUE;

    {
        HICON icon = LoadIcon(_osgui_instance(), L"APPLICATION_ICON");
        if (icon != NULL)
        {
            SendMessage(window->control.hwnd, WM_SETICON, ICON_BIG, (LPARAM)icon);
            SendMessage(window->control.hwnd, WM_SETICON, ICON_SMALL, (LPARAM)icon);
        }
    }

    return window;
}

/*---------------------------------------------------------------------------*/

OSWindow *oswindow_managed(void *native_ptr)
{
    OSWindow *window = heap_new(OSWindow);
    cassert_no_null(native_ptr);
    window->control.type = ekGUI_TYPE_WINDOW;
    window->control.hwnd = (HWND)native_ptr;
    window->control.def_wnd_proc = NULL;
    window->launch_resize_event = TRUE;
    window->state = i_ekSTATE_MANAGED;
    window->role = ENUM_MAX(gui_role_t);
    window->destroy_main_view = TRUE;
	return window;
}

/*---------------------------------------------------------------------------*/

static void i_remove_hotkey(HotKey *hotkey)
{
    listener_destroy(&hotkey->listener);
}

/*---------------------------------------------------------------------------*/

void oswindow_destroy(OSWindow **window)
{
    cassert_no_null(window);
    cassert_no_null(*window);
    if ((*window)->destroy_main_view == TRUE && (*window)->main_panel != NULL)
    {
        OSPanel *panel = (*window)->main_panel;
        oswindow_detach_panel(*window, panel);
        _ospanel_destroy(&panel);
    }

    cassert((*window)->main_panel == NULL);
    cassert(_oscontrol_num_children((*window)->control.hwnd) == 0);
    listener_destroy(&(*window)->OnMoved);
	listener_destroy(&(*window)->OnResize);
    listener_destroy(&(*window)->OnClose);

    arrst_destopt(&(*window)->hotkeys, i_remove_hotkey, HotKey);
    arrpt_destopt(&(*window)->tabstops, NULL, OSControl);

    if ((*window)->state != i_ekSTATE_MANAGED)
        _oscontrol_destroy((OSControl*)(*window));

    heap_delete(window, OSWindow);
}

/*---------------------------------------------------------------------------*/

void oswindow_OnMoved(OSWindow *window, Listener *listener)
{
	cassert_no_null(window);
    cassert(window->state != i_ekSTATE_MANAGED);
    listener_update(&window->OnMoved, listener);
}

/*---------------------------------------------------------------------------*/

void oswindow_OnResize(OSWindow *window, Listener *listener)
{
	cassert_no_null(window);
    cassert(window->state != i_ekSTATE_MANAGED);
    listener_update(&window->OnResize, listener);
}

/*---------------------------------------------------------------------------*/

void oswindow_OnClose(OSWindow *window, Listener *listener)
{
	cassert_no_null(window);
    cassert(window->state != i_ekSTATE_MANAGED);
    listener_update(&window->OnClose, listener);
}

/*---------------------------------------------------------------------------*/

void oswindow_title(OSWindow *window, const char_t *text)
{
	cassert_no_null(window);
    cassert(window->state != i_ekSTATE_MANAGED);
    _oscontrol_set_text((OSControl*)window, text);
}

/*---------------------------------------------------------------------------*/

void oswindow_edited(OSWindow *window, const bool_t is_edited)
{
	unref(window);
	unref(is_edited);
	cassert(FALSE);
}

/*---------------------------------------------------------------------------*/

void oswindow_movable(OSWindow *window, const bool_t is_movable)
{
	unref(window);
	unref(is_movable);
	cassert(FALSE);
}

/*---------------------------------------------------------------------------*/

void oswindow_z_order(OSWindow *window, OSWindow *below_window)
{
    HWND parent = NULL;
	cassert_no_null(window);
    cassert(FALSE);
    cassert(window->state != i_ekSTATE_MANAGED);

    if (below_window != NULL)
        parent = below_window->control.hwnd;

    SetWindowLongPtr(window->control.hwnd, GWLP_HWNDPARENT, (LONG_PTR)parent);
}

/*---------------------------------------------------------------------------*/

void oswindow_alpha(OSWindow *window, const real32_t alpha)
{
	unref(window);
	unref(alpha);
	cassert(FALSE);
}

/*---------------------------------------------------------------------------*/

void oswindow_enable_mouse_events(OSWindow *window, const bool_t enabled)
{
	unref(window);
	unref(enabled);
	cassert(FALSE);
}

/*---------------------------------------------------------------------------*/

void oswindow_hotkey(OSWindow *window, const vkey_t key, const uint32_t modifiers, Listener *listener)
{
    cassert_no_null(window);
    if (window->hotkeys == NULL && listener != NULL)
        window->hotkeys = arrst_create(HotKey);

    /* Update the hotkey(if exists) */
    arrst_foreach(hotkey, window->hotkeys, HotKey)
        if (hotkey->key == key && hotkey->modifiers == modifiers)
        {
            listener_update(&hotkey->listener, listener);
            return;
        }
    arrst_end();

    /* Adds a new hotkey */
    if (listener != NULL)
    {
        HotKey *hotkey = arrst_new(window->hotkeys, HotKey);
        hotkey->key = key;
        hotkey->modifiers = modifiers;
        hotkey->listener = listener;
    }
}

/*---------------------------------------------------------------------------*/

void oswindow_taborder(OSWindow *window, OSControl *control)
{
    cassert_no_null(window);
    cassert(window->state != i_ekSTATE_MANAGED);
    if (control != NULL)
    {
        cassert(control->type != ekGUI_TYPE_PANEL);
        arrpt_append(window->tabstops, control, OSControl);
    }
    else
    {
        arrpt_clear(window->tabstops, NULL, OSControl);
        /* Force to show the focus rectangle in all controls */
        /* https://stackoverflow.com/questions/46489537/focus-rectangle-not-showing-even-if-control-has-focus */
        SendMessage(window->control.hwnd, WM_UPDATEUISTATE, MAKEWPARAM(UIS_CLEAR, UISF_HIDEFOCUS), (LPARAM)NULL);
    }
}

/*---------------------------------------------------------------------------*/

void oswindow_tabstop(OSWindow* window, const bool_t next)
{
    HWND hwnd = GetFocus();
    cassert_no_null(window);
    if (next == TRUE)
        i_set_next_tabstop(window->tabstops, hwnd, &window->ctabstop);
    else
        i_set_previous_tabstop(window->tabstops, hwnd, &window->ctabstop);
}

/*---------------------------------------------------------------------------*/

void oswindow_focus(OSWindow *window, OSControl *control)
{
    cassert_no_null(window);
    cassert_no_null(control);
    cassert(window->state != i_ekSTATE_MANAGED);
    arrpt_foreach(tabstop, window->tabstops, OSControl)
        if (tabstop->hwnd == control->hwnd)
        {
            SetFocus(control->hwnd);
            window->ctabstop = control->hwnd;
            break;
        }
    arrpt_end();
}

/*---------------------------------------------------------------------------*/

void oswindow_attach_panel(OSWindow *window, OSPanel *panel)
{
    cassert_no_null(window);
    cassert(window->state != i_ekSTATE_MANAGED);
    cassert(window->main_panel == NULL);
    _oscontrol_attach_to_parent((OSControl*)panel, (OSControl*)window);
    window->main_panel = panel;
}

/*---------------------------------------------------------------------------*/

void oswindow_detach_panel(OSWindow *window, OSPanel *panel)
{
    cassert_no_null(window);
    cassert(window->state != i_ekSTATE_MANAGED);
    cassert(window->main_panel == panel);
    _oscontrol_detach_from_parent((OSControl*)panel, (OSControl*)window);
    window->main_panel = NULL;
}

/*---------------------------------------------------------------------------*/

void oswindow_attach_window(OSWindow *parent_window, OSWindow *child_window)
{
    //HWND prevParent = 0;
    unref(parent_window);
	unref(child_window);
    /*prevParent = SetParent(child_window->control.hwnd, parent_window->control.hwnd);
    unref(prevParent);
    prevParent = GetParent(child_window->control.hwnd);*/
    //SetWindowLong(child_window->control.hwnd, GWL_STYLE, child_window->dwStyle | WS_CHILD);
    //SetWindowLong(child_window->control.hwnd, GWL_EXSTYLE, child_window->dwExStyle);
    //oswindow_set_z_order(child_window, parent_window);
}

/*---------------------------------------------------------------------------*/

void oswindow_detach_window(OSWindow *parent_window, OSWindow *child_window)
{
    //HWND prevParent = 0;
    unref(parent_window);
    unref(child_window);
   /* cassert_no_null(parent_window);
	cassert_no_null(child_window);
    prevParent = SetParent(child_window->control.hwnd, GetDesktopWindow());
    cassert(prevParent == parent_window->control.hwnd);*/
    /*SetWindowLong(child_window->control.hwnd, GWL_STYLE, child_window->dwStyle);
    SetWindowLong(child_window->control.hwnd, GWL_EXSTYLE, child_window->dwExStyle);*/
}

/*---------------------------------------------------------------------------*/

void oswindow_launch(OSWindow *window, OSWindow *parent_window)
{
	cassert_no_null(window);
    cassert(window->state != i_ekSTATE_MANAGED);

    if (parent_window != NULL)
    {
        SetWindowPos(window->control.hwnd, parent_window->control.hwnd, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE);
        window->role = ekGUI_ROLE_OVERLAY;
    }

    window->launch_resize_event = FALSE;
    _oscontrol_set_visible((OSControl*)window, TRUE);
    window->launch_resize_event = TRUE;

    SetActiveWindow(window->control.hwnd);
    i_set_ctabstop(window->tabstops, &window->ctabstop);


    //LONG style = 0;
    //style = GetWindowLong(window->control.hwnd, GWL_STYLE);
    //if (parent_window != NULL)
    //{
    //    RECT rect;
    //    POINT pt = { 0, 0 };
    //    style |= WS_CHILD;
    //    GetWindowRect(window->control.hwnd, &rect);
    //    ClientToScreen(parent_window->control.hwnd, &pt);
    //    SetWindowLong(window->control.hwnd, GWL_STYLE, style);
    //    SetParent(window->control.hwnd, parent_window->control.hwnd);
    //    SetWindowPos(window->control.hwnd, NULL, rect.left - pt.x, rect.top - pt.y, 0, 0, SWP_NOSIZE | SWP_NOZORDER);
    //    window->role = ekGUI_WINDOW_ROLE_MODAL;
    //}
    //else
    //{
    //    style &= ~WS_CHILD;
    //    SetWindowLong(window->control.hwnd, GWL_STYLE, style);
    //}
}

/*---------------------------------------------------------------------------*/

void oswindow_hide(OSWindow *window, OSWindow *parent_window)
{
	cassert_no_null(window);
    cassert(window->state != i_ekSTATE_MANAGED);
	unref(parent_window);
    /*
    LONG style = 0;
    style = GetWindowLong(window->control.hwnd, GWL_STYLE);
    style &= ~WS_CHILD;
    SetWindowLong(window->control.hwnd, GWL_STYLE, style);
    SetParent(window->control.hwnd, NULL);*/
    window->role = ENUM_MAX(gui_role_t);
    window->launch_resize_event = FALSE;
    _oscontrol_set_visible((OSControl*)window, FALSE);
    window->launch_resize_event = TRUE;
}

/*---------------------------------------------------------------------------*/

static OSWindow *i_get_window(HWND hwnd)
{
    OSWindow *window = NULL;
    while (window == NULL && hwnd != NULL)
    {
        window = (OSWindow*)GetWindowLongPtr(hwnd, GWLP_USERDATA);
        if (window != NULL)
        {
            if (window->control.type != ekGUI_TYPE_WINDOW)
                window = NULL;
        }

        if (window == NULL)
            hwnd = GetParent(hwnd);
    }

    return window;
}

/*---------------------------------------------------------------------------*/

static BOOL i_IsDialogMessage(HWND hDlg, LPMSG lpMsg)
{
    OSWindow *window = NULL;
    cassert_no_null(lpMsg);
    window = i_get_window(lpMsg->hwnd);

    if (window != NULL)
    {
        if (lpMsg->message == WM_KEYDOWN)
        {
            if (lpMsg->wParam == VK_TAB)
            {
                if (window->control.hwnd == hDlg)
                {
                    SHORT lshif_state = GetAsyncKeyState(VK_LSHIFT);
                    SHORT rshif_state = GetAsyncKeyState(VK_RSHIFT);
                    BOOL previous = ((0x8000 & lshif_state) != 0) || ((0x8000 & rshif_state) != 0);
                    HWND hwnd = GetFocus();
                    if (previous == TRUE)
                        i_set_previous_tabstop(window->tabstops, hwnd, &window->ctabstop);
                    else
                        i_set_next_tabstop(window->tabstops, hwnd, &window->ctabstop);
                    return TRUE;
                }
            }
            else if (lpMsg->wParam == VK_RETURN)
            {
                i_press_defbutton(window);
                if (window->flags & ekWINDOW_RETURN)
                {
                    i_close(window, ekGUI_CLOSE_INTRO);
                    return TRUE;
                }
            }
            else if (lpMsg->wParam == VK_ESCAPE)
            {
                if (window->flags & ekWINDOW_ESC)
                {
                    i_close(window, ekGUI_CLOSE_ESC);
                    return TRUE;
                }
            }

            /* Check hotkeys */
            if (window->hotkeys != NULL)
            {
                WPARAM key = lpMsg->wParam;
                uint32_t modifiers = 0;

                if ((GetAsyncKeyState(VK_LSHIFT) & 0x8000) || (GetAsyncKeyState(VK_RSHIFT) & 0x8000))
                    modifiers |= ekMKEY_SHIFT;

                if ((GetAsyncKeyState(VK_LCONTROL) & 0x8000) || (GetAsyncKeyState(VK_RCONTROL) & 0x8000))
                    modifiers |= ekMKEY_CONTROL;

                if ((GetAsyncKeyState(VK_LMENU) & 0x8000) || (GetAsyncKeyState(VK_RMENU) & 0x8000))
                    modifiers |= ekMKEY_ALT;

                if ((GetAsyncKeyState(VK_LWIN) & 0x8000) || (GetAsyncKeyState(VK_RWIN) & 0x8000))
                    modifiers |= ekMKEY_COMMAND;

                arrst_foreach(hotkey, window->hotkeys, HotKey)
                    if (key == kVIRTUAL_KEY[hotkey->key] && modifiers == hotkey->modifiers)
                    {
                        if (hotkey->listener != NULL)
                        {
                            EvKey params;
                            params.key = hotkey->key;
                            params.modifiers = hotkey->modifiers;
                            listener_event(hotkey->listener, ekGUI_EVENT_KEYDOWN, window, &params, NULL, OSWindow, EvKey, void);
                            return TRUE;
                        }
                    }
                arrst_end()
            }
        }
        else if (lpMsg->message == WM_SETFOCUS)
        {
            arrpt_foreach(tabstop, window->tabstops, OSControl)
                if (tabstop->hwnd == lpMsg->hwnd)
                {
                    window->ctabstop = lpMsg->hwnd;
                    break;
                }
            arrpt_end();
        }
    }

    return FALSE;
}

/*---------------------------------------------------------------------------*/

uint32_t oswindow_launch_modal(OSWindow *window, OSWindow *parent_window)
{
    MSG msg;

    cassert_no_null(window);
    cassert(window->state != i_ekSTATE_MANAGED);

    if (parent_window != NULL)
    {
        SetWindowLongPtr(window->control.hwnd, GWLP_HWNDPARENT, (LONG_PTR)parent_window->control.hwnd);
        _oscontrol_set_enabled((OSControl*)parent_window, FALSE);
    }

    window->launch_resize_event = FALSE;
    _oscontrol_set_visible((OSControl*)window, TRUE);
    window->launch_resize_event = TRUE;
    SetActiveWindow(window->control.hwnd);
    i_set_ctabstop(window->tabstops, &window->ctabstop);

	while(GetMessage(&msg, NULL, 0, 0) > 0)
	{
        if (!i_IsDialogMessage(window->control.hwnd, &msg))
        {
			TranslateMessage(&msg);
			DispatchMessage(&msg);
        }
	}

    if (parent_window != NULL)
    {
        SetWindowLongPtr(window->control.hwnd, GWLP_HWNDPARENT, (LONG_PTR)GetDesktopWindow());
        _oscontrol_set_enabled((OSControl*)parent_window, TRUE);
        SetWindowPos(parent_window->control.hwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
        SetActiveWindow(parent_window->control.hwnd);
        i_set_ctabstop(parent_window->tabstops, &parent_window->ctabstop);
    }

    return(uint32_t)msg.wParam;
}

/*---------------------------------------------------------------------------*/

void oswindow_stop_modal(OSWindow *window, const uint32_t return_value)
{
	cassert_no_null(window);
    cassert(window->state != i_ekSTATE_MANAGED);
    SendMessage(window->control.hwnd, i_WM_MODAL_STOP, (WPARAM)return_value, (LPARAM)0);
}

/*---------------------------------------------------------------------------*/

//void oswindow_launch_sheet(OSWindow *window, OSWindow *parent);
//void oswindow_launch_sheet(OSWindow *window, OSWindow *parent)
//{
//	unref(window);
//	unref(parent);
//	cassert(FALSE);
//}

/*---------------------------------------------------------------------------*/

//void oswindow_stop_sheet(OSWindow *window, OSWindow *parent);
//void oswindow_stop_sheet(OSWindow *window, OSWindow *parent)
//{
//	unref(window);
//	unref(parent);
//	cassert(FALSE);
//}

/*---------------------------------------------------------------------------*/

void oswindow_get_origin(const OSWindow *window, real32_t *x, real32_t *y)
{
    _oscontrol_get_origin_in_screen(window->control.hwnd, x, y);
}

/*---------------------------------------------------------------------------*/

void oswindow_origin(OSWindow *window, const real32_t x, const real32_t y)
{
	cassert_no_null(window);
    _oscontrol_set_position((OSControl*)window, (int)x, (int)y);
}

/*---------------------------------------------------------------------------*/

void oswindow_get_size(const OSWindow *window, real32_t *width, real32_t *height)
{
    _oscontrol_get_size((const OSControl*)window, width, height);
}

/*---------------------------------------------------------------------------*/

void oswindow_size(OSWindow *window, const real32_t content_width, const real32_t content_height)
{
    RECT rect;
    BOOL ok;
	cassert_no_null(window);
    cassert(window->state != i_ekSTATE_MANAGED);
    rect.left = 0;
    rect.top = 0;
    rect.right = (LONG)content_width;
    rect.bottom = (LONG)content_height;
    ok = AdjustWindowRectEx(&rect, window->dwStyle, window->bMenu, window->dwExStyle);
    cassert_unref(ok != 0, ok);
    window->launch_resize_event = FALSE;
    ok = SetWindowPos(window->control.hwnd, NULL, 0, 0, (int)(rect.right - rect.left), (int)(rect.bottom - rect.top), SWP_NOMOVE | SWP_NOZORDER);
    cassert_unref(ok != 0, ok);
    window->launch_resize_event = TRUE;
}

/*---------------------------------------------------------------------------*/

void oswindow_set_default_pushbutton(OSWindow *window, OSButton *button)
{
    cassert_no_null(window);
    if (window->defbutton != NULL)
        _osbutton_unset_default(window->defbutton);

    if (button != NULL)
        _osbutton_set_default(button);

    window->defbutton = button;
}

/*---------------------------------------------------------------------------*/

void oswindow_set_cursor(OSWindow *window, Cursor *cursor)
{
    cassert_no_null(window);
    window->cursor = (HCURSOR)cursor;
}

/*---------------------------------------------------------------------------*/

void oswindow_property(OSWindow *window, const gui_prop_t property, const void *value)
{
    cassert_no_null(window);
    if (property == ekGUI_PROP_RESIZE)
    {
        cassert_no_null(value);
        window->resize_strategy = *((uint32_t*)value);
    }
    else if (property == ekGUI_PROP_CHILDREN)
    {
        window->destroy_main_view = FALSE;
    }
}

/*---------------------------------------------------------------------------*/

static void i_adjust_menu_size(OSWindow *window)
{
    RECT rect;
    BOOL ok = FALSE;
    cassert_no_null(window);
    ok = GetClientRect(window->control.hwnd, &rect);
    cassert_unref(ok != 0, ok);
    ok = AdjustWindowRectEx(&rect, window->dwStyle, window->bMenu, window->dwExStyle);
    cassert_unref(ok != 0, ok);
    SetWindowPos(window->control.hwnd, NULL, 0, 0, (int)(rect.right - rect.left), (int)(rect.bottom - rect.top), SWP_NOMOVE | SWP_NOZORDER);
}

/*---------------------------------------------------------------------------*/

void _oswindow_set_menubar(OSWindow *window, HMENU hmenu)
{
    BOOL ok = FALSE;
    cassert_no_null(window);
    cassert_no_null(hmenu);
    cassert(window->bMenu == FALSE);
    cassert(GetMenu(window->control.hwnd) == NULL);
    window->bMenu = TRUE;
    i_adjust_menu_size(window);
    ok = SetMenu(window->control.hwnd, hmenu);
    cassert_unref(ok == TRUE, ok);
    cassert(GetMenu(window->control.hwnd) == hmenu);
}

/*---------------------------------------------------------------------------*/

void _oswindow_unset_menubar(OSWindow *window, HMENU hmenu)
{
    BOOL ok = FALSE;
    cassert_no_null(window);
    cassert_unref(GetMenu(window->control.hwnd) == hmenu, hmenu);
    cassert(window->bMenu == TRUE);
    window->bMenu = FALSE;
    i_adjust_menu_size(window);
    ok = SetMenu(window->control.hwnd, NULL);
    cassert_unref(ok == TRUE, ok);
    cassert(GetMenu(window->control.hwnd) == NULL);
}

/*---------------------------------------------------------------------------*/

void _oswindow_change_menubar(OSWindow *window, HMENU prev_hmenu, HMENU new_hmenu)
{
    BOOL ok = FALSE;
    cassert_no_null(window);
    cassert_no_null(prev_hmenu);
    cassert_no_null(new_hmenu);
    cassert(window->bMenu == TRUE);
    cassert(GetMenu(window->control.hwnd) == prev_hmenu);
    ok = SetMenu(window->control.hwnd, new_hmenu);
    cassert(ok == TRUE);
    cassert(GetMenu(window->control.hwnd) == new_hmenu);
}

/*---------------------------------------------------------------------------*/

HWND _oswindow_set_current_popup_menu(OSWindow *window, HMENU hmenu)
{
    cassert_no_null(window);
    window->current_popup_menu = hmenu;
    return window->control.hwnd;
}

/*---------------------------------------------------------------------------*/

HWND _oswindow_hwnd(OSWindow *window)
{
    if (window != NULL)
        return window->control.hwnd;
    else
        return NULL;
}

/*---------------------------------------------------------------------------*/

bool_t _oswindow_proccess_message(MSG *msg, HACCEL accelerator_table)
{
    if (i_CURRENT_ACTIVE_WINDOW != NULL)
    {
        BOOL accelerator_message = FALSE;
        if (accelerator_table != NULL)
            accelerator_message = TranslateAccelerator(i_CURRENT_ACTIVE_WINDOW, accelerator_table, msg);

        if (accelerator_message == TRUE)
        {
            return TRUE;
        }
        else
        {
            if (i_IsDialogMessage(i_CURRENT_ACTIVE_WINDOW, msg) != 0)
                return TRUE;
            else
                return FALSE;
        }
    }
    else
    {
        return FALSE;
    }
}

/*---------------------------------------------------------------------------*/

bool_t _oswindow_in_resizing(HWND child_hwnd)
{
    HWND hwnd = NULL;
    OSWindow *window = NULL;
    cassert_no_null(child_hwnd);
    hwnd = GetAncestor(child_hwnd, GA_ROOT);
    cassert(hwnd != NULL);
    window = (OSWindow*)GetWindowLongPtr(hwnd, GWLP_USERDATA);
    cassert_no_null(window);
    return window->in_internal_resize;
}

/*---------------------------------------------------------------------------*/

void _oswindow_focus(OSControl *control)
{
    HWND hwnd;
    OSWindow *window;
    cassert_no_null(control);
    hwnd = GetAncestor(control->hwnd, GA_ROOT);
    window = (OSWindow*)GetWindowLongPtr(hwnd, GWLP_USERDATA);
    cassert_no_null(window);
    oswindow_focus(window, control);
}
