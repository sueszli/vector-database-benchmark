
// HelloKeypadDlg.cpp : implementation file
//

#include "stdafx.h"
#include "HelloKeypad.h"
#include "HelloKeypadDlg.h"
#include "afxdialogex.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CAboutDlg dialog used for App About

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CHelloKeypadDlg dialog
CHelloKeypadDlg* CHelloKeypadDlg::m_the_dialog = NULL;
CRITICAL_SECTION CHelloKeypadDlg::m_criSection;

CHelloKeypadDlg::CHelloKeypadDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_HelloKeypad_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
	m_the_dialog = this;
	InitializeCriticalSection(&m_criSection);
}

void CHelloKeypadDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CHelloKeypadDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_WM_LBUTTONDOWN()
	ON_WM_LBUTTONUP()
	ON_WM_KEYDOWN()
	ON_WM_ERASEBKGND()
END_MESSAGE_MAP()


// CHelloKeypadDlg message handlers

BOOL CHelloKeypadDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	// TODO: Add extra initialization here
	AfxBeginThread(CHelloKeypadDlg::ThreadHelloKeypad, NULL);
	if (m_display_mode == FRAME_BUFFER_MODE)
	{
		AfxBeginThread(CHelloKeypadDlg::ThreadRefreshUI, NULL);
	}

	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CHelloKeypadDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CHelloKeypadDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CHelloKeypadDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

BOOL CHelloKeypadDlg::OnEraseBkgnd(CDC* pDC)
{
	return FALSE;
}

#define DISPLAY_WIDTH	240
#define DISPLAY_HEIGHT	320

struct DISPLAY_DRIVER
{
	void(*draw_pixel)(int x, int y, unsigned int rgb);
	void(*fill_rect)(int x0, int y0, int x1, int y1, unsigned int rgb);
} my_driver;

void CHelloKeypadDlg::draw_pixel(int x, int y, unsigned int rgb)
{
	CDC* pDC = m_the_dialog->GetDC();
	if (pDC)
	{
		pDC->SetPixel(x, y, RGB(((((unsigned int)(rgb)) >> 16) & 0xFF), ((((unsigned int)(rgb)) >> 8) & 0xFF), (((unsigned int)(rgb)) & 0xFF)));
	}	
}

void CHelloKeypadDlg::fill_rect(int x0, int y0, int x1, int y1, unsigned int rgb)
{
	CDC* pDC = m_the_dialog->GetDC();
	if (pDC)
	{
		CBrush brush;
		brush.CreateSolidBrush(RGB(((((unsigned int)(rgb)) >> 16) & 0xFF), ((((unsigned int)(rgb)) >> 8) & 0xFF), (((unsigned int)(rgb)) & 0xFF)));
		CRect rect(x0, y0, x1 + 1, y1 + 1);
		pDC->FillRect(&rect, &brush);
	}
}

UINT CHelloKeypadDlg::ThreadHelloKeypad(LPVOID pParam)
{
	Sleep(500);
	my_driver.draw_pixel = draw_pixel;
	my_driver.fill_rect = fill_rect;
	if (m_the_dialog->m_display_mode == FRAME_BUFFER_MODE)
	{
		startHelloKeypad(calloc(DISPLAY_WIDTH * DISPLAY_HEIGHT, m_the_dialog->m_color_bytes), DISPLAY_WIDTH, DISPLAY_HEIGHT, m_the_dialog->m_color_bytes, NULL);
	}
	else
	{
		startHelloKeypad(NULL, DISPLAY_WIDTH, DISPLAY_HEIGHT, m_the_dialog->m_color_bytes, &my_driver);
	}
	return 0;
}

UINT CHelloKeypadDlg::ThreadRefreshUI(LPVOID pParam)
{
	CDC* pDC = m_the_dialog->GetDC();
	while (true)
	{
		Sleep(30);
		m_the_dialog->updateUI(pDC);
	}
}

void CHelloKeypadDlg::updateUI(CDC* pDC)
{
	void* data = NULL;
	CRect rcClient;
	GetClientRect(&rcClient);

	if (rcClient.EqualRect(m_block_rect) && m_ui_width != 0)
	{
		data = getUiOfHelloKeypad(NULL, NULL);
		goto RENDER;
	}

	m_block_rect = rcClient;
	data = getUiOfHelloKeypad(&m_ui_width, &m_ui_height, true);

	memset(&m_ui_bm_info, 0, sizeof(m_ui_bm_info));
	m_ui_bm_info.bmiHeader.biSize = sizeof(MYBITMAPINFO);
	m_ui_bm_info.bmiHeader.biWidth = m_ui_width;
	m_ui_bm_info.bmiHeader.biHeight = (0 - m_ui_height);//fix upside down. 
	m_ui_bm_info.bmiHeader.biBitCount = m_color_bytes * 8;
	m_ui_bm_info.bmiHeader.biPlanes = 1;
	m_ui_bm_info.bmiHeader.biSizeImage = m_ui_width * m_ui_height * m_color_bytes;

	switch (m_color_bytes)
	{
	case 2:
		m_ui_bm_info.bmiHeader.biCompression = BI_BITFIELDS;
		m_ui_bm_info.biRedMask = 0xF800;
		m_ui_bm_info.biGreenMask = 0x07E0;
		m_ui_bm_info.biBlueMask = 0x001F;
		break;
	case 4:
		m_ui_bm_info.bmiHeader.biCompression = BI_RGB;
		break;
	default:
		ASSERT(FALSE);
		break;
	}

	m_memDC.DeleteDC();
	m_blockBmp.DeleteObject();

	m_memDC.CreateCompatibleDC(pDC);
	m_blockBmp.CreateCompatibleBitmap(pDC, m_block_rect.Width(), m_block_rect.Height());
	m_memDC.SelectObject(&m_blockBmp);
	::SetStretchBltMode(m_memDC, STRETCH_HALFTONE);

RENDER:
	if (!data)
	{
		return;
	}
	register int block_width = m_block_rect.Width();
	register int block_height = m_block_rect.Height();
	::StretchDIBits(m_memDC.m_hDC, 0, 0, block_width, block_height,
		0, 0, m_ui_width, m_ui_height, data, (BITMAPINFO*)& m_ui_bm_info, DIB_RGB_COLORS, SRCCOPY);
	pDC->BitBlt(m_block_rect.left, m_block_rect.top, block_width, block_height, &m_memDC, 0, 0, SRCCOPY);
}

void CHelloKeypadDlg::OnLButtonUp(UINT nFlags, CPoint point)
{
	sendTouch2HelloKeypad(point.x, point.y, false);
}

void CHelloKeypadDlg::OnLButtonDown(UINT nFlags, CPoint point)
{
	sendTouch2HelloKeypad(point.x, point.y, true);
}

void CHelloKeypadDlg::OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	unsigned int key = 2;
	switch (nChar)
	{
	case 68:
		key = 0;
		break;
	case 65:
		key = 1;
		break;
	}
	sendKey2HelloKeypad(key);
}