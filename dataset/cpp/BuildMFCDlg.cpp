
// BuildMFCDlg.cpp : implementation file
//

#include "stdafx.h"
#include "BuildMFC.h"
#include "BuildMFCDlg.h"
#include "afxdialogex.h"
#include "UiBlock.h"

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

// CBuildMFCDlg dialog
CBuildMFCDlg* CBuildMFCDlg::m_the_dialog = 0;
CBuildMFCDlg::CBuildMFCDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_BUILDMFC_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
	m_the_dialog = this;
}

void CBuildMFCDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CBuildMFCDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_WM_ERASEBKGND()
	ON_WM_LBUTTONDOWN()
	ON_WM_LBUTTONUP()
	ON_WM_MOUSEMOVE()
	ON_WM_LBUTTONDBLCLK()
END_MESSAGE_MAP()

// CBuildMFCDlg message handlers

BOOL CBuildMFCDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(false);
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
	SetIcon(m_hIcon, true);			// Set big icon
	SetIcon(m_hIcon, false);		// Set small icon

	// TODO: Add extra initialization here
	AfxBeginThread(CBuildMFCDlg::ThreadHostMonitor, 0);
	AfxBeginThread(CBuildMFCDlg::ThreadRefreshUI, 0);
	ShellExecute(0, L"open", L"https://github.com/idea4good/GuiLite", L"", L"", SW_SHOWNORMAL);
	gSyncData = sync_data;
	sync_data(60, 98, 30, 120, 80, 100);//Ping cloud

	return true;  // return true  unless you set the focus to a control
}

void CBuildMFCDlg::OnSysCommand(UINT nID, LPARAM lParam)
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
void CBuildMFCDlg::OnPaint()
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
HCURSOR CBuildMFCDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

BOOL CBuildMFCDlg::OnEraseBkgnd(CDC* pDC)
{
	return true;
}

void CBuildMFCDlg::updateUI(CDC* pDC)
{
	CRect rcClient;
	GetClientRect(&rcClient);
	m_uiBlock->renderUI(rcClient, pDC);
}

void CBuildMFCDlg::OnLButtonDown(UINT nFlags, CPoint point)
{
	m_uiBlock->OnLButtonDown(nFlags, point);
}

void CBuildMFCDlg::OnLButtonUp(UINT nFlags, CPoint point)
{
	m_uiBlock->OnLButtonUp(nFlags, point);
}

void CBuildMFCDlg::OnMouseMove(UINT nFlags, CPoint point)
{
	m_uiBlock->OnMouseMove(nFlags, point);
}

void CBuildMFCDlg::OnLButtonDblClk(UINT nFlags, CPoint point)
{
}

UINT CBuildMFCDlg::ThreadHostMonitor(LPVOID pParam)
{
	m_the_dialog->m_uiBlock = new CUiBlock(COLOR_BYTES);
	startHostMonitor(calloc(1024 * 768, COLOR_BYTES), 1024, 768, COLOR_BYTES);
	return 0;
}

UINT CBuildMFCDlg::ThreadRefreshUI(LPVOID pParam)
{
	CDC* pDC = m_the_dialog->GetDC();
	while (true)
	{
		Sleep(30);
		m_the_dialog->updateUI(pDC);
	}
}

int sync_data(int hr, int spo2, int rr, int nibp_sys, int nibp_dia, int nibp_mean)
{
	return 0;
	wchar_t arguments[128];
	memset(arguments, 0, sizeof(arguments));
	swprintf(arguments, 128, L"/C sync_data.bat %d %d %d %d %d %d WinMFC",
		hr, spo2, rr, nibp_sys, nibp_dia, nibp_mean);
	HINSTANCE ret = ShellExecute(0, L"open", L"cmd.exe", arguments, 0, SW_HIDE);
	return 0;
}
