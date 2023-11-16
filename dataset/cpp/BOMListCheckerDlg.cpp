
// BOMListCheckerDlg.cpp : 구현 파일
//

#include "stdafx.h"
#include "BOMListChecker.h"
#include "BOMListCheckerDlg.h"
#include "atlimage.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 응용 프로그램 정보에 사용되는 CAboutDlg 대화 상자입니다.

class CAboutDlg : public CDialog
{
public:
	CAboutDlg();

// 대화 상자 데이터입니다.
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

// 구현입니다.
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
END_MESSAGE_MAP()


// CBOMListCheckerDlg 대화 상자




CBOMListCheckerDlg::CBOMListCheckerDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CBOMListCheckerDlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
	m_imageloadF = 0;
}

void CBOMListCheckerDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_BUTTON_TEST, m_ctrlButtonTest);
	DDX_Control(pDX, IDC_BUTTON6, m_ctrlButton);
}

BEGIN_MESSAGE_MAP(CBOMListCheckerDlg, CDialog)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	//}}AFX_MSG_MAP
	ON_BN_CLICKED(IDC_BUTTON_TORED, &CBOMListCheckerDlg::OnBnClickedButtonTored)
	ON_BN_CLICKED(IDC_BUTTON_TOBLUE, &CBOMListCheckerDlg::OnBnClickedButtonToblue)
	ON_BN_CLICKED(IDC_BUTTON_TOGREEN, &CBOMListCheckerDlg::OnBnClickedButtonTogreen)
	ON_BN_CLICKED(IDC_BUTTON_IMG_LOAD, &CBOMListCheckerDlg::OnBnClickedButtonImgLoad)
	ON_WM_LBUTTONDBLCLK()
	ON_WM_NCLBUTTONDOWN()
	ON_WM_MOUSEMOVE()
	ON_WM_MBUTTONUP()
	ON_WM_LBUTTONUP()
	ON_WM_LBUTTONDOWN()
END_MESSAGE_MAP()


// CBOMListCheckerDlg 메시지 처리기

BOOL CBOMListCheckerDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// 시스템 메뉴에 "정보..." 메뉴 항목을 추가합니다.

	// IDM_ABOUTBOX는 시스템 명령 범위에 있어야 합니다.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
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

	// 이 대화 상자의 아이콘을 설정합니다. 응용 프로그램의 주 창이 대화 상자가 아닐 경우에는
	//  프레임워크가 이 작업을 자동으로 수행합니다.
	SetIcon(m_hIcon, TRUE);			// 큰 아이콘을 설정합니다.
	SetIcon(m_hIcon, FALSE);		// 작은 아이콘을 설정합니다.

	// TODO: 여기에 추가 초기화 작업을 추가합니다.

	HRESULT hResult = Image.Load(_T("E:\\igs_svn_data\\PROJECT_02\\03_SW\\06_BOMListChecker\\BOMListChecker\\res\\USM_FRONT.bmp"));
	if(FAILED(hResult))
	{
		AfxMessageBox(_T("Load Faild"));
		//return;
	}
	else
		m_imageloadF = 1;

	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
}

void CBOMListCheckerDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialog::OnSysCommand(nID, lParam);
	}
}

// 대화 상자에 최소화 단추를 추가할 경우 아이콘을 그리려면
//  아래 코드가 필요합니다. 문서/뷰 모델을 사용하는 MFC 응용 프로그램의 경우에는
//  프레임워크에서 이 작업을 자동으로 수행합니다.

void CBOMListCheckerDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 그리기를 위한 디바이스 컨텍스트

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 클라이언트 사각형에서 아이콘을 가운데에 맞춥니다.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 아이콘을 그립니다.
		dc.DrawIcon(x, y, m_hIcon);

		if(m_imageloadF)
		{
			Image.BitBlt(dc.m_hDC, 0, 0);
		}
	}
	else
	{
		CPaintDC dc(this);
		if(m_imageloadF)
		{
			//CPaintDC dc(this);
			Image.BitBlt(dc.m_hDC, 0, 0);
		}

		CString strTmp = _T("");
		strTmp.Format(_T("%03d, %03d"), m_ptMouse.x, m_ptMouse.y);
		dc.TextOutW(10, 10, strTmp);

		


		CDialog::OnPaint();
	}
}

// 사용자가 최소화된 창을 끄는 동안에 커서가 표시되도록 시스템에서
//  이 함수를 호출합니다.
HCURSOR CBOMListCheckerDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


void CBOMListCheckerDlg::OnBnClickedButtonTored()
{
	m_ctrlButtonTest.SetFaceColor(RGB(255, 0, 0), true);
}

void CBOMListCheckerDlg::OnBnClickedButtonToblue()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	m_ctrlButtonTest.SetFaceColor(RGB(0, 255, 0), true);
}

void CBOMListCheckerDlg::OnBnClickedButtonTogreen()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	m_ctrlButtonTest.SetFaceColor(RGB(0, 0, 255), true);
	//CImage Image;
}

void CBOMListCheckerDlg::OnBnClickedButtonImgLoad()
{
	//char szFilter[] = "Image (*.BMP, *.GIF, *.JPG) | *.BMP;*.GIF;*.JPG | All Files(*.*)|*.*||";
	
	CString strPathName;
	CString szFilter = _T("Image (*.BMP, *.GIF, *.JPG) | *.BMP;*.GIF;*.JPG | All Files(*.*)|*.*||");
	CFileDialog dlg(TRUE, NULL, NULL, OFN_HIDEREADONLY, szFilter);
	if(IDOK == dlg.DoModal()) {
		strPathName = dlg.GetPathName(); 
		
	}

	HRESULT hResult = Image.Load(strPathName);
	if(FAILED(hResult))
	{
		AfxMessageBox(_T("Load Faild"));
		return;
	}

	m_imageloadF = 1;
	//UpdateWindow();
	Invalidate();

}

void CBOMListCheckerDlg::OnLButtonDblClk(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.



	CDialog::OnLButtonDblClk(nFlags, point);
}

void CBOMListCheckerDlg::OnNcLButtonDown(UINT nHitTest, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	CDialog::OnNcLButtonDown(nHitTest, point);
}

void CBOMListCheckerDlg::OnMouseMove(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.
	m_ptMouse = point;
	RedrawWindow();

	CDialog::OnMouseMove(nFlags, point);
}

void CBOMListCheckerDlg::OnMButtonUp(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	CDialog::OnMButtonUp(nFlags, point);
}

void CBOMListCheckerDlg::OnLButtonUp(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.
	/*RECT rect;
	int width, height;
	m_ctrlButton.GetWindowRect(&rect);
	width = rect.right - rect.left;
	height = rect.bottom - rect.top;
	m_ctrlButton.MoveWindow(point.x, point.y, width,height , true);*/
	m_ptMouse_up = point;

	
    CClientDC dc( this );
     
    // 펜 굵기 5의 빨간색 실선을 그린다.
    CPen pen;
    pen.CreatePen( PS_SOLID, 5, RGB(255,0,0) );    // 빨간색 펜 생성
    CPen* oldPen = dc.SelectObject( &pen );
    dc.MoveTo( m_ptMouse_up.x, m_ptMouse_up.y );
	dc.Rectangle(m_ptMouse_down.x, m_ptMouse_down.y,  m_ptMouse_up.x,m_ptMouse_up.y);         // 빨간색으로 선을 그림
    dc.SelectObject( oldPen );
 
    // 만약 빨간색으로 그린 후 파란색으로 그려야 한다면, 다시 새로운 펜을 만들고 그려줘야 한다.
    // 펜 굵기가 10인 파란색 실선을 그림다.
    //pen.DeleteObject();        // 빨간색 펜을 삭제한다.
    //pen.CreatePen( PS_SOLID, 10, RGB(0,0,255) );  // 파란색 펜 생성
    //oldPen = dc.SelectObject( &pen );
    //dc.MoveTo( 30, 30 );
    //dc.LineTo( 40, 40 );
    //dc.SelectObject( oldPen );

	//Invalidate();




	CDialog::OnLButtonUp(nFlags, point);
}

void CBOMListCheckerDlg::OnLButtonDown(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	m_ptMouse_down = point;

	CDialog::OnLButtonDown(nFlags, point);
}
