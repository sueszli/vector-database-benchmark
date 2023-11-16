
// BOMListCheckerDlg.cpp : ���� ����
//

#include "stdafx.h"
#include "BOMListChecker.h"
#include "BOMListCheckerDlg.h"
#include "atlimage.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// ���� ���α׷� ������ ���Ǵ� CAboutDlg ��ȭ �����Դϴ�.

class CAboutDlg : public CDialog
{
public:
	CAboutDlg();

// ��ȭ ���� �������Դϴ�.
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV �����Դϴ�.

// �����Դϴ�.
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


// CBOMListCheckerDlg ��ȭ ����




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


// CBOMListCheckerDlg �޽��� ó����

BOOL CBOMListCheckerDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// �ý��� �޴��� "����..." �޴� �׸��� �߰��մϴ�.

	// IDM_ABOUTBOX�� �ý��� ��� ������ �־�� �մϴ�.
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

	// �� ��ȭ ������ �������� �����մϴ�. ���� ���α׷��� �� â�� ��ȭ ���ڰ� �ƴ� ��쿡��
	//  �����ӿ�ũ�� �� �۾��� �ڵ����� �����մϴ�.
	SetIcon(m_hIcon, TRUE);			// ū �������� �����մϴ�.
	SetIcon(m_hIcon, FALSE);		// ���� �������� �����մϴ�.

	// TODO: ���⿡ �߰� �ʱ�ȭ �۾��� �߰��մϴ�.

	HRESULT hResult = Image.Load(_T("E:\\igs_svn_data\\PROJECT_02\\03_SW\\06_BOMListChecker\\BOMListChecker\\res\\USM_FRONT.bmp"));
	if(FAILED(hResult))
	{
		AfxMessageBox(_T("Load Faild"));
		//return;
	}
	else
		m_imageloadF = 1;

	return TRUE;  // ��Ŀ���� ��Ʈ�ѿ� �������� ������ TRUE�� ��ȯ�մϴ�.
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

// ��ȭ ���ڿ� �ּ�ȭ ���߸� �߰��� ��� �������� �׸�����
//  �Ʒ� �ڵ尡 �ʿ��մϴ�. ����/�� ���� ����ϴ� MFC ���� ���α׷��� ��쿡��
//  �����ӿ�ũ���� �� �۾��� �ڵ����� �����մϴ�.

void CBOMListCheckerDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // �׸��⸦ ���� ����̽� ���ؽ�Ʈ

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Ŭ���̾�Ʈ �簢������ �������� ����� ����ϴ�.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// �������� �׸��ϴ�.
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

// ����ڰ� �ּ�ȭ�� â�� ���� ���ȿ� Ŀ���� ǥ�õǵ��� �ý��ۿ���
//  �� �Լ��� ȣ���մϴ�.
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
	// TODO: ���⿡ ��Ʈ�� �˸� ó���� �ڵ带 �߰��մϴ�.
	m_ctrlButtonTest.SetFaceColor(RGB(0, 255, 0), true);
}

void CBOMListCheckerDlg::OnBnClickedButtonTogreen()
{
	// TODO: ���⿡ ��Ʈ�� �˸� ó���� �ڵ带 �߰��մϴ�.
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
	// TODO: ���⿡ �޽��� ó���� �ڵ带 �߰� ��/�Ǵ� �⺻���� ȣ���մϴ�.



	CDialog::OnLButtonDblClk(nFlags, point);
}

void CBOMListCheckerDlg::OnNcLButtonDown(UINT nHitTest, CPoint point)
{
	// TODO: ���⿡ �޽��� ó���� �ڵ带 �߰� ��/�Ǵ� �⺻���� ȣ���մϴ�.

	CDialog::OnNcLButtonDown(nHitTest, point);
}

void CBOMListCheckerDlg::OnMouseMove(UINT nFlags, CPoint point)
{
	// TODO: ���⿡ �޽��� ó���� �ڵ带 �߰� ��/�Ǵ� �⺻���� ȣ���մϴ�.
	m_ptMouse = point;
	RedrawWindow();

	CDialog::OnMouseMove(nFlags, point);
}

void CBOMListCheckerDlg::OnMButtonUp(UINT nFlags, CPoint point)
{
	// TODO: ���⿡ �޽��� ó���� �ڵ带 �߰� ��/�Ǵ� �⺻���� ȣ���մϴ�.

	CDialog::OnMButtonUp(nFlags, point);
}

void CBOMListCheckerDlg::OnLButtonUp(UINT nFlags, CPoint point)
{
	// TODO: ���⿡ �޽��� ó���� �ڵ带 �߰� ��/�Ǵ� �⺻���� ȣ���մϴ�.
	/*RECT rect;
	int width, height;
	m_ctrlButton.GetWindowRect(&rect);
	width = rect.right - rect.left;
	height = rect.bottom - rect.top;
	m_ctrlButton.MoveWindow(point.x, point.y, width,height , true);*/
	m_ptMouse_up = point;

	
    CClientDC dc( this );
     
    // �� ���� 5�� ������ �Ǽ��� �׸���.
    CPen pen;
    pen.CreatePen( PS_SOLID, 5, RGB(255,0,0) );    // ������ �� ����
    CPen* oldPen = dc.SelectObject( &pen );
    dc.MoveTo( m_ptMouse_up.x, m_ptMouse_up.y );
	dc.Rectangle(m_ptMouse_down.x, m_ptMouse_down.y,  m_ptMouse_up.x,m_ptMouse_up.y);         // ���������� ���� �׸�
    dc.SelectObject( oldPen );
 
    // ���� ���������� �׸� �� �Ķ������� �׷��� �Ѵٸ�, �ٽ� ���ο� ���� ����� �׷���� �Ѵ�.
    // �� ���Ⱑ 10�� �Ķ��� �Ǽ��� �׸���.
    //pen.DeleteObject();        // ������ ���� �����Ѵ�.
    //pen.CreatePen( PS_SOLID, 10, RGB(0,0,255) );  // �Ķ��� �� ����
    //oldPen = dc.SelectObject( &pen );
    //dc.MoveTo( 30, 30 );
    //dc.LineTo( 40, 40 );
    //dc.SelectObject( oldPen );

	//Invalidate();




	CDialog::OnLButtonUp(nFlags, point);
}

void CBOMListCheckerDlg::OnLButtonDown(UINT nFlags, CPoint point)
{
	// TODO: ���⿡ �޽��� ó���� �ڵ带 �߰� ��/�Ǵ� �⺻���� ȣ���մϴ�.

	m_ptMouse_down = point;

	CDialog::OnLButtonDown(nFlags, point);
}
