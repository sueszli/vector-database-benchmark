
// FWDownLoaderDlg.cpp : 구현 파일
//

#include "stdafx.h"
#include "FWDownLoader.h"
#include "FWDownLoaderDlg.h"

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


// CFWDownLoaderDlg 대화 상자




CFWDownLoaderDlg::CFWDownLoaderDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CFWDownLoaderDlg::IDD, pParent)
	, m_editReport(_T(""))
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CFWDownLoaderDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_COMBO_PORT, m_ctrlComboPort);
	DDX_Control(pDX, IDC_COMBO_BAUDRATE, m_ctrlComboBaudRate);
	DDX_Text(pDX, IDC_RICHEDIT2_REPORT, m_editReport);
}

BEGIN_MESSAGE_MAP(CFWDownLoaderDlg, CDialog)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_MESSAGE(WM_RECEIVEDATA, OnReceiveData)	// CComm 클래스에서 보내는 메시지를 받아서 처리하기 위한 응용 함수 선언
	//}}AFX_MSG_MAP
	ON_BN_CLICKED(IDC_BUTTON_OPEN_DEVMGR, &CFWDownLoaderDlg::OnBnClickedButtonOpenDevmgr)
	ON_BN_CLICKED(IDC_BUTTON_COM_OPEN, &CFWDownLoaderDlg::OnBnClickedButtonComOpen)
	ON_BN_CLICKED(IDC_BUTTON_COM_CLOSE, &CFWDownLoaderDlg::OnBnClickedButtonComClose)
	ON_BN_CLICKED(IDC_BUTTON_COM_TEST, &CFWDownLoaderDlg::OnBnClickedButtonComTest)
	ON_BN_CLICKED(IDC_BUTTON_AUTOSYNCH, &CFWDownLoaderDlg::OnBnClickedButtonAutosynch)
	ON_BN_CLICKED(IDC_BUTTON_GET_STATUS, &CFWDownLoaderDlg::OnBnClickedButtonGetStatus)
	ON_BN_CLICKED(IDC_BUTTON_SET_CCFG, &CFWDownLoaderDlg::OnBnClickedButtonSectorProtectOfCcfg)
	ON_BN_CLICKED(IDC_BUTTON_BACKDOOR_EN, &CFWDownLoaderDlg::OnBnClickedButtonBackdoorEn)
	ON_BN_CLICKED(IDC_BUTTON_BACKDOOR_PIN, &CFWDownLoaderDlg::OnBnClickedButtonBackdoorPin)
	ON_BN_CLICKED(IDC_BUTTON_BACKDOOR_LEVEL, &CFWDownLoaderDlg::OnBnClickedButtonBackdoorLevel)
END_MESSAGE_MAP()


// CFWDownLoaderDlg 메시지 처리기

BOOL CFWDownLoaderDlg::OnInitDialog()
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


	m_Comm.SetHwnd(this->m_hWnd); // CComm 클래스에서 유저 메시지를 사용할 수 있도록 window 핸들을 전달한다.

	m_nComPort = 0; // COM 1
	m_nParity = 0; // no parity bit
	m_nBaudRate = 0; // 1200 baud rates
	m_nDataBits = 1; //8bits
	m_nStopBits = 0; //one stop ibt
	m_nFlowControl = 0; // No control

	//Comport 초기화
	CString portName, boardId;
	for(int i=1; i<30; i++){
		portName.Format(_T("COM%d"),i);
		m_ctrlComboPort.AddString(portName);
	}

	m_ctrlComboPort.SetCurSel(2);
	m_ctrlComboBaudRate.SetCurSel(5);


	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
}

void CFWDownLoaderDlg::OnSysCommand(UINT nID, LPARAM lParam)
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

void CFWDownLoaderDlg::OnPaint()
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
	}
	else
	{
		CDialog::OnPaint();
	}
}

// 사용자가 최소화된 창을 끄는 동안에 커서가 표시되도록 시스템에서
//  이 함수를 호출합니다.
HCURSOR CFWDownLoaderDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

LONG CFWDownLoaderDlg::OnReceiveData(UINT WParam, LONG a)
{
	int i;
	int nLen = WParam;
	int num = 1;
	CString str;
	static char sData[4096] = {0,};
	int dataLength = 0;
	int low=0, high=0;
	static int index = 0;

	UpdateData(TRUE);
	BYTE ch;
	int temp;
	// 수신된 데이터를 문자로 출력한다.
	for (i=0; i<nLen; i++)
	{
		ch=m_Comm.abIn[i];
		str.Format("%02x", ch);
		m_editReport += str;
		sData[index] = ch;
		index++;
	}

	
	m_editReport += _T("\r\n");
	UpdateData(false);
	return 0;

}
void CFWDownLoaderDlg::OnBnClickedButtonOpenDevmgr()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	//WinExec(("C:/Windows/System32/hdwwiz.cpl"), SW_SHOW);

	system("hdwwiz.cpl");

	//ShellExecute(NULL, _T("open"), _T("C:/Windows/System32/hdwwiz.cpl"), NULL, NULL, SW_SHOW);
}

void CFWDownLoaderDlg::OnBnClickedButtonComOpen()
{
	UpdateData(TRUE);

	m_nComPort = m_ctrlComboPort.GetCurSel();
	m_nBaudRate = m_ctrlComboBaudRate.GetCurSel();

	if(m_Comm.OpenCommPort(&Int2TTY()) != TRUE)
	{
		// COM port 열기가 실패하면 에러 윈도우를 보여준다.
		m_editReport.Format("COM Port %d : Port Open Fail !!\r\n",m_nComPort+1 );
		UpdateData(FALSE);


		//((CButton*)GetDlgItem(IDC_CommStart))->SetCheck(!bCheck);
		return;
	}

	//GetDlgItem(IDC_CommStart)->SetWindowText("Close");

	//GetDlgItem(IDC_ComPort)->EnableWindow(FALSE);
	//GetDlgItem(IDC_BaudRate)->EnableWindow(FALSE);
	m_editReport.Format("COM Port %d : Port Open Success !!\r\n",m_nComPort+1 );
	UpdateData(FALSE);
}

TTYSTRUCT CFWDownLoaderDlg::Int2TTY()
{
	//직접 코딩할 부분(3)
	//COM port 를 열기위한 파라미터 값 설정
	TTYSTRUCT tty;
	ZERO_MEMORY(tty);

	tty.byCommPort = (BYTE)m_nComPort + 1;
	tty.byXonXoff = FALSE;
	tty.byByteSize = (BYTE)_nDataValues[m_nDataBits];
	tty.byFlowCtrl = (BYTE)_nFlow[m_nFlowControl];
	tty.byParity = (BYTE)m_nParity;
	tty.byStopBits = (BYTE)_nStopBits[m_nStopBits];
	tty.dwBaudRate = (DWORD)_nBaudRates[m_nBaudRate];

	return tty;
}


void CFWDownLoaderDlg::OnBnClickedButtonComClose()
{
	m_Comm.CloseConnection();
	m_editReport.Format("COM Port %d : Port close Success !!\r\n",m_nComPort+1 );
	UpdateData(FALSE);
}

void CFWDownLoaderDlg::OnBnClickedButtonComTest()
{
	unsigned char ucCommand[3];
	ucCommand[0] = 0x03;
	ucCommand[1] = 0x20;
	ucCommand[2] = 0x20;

	CString str;
	m_editReport += _T("\r\nTransmitData : ");
	for(int i=0; i<3; i++)
	{
		str.Format("%02x", ucCommand[i]);
		m_editReport += str;

	}
	
	m_Comm.WriteCommBlock((char*)ucCommand,3);
	//m_Comm.WriteCommBlock((char*)(ucCommand+1),1);
	//m_Comm.WriteCommBlock((char*)(ucCommand+2),1);
	m_editReport += _T("\r\nReceivedData : ");
	UpdateData(FALSE);
}

void CFWDownLoaderDlg::OnBnClickedButtonAutosynch()
{
	unsigned char ucCommand[2];
	ucCommand[0] = 0x55;
	ucCommand[1] = 0x55;
	

	CString str;
	m_editReport += _T("\r\nTransmitData : ");
	for(int i=0; i<2; i++)
	{
		str.Format("%02x", ucCommand[i]);
		m_editReport += str;

	}
	
	m_Comm.WriteCommBlock((char*)ucCommand,2);
	m_editReport += _T("\r\nReceivedData : ");
	UpdateData(FALSE);
}

void CFWDownLoaderDlg::OnBnClickedButtonGetStatus()
{
	unsigned char ucCommand[3];
	ucCommand[0] = 0x03;
	ucCommand[1] = 0x23;
	ucCommand[2] = 0x23;

	CString str;
	m_editReport += _T("\r\nTransmitData : ");
	for(int i=0; i<3; i++)
	{
		str.Format("%02x", ucCommand[i]);
		m_editReport += str;

	}
	
	m_Comm.WriteCommBlock((char*)ucCommand,1);
	m_Comm.WriteCommBlock((char*)(ucCommand+1),1);
	m_Comm.WriteCommBlock((char*)(ucCommand+2),1);
	m_editReport += _T("\r\nReceivedData : ");
	UpdateData(FALSE);
}

void CFWDownLoaderDlg::OnBnClickedButtonSectorProtectOfCcfg()
{
	unsigned char ucCommand[11];
	ucCommand[0] = 0x0B;

	ucCommand[1] = 0x00;
	ucCommand[2] = 0x2D;

	ucCommand[3] = 0x00;
	ucCommand[4] = 0x00;
	ucCommand[5] = 0x00;
	ucCommand[6] = 0x00;

	ucCommand[7] = 0x80;
	ucCommand[8] = 0x00;
	ucCommand[9] = 0x00;
	ucCommand[10] = 0x00;

	for(int i=2; i<11; i++)
	{
		ucCommand[1] += ucCommand[i];
	}
	

	CString str;
	m_editReport += _T("\r\nTransmitData : ");
	for(int i=0; i<11; i++)
	{
		str.Format("%02x", ucCommand[i]);
		m_editReport += str;

	}
	
	m_Comm.WriteCommBlock((char*)ucCommand,11);
	m_editReport += _T("\r\nReceivedData : ");
	UpdateData(FALSE);
}

void CFWDownLoaderDlg::OnBnClickedButtonBackdoorEn()
{
	unsigned char ucCommand[11];
	ucCommand[0] = 0x0B;

	ucCommand[1] = 0x00;
	ucCommand[2] = 0x2D;

	ucCommand[3] = 0x00;
	ucCommand[4] = 0x00;
	ucCommand[5] = 0x00;
	ucCommand[6] = 0x0B;

	ucCommand[7] = 0x00;
	ucCommand[8] = 0x00;
	ucCommand[9] = 0x00;
	ucCommand[10] = 0xC5;

	for(int i=2; i<11; i++)
	{
		ucCommand[1] += ucCommand[i];
	}
	

	CString str;
	m_editReport += _T("\r\nTransmitData : ");
	for(int i=0; i<11; i++)
	{
		str.Format("%02x", ucCommand[i]);
		m_editReport += str;

	}
	
	m_Comm.WriteCommBlock((char*)ucCommand,11);
	m_editReport += _T("\r\nReceivedData : ");
	UpdateData(FALSE);
}

void CFWDownLoaderDlg::OnBnClickedButtonBackdoorPin()
{
	unsigned char ucCommand[11];
	ucCommand[0] = 0x0B;

	ucCommand[1] = 0x00;
	ucCommand[2] = 0x2D;

	ucCommand[3] = 0x00;
	ucCommand[4] = 0x00;
	ucCommand[5] = 0x00;
	ucCommand[6] = 0x0C;

	ucCommand[7] = 0x00;
	ucCommand[8] = 0x00;
	ucCommand[9] = 0x00;
	ucCommand[10] = 0x0A;

	for(int i=2; i<11; i++)
	{
		ucCommand[1] += ucCommand[i];
	}
	

	CString str;
	m_editReport += _T("\r\nTransmitData : ");
	for(int i=0; i<11; i++)
	{
		str.Format("%02x", ucCommand[i]);
		m_editReport += str;

	}
	
	m_Comm.WriteCommBlock((char*)ucCommand,11);
	m_editReport += _T("\r\nReceivedData : ");
	UpdateData(FALSE);
}

void CFWDownLoaderDlg::OnBnClickedButtonBackdoorLevel()
{
	unsigned char ucCommand[11];
	ucCommand[0] = 0x0B;

	ucCommand[1] = 0x00;
	ucCommand[2] = 0x2D;

	ucCommand[3] = 0x00;
	ucCommand[4] = 0x00;
	ucCommand[5] = 0x00;
	ucCommand[6] = 0x0D;

	ucCommand[7] = 0x00;
	ucCommand[8] = 0x00;
	ucCommand[9] = 0x00;
	ucCommand[10] = 0x01;

	for(int i=2; i<11; i++)
	{
		ucCommand[1] += ucCommand[i];
	}
	

	CString str;
	m_editReport += _T("\r\nTransmitData : ");
	for(int i=0; i<11; i++)
	{
		str.Format("%02x", ucCommand[i]);
		m_editReport += str;

	}
	
	m_Comm.WriteCommBlock((char*)ucCommand,11);
	m_editReport += _T("\r\nReceivedData : ");
	UpdateData(FALSE);
}
