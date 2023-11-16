
// FWDownLoaderDlg.cpp : ���� ����
//

#include "stdafx.h"
#include "FWDownLoader.h"
#include "FWDownLoaderDlg.h"

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


// CFWDownLoaderDlg ��ȭ ����




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
	ON_MESSAGE(WM_RECEIVEDATA, OnReceiveData)	// CComm Ŭ�������� ������ �޽����� �޾Ƽ� ó���ϱ� ���� ���� �Լ� ����
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


// CFWDownLoaderDlg �޽��� ó����

BOOL CFWDownLoaderDlg::OnInitDialog()
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


	m_Comm.SetHwnd(this->m_hWnd); // CComm Ŭ�������� ���� �޽����� ����� �� �ֵ��� window �ڵ��� �����Ѵ�.

	m_nComPort = 0; // COM 1
	m_nParity = 0; // no parity bit
	m_nBaudRate = 0; // 1200 baud rates
	m_nDataBits = 1; //8bits
	m_nStopBits = 0; //one stop ibt
	m_nFlowControl = 0; // No control

	//Comport �ʱ�ȭ
	CString portName, boardId;
	for(int i=1; i<30; i++){
		portName.Format(_T("COM%d"),i);
		m_ctrlComboPort.AddString(portName);
	}

	m_ctrlComboPort.SetCurSel(2);
	m_ctrlComboBaudRate.SetCurSel(5);


	return TRUE;  // ��Ŀ���� ��Ʈ�ѿ� �������� ������ TRUE�� ��ȯ�մϴ�.
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

// ��ȭ ���ڿ� �ּ�ȭ ���߸� �߰��� ��� �������� �׸�����
//  �Ʒ� �ڵ尡 �ʿ��մϴ�. ����/�� ���� ����ϴ� MFC ���� ���α׷��� ��쿡��
//  �����ӿ�ũ���� �� �۾��� �ڵ����� �����մϴ�.

void CFWDownLoaderDlg::OnPaint()
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
	}
	else
	{
		CDialog::OnPaint();
	}
}

// ����ڰ� �ּ�ȭ�� â�� ���� ���ȿ� Ŀ���� ǥ�õǵ��� �ý��ۿ���
//  �� �Լ��� ȣ���մϴ�.
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
	// ���ŵ� �����͸� ���ڷ� ����Ѵ�.
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
	// TODO: ���⿡ ��Ʈ�� �˸� ó���� �ڵ带 �߰��մϴ�.
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
		// COM port ���Ⱑ �����ϸ� ���� �����츦 �����ش�.
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
	//���� �ڵ��� �κ�(3)
	//COM port �� �������� �Ķ���� �� ����
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
