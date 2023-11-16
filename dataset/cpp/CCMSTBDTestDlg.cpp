
// CCMSTBDTestDlg.cpp : ���� ����
//

#include "stdafx.h"
#include "CCMSTBDTest.h"
#include "CCMSTBDTestDlg.h"

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


// CCCMSTBDTestDlg ��ȭ ����




CCCMSTBDTestDlg::CCCMSTBDTestDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CCCMSTBDTestDlg::IDD, pParent)
	, m_editSTX(_T("CCMS"))
	, m_editTYPE(0x01)
	, m_editLENGTH(0x0001)
	, m_editDATA(0x01)
	, m_editCHECKSUM(0xff)
	, m_editETX(_T("CCME"))
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CCCMSTBDTestDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_COMBO_PORT, m_ctrlComboPort);
	DDX_Control(pDX, IDC_COMBO_BAUDRATE, m_ctrlComboBaudRate);
	DDX_Text(pDX, IDC_EDIT_STX, m_editSTX);
	DDX_Text(pDX, IDC_EDIT_TYPE, m_editTYPE);
	DDX_Text(pDX, IDC_EDIT_LENGTH, m_editLENGTH);
	DDX_Text(pDX, IDC_EDIT_DATA, m_editDATA);
	DDX_Text(pDX, IDC_EDIT_CHECK, m_editCHECKSUM);
	DDX_Text(pDX, IDC_EDIT_ETX, m_editETX);
}

BEGIN_MESSAGE_MAP(CCCMSTBDTestDlg, CDialog)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	//}}AFX_MSG_MAP
	ON_MESSAGE(WM_RECEIVEDATA, OnReceiveData)
	ON_BN_CLICKED(IDC_BUTTON_DEVICE_MANAGER, &CCCMSTBDTestDlg::OnBnClickedButtonDeviceManager)
	ON_BN_CLICKED(IDC_BUTTON_COMPORT_OPEN, &CCCMSTBDTestDlg::OnBnClickedButtonComportOpen)
	ON_BN_CLICKED(IDC_BUTTON_COMPORT_CLOSE, &CCCMSTBDTestDlg::OnBnClickedButtonComportClose)
	ON_BN_CLICKED(IDC_BUTTON_SEND, &CCCMSTBDTestDlg::OnBnClickedButtonSend)
END_MESSAGE_MAP()


// CCCMSTBDTestDlg �޽��� ó����

BOOL CCCMSTBDTestDlg::OnInitDialog()
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

	m_ctrlComboPort.SetCurSel(4);
	m_ctrlComboBaudRate.SetCurSel(3);

	return TRUE;  // ��Ŀ���� ��Ʈ�ѿ� �������� ������ TRUE�� ��ȯ�մϴ�.
}

void CCCMSTBDTestDlg::OnSysCommand(UINT nID, LPARAM lParam)
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

void CCCMSTBDTestDlg::OnPaint()
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
HCURSOR CCCMSTBDTestDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

LONG CCCMSTBDTestDlg::OnReceiveData(UINT WParam, LONG a)
{
	int i;
	int nLen = WParam;
	int num = 1;
	CString str;
	static char sData[4096] = {0,};
	char pData[4096] = {0, };
	int dataLength = 0;
	int low=0, high=0;
	static int index = 0;

	//UpdateData(TRUE);
	BYTE ch;
	int temp;
	// ���ŵ� �����͸� ���ڷ� ����Ѵ�.
	for (i=0; i<nLen; i++)
	{
		ch=m_Comm.abIn[i];
		str.Format("%02x", ch);
		//m_editReport += str;
		sData[index] = ch;
		index++;
	}

	//int idxSTX = 0;
	//int stxCnt = 0;
	//int stxOffset = 0;

	//int scmID, pType, pLength;
	//unsigned short pCRC=0, pCRC_high=0, pCRC_low=0, dCRC;

	//if(index>8){
	//	while(1){
	//		if(sData[idxSTX] == (m_editSCMID & 0xff)){
	//			if((sData[idxSTX-1] == 0x00) & (sData[idxSTX-2] == 0x00) & (sData[idxSTX-3] == 0x00)){
	//				stxCnt = idxSTX;
	//				break;
	//			}
	//			else{
	//				index = 0;
	//				return 0;
	//			}
	//				

	//		}
	//		else{
	//			idxSTX++;
	//		}
	//		if(idxSTX > index)
	//		{
	//			SetDlgItemText(IDC_STATIC_STATUS_MESSAGE, _T("STX Error"));
	//			index = 0;
	//			return 0;
	//		}
	//	}
	//	scmID = sData[idxSTX++];
	//	pType = sData[idxSTX++];
	//	pLength = (sData[idxSTX]<<8) | sData[idxSTX+1];
	//	idxSTX++;idxSTX++;

	//	if(stxCnt+4+pLength+2 > index) return 0;
	//	if(pLength > 0)
	//	{
	//		memcpy(pData, &sData[idxSTX], sizeof(char) * pLength);
	//		idxSTX +=pLength; 

	//		pCRC = (sData[idxSTX]&0x00ff) << 8 | sData[idxSTX+1]&0x00ff;
	//		dCRC = crcsum((unsigned char *)(sData+stxCnt), pLength+4, CRC_INIT);
	//		if(pCRC != dCRC)
	//		{
	//			SetDlgItemText(IDC_STATIC_STATUS_MESSAGE, _T("CRC Error"));
	//			index = 0;
	//			return 0;
	//		}
	//		else{
	//			SetDlgItemText(IDC_STATIC_STATUS_MESSAGE, _T("ACK SUCCESS"));
	//		}

	//		int tmp_high, tmp_low;
	//		CString sHigh_Low;

	//		m_editPLCReturnVal.Empty();
	//		
	//		for(int i=1; i<pLength; i++)
	//		{
	//			tmp_high = (pData[i] & 0xF0)>>4;
	//			tmp_low = (pData[i] & 0x0F);
	//			sHigh_Low.Format(_T("%02x %02x "), tmp_high, tmp_low);
	//			m_editPLCReturnVal += sHigh_Low;
	//		}

	//		UpdateData(false);


	//	}
	//	else if( pLength == 0)
	//	{
	//		pCRC = (sData[idxSTX]&0x00ff) << 8 | sData[idxSTX+1]&0x00ff;
	//		dCRC = crcsum((unsigned char *)(sData+stxCnt), pLength+4, CRC_INIT);
	//		if(pCRC != dCRC)
	//		{
	//			SetDlgItemText(IDC_STATIC_STATUS_MESSAGE, _T("CRC Error"));
	//			index = 0;
	//			return 0;
	//		}
	//		else{
	//			SetDlgItemText(IDC_STATIC_STATUS_MESSAGE, _T("ACK SUCCESS"));
	//		}
	//	}
	//	else return 0;


	//	index = 0;

	//}
	
	//m_editReport += _T("\r\n");
	//UpdateData(false);
	return 0;

}

TTYSTRUCT CCCMSTBDTestDlg::Int2TTY()
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
void CCCMSTBDTestDlg::OnBnClickedButtonDeviceManager()
{
	// TODO: ���⿡ ��Ʈ�� �˸� ó���� �ڵ带 �߰��մϴ�.
	system("hdwwiz.cpl");
}

void CCCMSTBDTestDlg::OnBnClickedButtonComportOpen()
{
	UpdateData(TRUE);

	m_nComPort = m_ctrlComboPort.GetCurSel();
	m_nBaudRate = m_ctrlComboBaudRate.GetCurSel();

	if(m_Comm.OpenCommPort(&Int2TTY()) != TRUE)
	{
		SetDlgItemText(IDC_STATIC_STATUS_MESSAGE, _T("COM Port Open Fail."));
		UpdateData(FALSE);

		return;
	}
	SetDlgItemText(IDC_STATIC_STATUS_MESSAGE, _T("COM Port Open Success."));
	m_isComPortOpen = 1;
	UpdateData(FALSE);
}

void CCCMSTBDTestDlg::OnBnClickedButtonComportClose()
{
	m_Comm.CloseConnection();
	SetDlgItemText(IDC_STATIC_STATUS_MESSAGE, _T("COM Port Closed."));
	m_isComPortOpen = 0;
}

void CCCMSTBDTestDlg::OnBnClickedButtonSend()
{
	UpdateData(TRUE);

	txPacket packet;
	BYTE sendBuf[128] = {0,};

	//STX
	sendBuf[0] = m_editSTX.GetAt(0);
	sendBuf[1] = m_editSTX.GetAt(1);
	sendBuf[2] = m_editSTX.GetAt(2);
	sendBuf[3] = m_editSTX.GetAt(3);

	//TYPE
	sendBuf[4] = m_editTYPE;

	//LENGTH
	sendBuf[5] = (m_editLENGTH>>8) & 0xFF;
	sendBuf[6] = (m_editLENGTH)    & 0xFF;

	//DATA
	sendBuf[7] = m_editDATA;

	//CHECK SUM
	m_editCHECKSUM = getCheckSum(sendBuf, 8);
	sendBuf[8] = m_editCHECKSUM;

	//ETX
	sendBuf[9] = m_editETX.GetAt(0);
	sendBuf[10] = m_editETX.GetAt(1);
	sendBuf[11] = m_editETX.GetAt(2);
	sendBuf[12] = m_editETX.GetAt(3);

	m_Comm.WriteCommBlock((char*)sendBuf,13);

	UpdateData(FALSE);

}

BYTE CCCMSTBDTestDlg::getCheckSum(BYTE * data, int length)
{
	BYTE csum;

	csum = 0;
	for(;length>0;length--)
	{
		csum += *data++;
	}

	return 0xFF - csum;
}
