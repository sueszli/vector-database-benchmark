
// CCMSTBDTestDlg.cpp : 구현 파일
//

#include "stdafx.h"
#include "CCMSTBDTest.h"
#include "CCMSTBDTestDlg.h"

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


// CCCMSTBDTestDlg 대화 상자




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


// CCCMSTBDTestDlg 메시지 처리기

BOOL CCCMSTBDTestDlg::OnInitDialog()
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

	m_ctrlComboPort.SetCurSel(4);
	m_ctrlComboBaudRate.SetCurSel(3);

	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
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

// 대화 상자에 최소화 단추를 추가할 경우 아이콘을 그리려면
//  아래 코드가 필요합니다. 문서/뷰 모델을 사용하는 MFC 응용 프로그램의 경우에는
//  프레임워크에서 이 작업을 자동으로 수행합니다.

void CCCMSTBDTestDlg::OnPaint()
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
	// 수신된 데이터를 문자로 출력한다.
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
void CCCMSTBDTestDlg::OnBnClickedButtonDeviceManager()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
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
