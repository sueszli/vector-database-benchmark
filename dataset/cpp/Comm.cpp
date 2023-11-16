//Comm.cpp Rs232c����� �ϱ� ���� Ŭ����
// 2001�� 3�� 26�� (��) ����ũ�� �κ�Ʈ  S/W�� ������
//
#include "stdafx.h"
#include "comm.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

IMPLEMENT_DYNCREATE(CComm, CObject)

// ����������� Table
BYTE _nFlow[] = {FC_NONE,FC_DTRDSR,FC_RTSCTS,FC_XONXOFF};
// ��� ����Ÿ ������ ���̺�
int _nDataValues[] = {7,8};
// ��żӵ� Table

int _nBaudRates[] = {	CBR_1200,	CBR_2400,	CBR_4800,
						CBR_9600,	CBR_14400,	CBR_19200,
						CBR_38400,	CBR_56000,	CBR_57600, 
						CBR_115200, 921600};
BYTE _nStopBits[] = { ONESTOPBIT, TWOSTOPBITS };
//BOOL	bTxEmpty=TRUE;

UINT ReceMessage;

CComm::CComm( )
{
	idComDev=NULL;
	fConnected=FALSE;	
	bTxEmpty=TRUE;
}

CComm::~CComm( )
{
    DestroyComm();
}

//BEGIN_MESSAGE_MAP(CComm, CObject)
	//{{AFX_MSG_MAP(CComm)
		// NOTE - the ClassWizard will add and remove mapping macros here.
	//}}AFX_MSG_MAP
//END_MESSAGE_MAP()


/////////////////////////////////////////////////////////////////////////////
// CComm message handlers
//CommWatchProc()
//����� �ϴ� ���μ��� �� ����Ÿ�� �������� �����ϴ�
//��ƾ ����ƾ�� OpenComPort �Լ� ����� ���ν����� �����
//OpenComPort �Լ� ����
DWORD CommWatchProc(LPVOID lpData)
{
	DWORD       dwEvtMask ;
	OVERLAPPED  os ;
	CComm*      npComm = (CComm*) lpData ;
	char InData[COM_MAXBLOCK + 1];
	int        nLength ;
	//idCommDev ��� �ڵ鿡 �ƹ��� com ��Ʈ�� �Ⱥپ� ������
	// ���� ����
	if ( npComm == NULL || 
		!npComm->IsKindOf( RUNTIME_CLASS( CComm ) ) )
		return (DWORD)(-1);

	memset( &os, 0, sizeof( OVERLAPPED ) ) ;


	os.hEvent = CreateEvent( NULL,    // no security
                            TRUE,    // explicit reset req
                            FALSE,   // initial event reset
                            NULL ) ; // no name
	if ( os.hEvent == NULL )
	{
		MessageBox( NULL, "Failed to create event for thread!", "comm Error!",
					MB_ICONEXCLAMATION | MB_OK );
		return ( FALSE ) ;
	}

	DWORD dwEventFlags = EV_BREAK | EV_CTS | EV_DSR | EV_ERR | EV_RING | 
							EV_RLSD | EV_RXCHAR | EV_RXFLAG | EV_TXEMPTY;
	if (!SetCommMask(npComm->idComDev, dwEventFlags))
		return ( FALSE ) ;

	while (npComm->fConnected)
	{		
		dwEvtMask = 0;
		
		WaitCommEvent(npComm->idComDev, &dwEvtMask, NULL );
		
		if ((dwEvtMask & EV_BREAK) == EV_BREAK)
		{
			;;;
		}
		else if ((dwEvtMask & EV_CTS) == EV_CTS)
		{
			;;;
		}
		else if ((dwEvtMask & EV_ERR) == EV_ERR)
		{
			;;;
		}
		else if ((dwEvtMask & EV_RING) == EV_RING)
		{
			;;;
		}
		else if ((dwEvtMask & EV_RLSD) == EV_RLSD)
		{
			;;;
		}
		else if ((dwEvtMask & EV_RXCHAR) == EV_RXCHAR)
		{				
			do
			{			
				memset(InData,0,65536);
				if (nLength = npComm->ReadCommBlock((LPSTR) InData, COM_MAXBLOCK ))
				{
                  npComm->SetReadData(InData,nLength);
                  //�̰����� ����Ÿ�� �޴´�.
		      	}
			}
			while ( nLength > 0 ) ;
			
		}
		else if ((dwEvtMask & EV_RXFLAG) == EV_RXFLAG)
		{
			;;;
		}
		else if ((dwEvtMask & EV_TXEMPTY) == EV_TXEMPTY)
		{
			npComm->bTxEmpty = TRUE;
		}
	}
	
	CloseHandle( os.hEvent );

	return( TRUE );
} 

//����Ÿ�� �а� ����Ÿ�� �о��ٴ�
//�޼����� �����Ѵ�.
void CComm::SetReadData(LPSTR data, int nLen)
{	
	memset(abIn, 0, 65536); 
	if(nLen<0 || nLen > 65535)
		return;
	memcpy(abIn,data,nLen);	
	//������ �����쿡 WM_RECEIVEDATA�޼�����
	//�����־� ���� ����Ÿ�� ���Դٴ°���
	//�˷��ش�.
	
	//SendMessage(m_hwnd,ReceMessage,nLen,0);

	
	SendMessage(m_hwnd,WM_RECEIVEDATA,nLen,0);
}

//�޼����� ������ hwnd����
void CComm::SetHwnd(HWND hwnd)
{
	m_hwnd=hwnd;
}

//com ��Ʈ�� ���� ������ �õ��Ѵ�.
//OpenComport()

BOOL CComm::OpenCommPort(LPTTYSTRUCT lpTTY)
{            
	char       szPort[ 15 ] ;
	BOOL       fRetVal ;
	COMMTIMEOUTS  CommTimeOuts ;

	osWrite.Offset = 0 ;
	osWrite.OffsetHigh = 0 ;
	osRead.Offset = 0 ;
	osRead.OffsetHigh = 0 ;

	//�̺�Ʈ â�� ����
	osRead.hEvent = CreateEvent( NULL, TRUE, FALSE, NULL ); 
	if (osRead.hEvent == NULL)
	{
		return FALSE ;
	}	
	osWrite.hEvent = CreateEvent( NULL,   TRUE,  FALSE,   NULL );
	if (NULL == osWrite.hEvent)
	{
		CloseHandle( osRead.hEvent );
		return FALSE;
	}

   
	if (lpTTY->byCommPort > COM_MAXPORTS)
		lstrcpy( szPort, "\\\\.\\TELNET" ) ;
	else
		wsprintf( szPort, "COM%d", lpTTY->byCommPort );

	// COMM device�� ȭ���������� �����Ѵ�.

	if ((idComDev =
		CreateFile( szPort, GENERIC_READ | GENERIC_WRITE,
                  0,                    // exclusive access
                  NULL,                 // no security attrs
                  OPEN_EXISTING,
                  FILE_ATTRIBUTE_NORMAL | 
                  FILE_FLAG_OVERLAPPED, // overlapped I/O
                  NULL )) == (HANDLE) -1 )
		return ( FALSE ) ;
	else
	{
		//����Ʈ���� ����Ÿ�� ��ȯ�ϴ� ����� char������ �⺻���� ��������
		SetCommMask( idComDev, EV_RXCHAR ) ;
		SetupComm( idComDev, 4096, 4096 ) ;
		//����̽��� �����Ⱑ ������ �𸣴ϱ� ������ û�Ҹ� ����!
		PurgeComm( idComDev, PURGE_TXABORT | PURGE_RXABORT |
                                      PURGE_TXCLEAR | PURGE_RXCLEAR ) ;

     
		CommTimeOuts.ReadIntervalTimeout = 0xFFFFFFFF ;
		CommTimeOuts.ReadTotalTimeoutMultiplier = 0 ;
		CommTimeOuts.ReadTotalTimeoutConstant = 0 ;
		CommTimeOuts.WriteTotalTimeoutMultiplier = 0 ;
		CommTimeOuts.WriteTotalTimeoutConstant = 0 ;
		SetCommTimeouts( idComDev, &CommTimeOuts ) ;
	}

	fRetVal = SetupConnection(lpTTY) ;

	if (fRetVal)//������ �Ǿ��ٸ� fRetVal TRUE�̹Ƿ�
	{
		fConnected = TRUE ;//����Ǿ��ٰ� ������
		//���ν����� CommWatchProc�� �����ϴϱ� ���߿� ����Ÿ�� �Դٰ���
		//�ϸ� ��� ������ CommWatchProc�� ����Ѵ�.
		AfxBeginThread((AFX_THREADPROC)CommWatchProc,(LPVOID)this);
	}
	else
	{
		fConnected = FALSE ;
		CloseHandle( idComDev) ;
	}

	return ( fRetVal ) ;
} 

//ȭ�Ϸ� ������ ����Ʈ�� ���� ��Ʈ�� ������ ��Ų��.
//SetupConnection ������ CreateComPort�� ���־�� �Ѵ�.
BOOL CComm::SetupConnection(LPTTYSTRUCT lpTTY)
{	
	BOOL       fRetVal;	
	DCB        dcb;

	dcb.DCBlength = sizeof( DCB ) ;

	GetCommState( idComDev, &dcb ) ;//dcb�� �⺻���� �޴´�.


	dcb.BaudRate = lpTTY->dwBaudRate;
	dcb.ByteSize = lpTTY->byByteSize;
	dcb.Parity = lpTTY->byParity;
	dcb.StopBits = lpTTY->byStopBits;
	
	// setup hardware flow control FC_DTRDSR
	BOOL bSet = (BYTE)((lpTTY->byFlowCtrl & FC_DTRDSR) != 0);

	dcb.fOutxDsrFlow = bSet;
	
	if(bSet)
		dcb.fDtrControl = DTR_CONTROL_HANDSHAKE;
	else
		dcb.fDtrControl = DTR_CONTROL_ENABLE;
	
	// FC_RTSCTS
	bSet = (BYTE)((lpTTY->byFlowCtrl & FC_RTSCTS) != 0);
	dcb.fOutxCtsFlow = bSet;
	
	if(bSet)
		dcb.fRtsControl = RTS_CONTROL_HANDSHAKE;	
	else
		dcb.fRtsControl = RTS_CONTROL_ENABLE;
		
	// setup software flow control FC_XONXOFF
	bSet = (BYTE)((lpTTY->byFlowCtrl & FC_XONXOFF) != 0);

	dcb.fInX = dcb.fOutX = bSet;
	dcb.XonChar = ASCII_XON;
	dcb.XoffChar = ASCII_XOFF;
	dcb.XonLim = 1;
	dcb.XoffLim = 1;
	
	// other various settings
	dcb.fBinary = TRUE;
	dcb.fParity = TRUE;

	fRetVal = SetCommState( idComDev, &dcb ) ; //����� Dcb ����

	return ( fRetVal ) ;   
} 

//����Ʈ�� ���� ����Ÿ�� �д´�.
int CComm::ReadCommBlock(LPSTR lpszBlock, int nMaxLength )
{
	BOOL       fReadStat ;
	COMSTAT    ComStat ;
	DWORD      dwErrorFlags;
	DWORD      dwLength;

	// only try to read number of bytes in queue 
	ClearCommError( idComDev, &dwErrorFlags, &ComStat ) ;
	dwLength = min( (DWORD) nMaxLength, ComStat.cbInQue ) ;
	OutputDebugString("\n\rmyEom");

	if (dwLength > 0)
	{
		fReadStat = ReadFile( idComDev, lpszBlock, dwLength, &dwLength, &osRead ) ;
		if (!fReadStat)
		{
         	if(GetLastError() == ERROR_IO_PENDING)
			{
				OutputDebugString("\n\rIO Pending");
				// we have to wait for read to complete.  This function will timeout
				// according to the CommTimeOuts.ReadTotalTimeoutConstant variable
				// Every time it times out, check for port errors								
				while(!GetOverlappedResult(idComDev, &osRead,
					&dwLength, TRUE))
				{
					DWORD dwError = GetLastError();
					if(dwError == ERROR_IO_INCOMPLETE)
					{
						// normal result if not finished
						continue;
					}
					else
					{
						// CAN DISPLAY ERROR MESSAGE HERE
						OutputDebugString("\n\rmyEom_ERROR_01");
						// an error occured, try to recover
						::ClearCommError(idComDev, &dwErrorFlags, &ComStat);
						if(dwErrorFlags > 0)
						{
							// CAN DISPLAY ERROR MESSAGE HERE
							OutputDebugString("\n\rmyEom_ERROR_02");
						}
						break;
					}
				}
			}
			else
			{
				// some other error occured
				dwLength = 0;
				ClearCommError(idComDev, &dwErrorFlags, &ComStat);
				if(dwErrorFlags > 0)
				{
					// CAN DISPLAY ERROR MESSAGE HERE
					OutputDebugString("\n\rmyEom_ERROR_03");

				}
			}
		}
   }
   
   return ( dwLength ) ;
} 

// ����Ʈ�� �����͸� ����.
BOOL CComm::WriteCommBlock( LPSTR lpByte , DWORD dwBytesToWrite)
{
	DWORD		dwErrorFlags;
	BOOL        fWriteStat ;
	DWORD       dwBytesWritten ;
	COMSTAT ComStat;

	DWORD	dwLength = dwBytesToWrite;

	bTxEmpty = FALSE;
	fWriteStat = WriteFile( idComDev, lpByte, dwBytesToWrite, &dwBytesWritten, &osWrite ) ;
	//if (dwBytesToWrite != dwBytesWritten)
	
	if(!fWriteStat)
	{
		if(GetLastError() == ERROR_IO_PENDING)
		{
			OutputDebugString("\n\rIO Pending");
			// ���� ���ڰ� ���� �ְų� ������ ���ڰ� ���� ���� ��� Overapped IO��
			// Ư���� ���� ERROR_IO_PENDING ���� �޽����� ���޵ȴ�.
			//timeouts�� ������ �ð���ŭ ��ٷ��ش�.
			while(!GetOverlappedResult(idComDev, &osWrite,
						&dwLength, TRUE))
			{
				DWORD dwError = GetLastError();
				if(dwError == ERROR_IO_INCOMPLETE)
				{
					// normal result if not finished
					continue;
				}
				else
				{
					// CAN DISPLAY ERROR MESSAGE HERE
					// an error occured, try to recover
					::ClearCommError(idComDev, &dwErrorFlags, &ComStat);
					if(dwErrorFlags > 0)
					{
						// CAN DISPLAY ERROR MESSAGE HERE
						AfxMessageBox("ERROR");
					}
					break;
				}
			}
		}
		else
		{	// some other error occured
			dwLength = 0;
			ClearCommError(idComDev, &dwErrorFlags, &ComStat);
			if(dwErrorFlags > 0)
			{
				// CAN DISPLAY ERROR MESSAGE HERE				
				AfxMessageBox("ERROR");
			}
		}		
	}

	return ( TRUE ) ;
} 


//����Ʈ�� ������ �����Ѵ�.
BOOL CComm::DestroyComm()
{
	if (fConnected)
		CloseConnection();

	if (osRead.hEvent!=NULL)
		CloseHandle( osRead.hEvent ) ;

	if (osWrite.hEvent != NULL)
		CloseHandle( osWrite.hEvent ) ;

	return ( TRUE ) ;
} 

//������ �ݴ´�.
BOOL CComm::CloseConnection()
{

   // set connected flag to FALSE

   fConnected = FALSE ;

   // disable event notification and wait for thread
   // to halt

   SetCommMask( idComDev, 0 ) ;


   EscapeCommFunction( CLRDTR ) ;
   EscapeCommFunction( CLRRTS ) ;


   PurgeComm( idComDev, PURGE_TXABORT | PURGE_RXABORT |
                                   PURGE_TXCLEAR | PURGE_RXCLEAR ) ;
   CloseHandle( idComDev ) ;
 
   return ( TRUE ) ;

}

void CComm::EscapeCommFunction(DWORD dwFunc)
{
	// TODO: Add your control notification handler code here
	::EscapeCommFunction(idComDev,dwFunc);
}