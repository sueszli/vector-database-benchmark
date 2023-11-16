#include "StdAfx.h"
#include "ExternalNetComm.h"

enum {
	IDX_SOCK_ERR_NO_ERROR = 1,
	IDX_SOCK_ERR_SOCK_INIT,
	IDX_SOCK_ERR_CREATE,
	IDX_SOCK_ERR_LISTEN,
};

#define SZ_BUF_NET_SND_PKT			(256 *1024)
#define SZ_BUF_NET_RES_PKT			(256 *1)

UINT TH_ExternalNetCommProc (LPVOID pParam)
{

	int i, idxErr;
	CExternalNetComm *pMan = (CExternalNetComm *)pParam;
	pMan->m_sListenSocket = new CAsyncSocket;
	pMan->m_sConnectSocket = new CAsyncSocket;
	pMan->m_pSockDummy = new CAsyncSocket;

	// Socket creation[S]
	if (pMan->m_sListenSocket->Create (pMan->m_iPort) == FALSE)
	{
		idxErr = IDX_SOCK_ERR_CREATE;
		goto END_OF_THREAD;
	}
	// Socket creation[E]

	// Socket listen[S]
	while (1)
	{
		if (pMan->m_sListenSocket->Listen () == TRUE)
		{
			break;
		}

		Sleep (10);
	}
	// Socket listen[E]

	pMan->m_bSockReady = TRUE;
	while (pMan->m_bEndThread == FALSE)
	{
		if (pMan->m_bConnected == FALSE)
		{
			if (pMan->m_sListenSocket->Accept (*pMan->m_sConnectSocket) == TRUE)
			{
				pMan->m_bConnected = TRUE;
			}
			else
			{
				//AfxMessageBox(_T("Disconnected"));
				//Sleep(1000);
				//break;
			}
			Sleep(100);
		}
		else
		{
			if (pMan->m_sListenSocket->Accept (*pMan->m_pSockDummy) == TRUE)
			{
				pMan->m_pSockDummy->Send("RECO", 100);
				pMan->m_pSockDummy->Close ();
				pMan->m_sConnectSocket->Close ();
				pMan->m_bConnected = FALSE;
				continue;

			}
			if (pMan->PktRcv () == FALSE)
			{
				//AfxMessageBox(_T("Disconnected"));
				Sleep(50);
			}
		}
	}
	

END_OF_THREAD:
	delete pMan->m_sListenSocket;
	delete pMan->m_sConnectSocket;

	pMan->m_bEndThread = FALSE;
	pMan->m_bSockReady = idxErr;
	return 0;
}

CExternalNetComm::CExternalNetComm(void)
{
	m_iPort = 4000;
	m_strMessage = _T("");
	m_bEndThread = TRUE;

	m_bufSnd = new BYTE[SZ_BUF_NET_SND_PKT];
	m_bufRcv = new BYTE[SZ_BUF_NET_RES_PKT];
}

CExternalNetComm::~CExternalNetComm(void)
{
	
	m_sConnectSocket->Close ();
	delete m_sListenSocket;
	delete m_sConnectSocket;
}

BOOL CExternalNetComm::InitExternalNetCommManager ()
{
	if (m_bEndThread == TRUE)
	{
		m_bEndThread = FALSE;
		m_bSockReady = FALSE;
		m_bConnected = FALSE;

		AfxBeginThread (TH_ExternalNetCommProc, this);
		while (m_bSockReady == FALSE)
		{
			Sleep (10);
		}

		if (m_bSockReady != TRUE)
		{
			m_bEndThread = TRUE;
			return FALSE;
		}
	}

	return TRUE;
}
void CExternalNetComm::DestroyExternalNetCommManager ()
{
	if (m_bEndThread == FALSE)
	{
		m_bEndThread = TRUE;
		while (m_bEndThread == TRUE)
		{
			Sleep (10);
		}
		m_bEndThread = TRUE;
		m_bSockReady = FALSE;
		m_bConnected = FALSE;
	}
}
BOOL CExternalNetComm::PktRcv ()
{

	int numRcv;
	INFO_BACK_DRAWING_ALL *pIBDA = &glInfoGlobal.iBDA;
	INFO_BACK_DRAWING_ITEM *pIBDI;

	pIBDI = &pIBDA->bufBDI[pIBDA->bufIdxMainBDI[0]];
	
	BYTE stPa2F[200]={'P','G','S','#','T','O','T','A','L','#','2','#',};
	BYTE stPa3F[200]={'P','G','S','#','T','O','T','A','L','#','3','#',};

	numRcv = m_sConnectSocket->Receive (m_bufRcv, SZ_BUF_NET_RES_PKT);

	if (numRcv == 0)	// The connection has been closed.
	{
		return FALSE;
	}
	else if (numRcv == SOCKET_ERROR)
	{
		return FALSE;
	}
	else
	{
		if(m_bufRcv[0]=='P'&&m_bufRcv[1]=='A'&&m_bufRcv[2]=='S'&&m_bufRcv[3]=='M'&&
			m_bufRcv[4]=='#'&&m_bufRcv[5]=='T'&&m_bufRcv[6]=='O'&&m_bufRcv[7]=='T'&&
			m_bufRcv[8]=='A'&&m_bufRcv[9]=='L'&&m_bufRcv[10]=='#')
		{
			if(isConnectedCCM == FALSE)
			{
				m_sConnectSocket->Send ("NCCM", 100); // Not Connected CCM
				return TRUE;
			}
			
			MakeStreamCurStatFromBDI (pIBDI, stPa2F+12, stPa3F+12);

			if(m_bufRcv[11] == '2')
			{
				m_sConnectSocket->Send (stPa2F, 100);
				return TRUE;
			}
			else if(m_bufRcv[11] == '3')
			{
				m_sConnectSocket->Send (stPa3F, 100);
				return TRUE;
			}
			else
			{
				m_sConnectSocket->Send ("RFOR", 100); // Reqested Floor Information is Out of Range
				return TRUE;
			}
		}
		else
		{
			m_sConnectSocket->Send ("INVP", 14);
			return TRUE;
		}
	}

	return TRUE;
}

//void CExternalNetComm::OnAccept(int nErrorCode)
//{
//	// TODO: 여기에 특수화된 코드를 추가 및/또는 기본 클래스를 호출합니다.
//
//	m_sListenSocket->Accept (*m_sConnectSocket);
//	m_bConnected = TRUE;
//	//PktRcv ();
//	CAsyncSocket::OnAccept(nErrorCode);
//}
