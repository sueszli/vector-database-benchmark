/*
 * Copyright: JessMA Open Source (ldcsaa@gmail.com)
 *
 * Author	: Bruce Liang
 * Website	: https://github.com/ldcsaa
 * Project	: https://github.com/ldcsaa/HP-Socket
 * Blog		: http://www.cnblogs.com/ldcsaa
 * Wiki		: http://www.oschina.net/p/hp-socket
 * QQ Group	: 44636872, 75375912
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
#include "SocketObject4C.h"
#include "TcpServer.h"
#include "TcpClient.h"
#include "TcpAgent.h"
#include "TcpPullServer.h"
#include "TcpPullClient.h"
#include "TcpPullAgent.h"
#include "TcpPackServer.h"
#include "TcpPackClient.h"
#include "TcpPackAgent.h"
#include "HPThreadPool.h"

#ifdef _UDP_SUPPORT
#include "UdpServer.h"
#include "UdpClient.h"
#include "UdpCast.h"
#include "UdpNode.h"
#include "UdpArqServer.h"
#include "UdpArqClient.h"
#endif

#ifdef _HTTP_SUPPORT
#include "HttpServer.h"
#include "HttpAgent.h"
#include "HttpClient.h"
#endif

/*****************************************************************************************************************************************************/
/****************************************************************** TCP/UDP Exports ******************************************************************/
/*****************************************************************************************************************************************************/

typedef C_HP_ObjectT<CTcpServer, ITcpServerListener>							C_HP_TcpServer;
typedef C_HP_ObjectT<CTcpPullServer, ITcpServerListener, sizeof(IPullSocket)>	C_HP_TcpPullServer;
typedef C_HP_ObjectT<CTcpPackServer, ITcpServerListener, sizeof(IPackSocket)>	C_HP_TcpPackServer;

typedef C_HP_ObjectT<CTcpAgent, ITcpAgentListener>								C_HP_TcpAgent;
typedef C_HP_ObjectT<CTcpPullAgent, ITcpAgentListener, sizeof(IPullSocket)>		C_HP_TcpPullAgent;
typedef C_HP_ObjectT<CTcpPackAgent, ITcpAgentListener, sizeof(IPackSocket)>		C_HP_TcpPackAgent;

typedef C_HP_ObjectT<CTcpClient, ITcpClientListener>							C_HP_TcpClient;
typedef C_HP_ObjectT<CTcpPullClient, ITcpClientListener, sizeof(IPullClient)>	C_HP_TcpPullClient;
typedef C_HP_ObjectT<CTcpPackClient, ITcpClientListener, sizeof(IPackClient)>	C_HP_TcpPackClient;

#ifdef _UDP_SUPPORT

typedef C_HP_ObjectT<CUdpServer, IUdpServerListener>							C_HP_UdpServer;
typedef C_HP_ObjectT<CUdpClient, IUdpClientListener>							C_HP_UdpClient;
typedef C_HP_ObjectT<CUdpCast, IUdpCastListener>								C_HP_UdpCast;
typedef C_HP_ObjectT<CUdpNode, IUdpNodeListener>								C_HP_UdpNode;

typedef C_HP_ObjectT<CUdpArqServer, IUdpServerListener, sizeof(IArqSocket)>		C_HP_UdpArqServer;
typedef C_HP_ObjectT<CUdpArqClient, IUdpClientListener, sizeof(IArqClient)>		C_HP_UdpArqClient;

#endif

/****************************************************/
/**************** TCP/UDP 对象创建函数 ***************/

HPSOCKET_API HP_TcpServer __HP_CALL Create_HP_TcpServer(HP_TcpServerListener pListener)
{
	return (HP_TcpServer)(new C_HP_TcpServer((ITcpServerListener*)pListener));
}

HPSOCKET_API HP_TcpAgent __HP_CALL Create_HP_TcpAgent(HP_TcpAgentListener pListener)
{
	return (HP_TcpAgent)(new C_HP_TcpAgent((ITcpAgentListener*)pListener));
}

HPSOCKET_API HP_TcpClient __HP_CALL Create_HP_TcpClient(HP_TcpClientListener pListener)
{
	return (HP_TcpClient)(new C_HP_TcpClient((ITcpClientListener*)pListener));
}

HPSOCKET_API HP_TcpPullServer __HP_CALL Create_HP_TcpPullServer(HP_TcpPullServerListener pListener)
{
	return (HP_TcpPullServer)(new C_HP_TcpPullServer((ITcpServerListener*)pListener));
}

HPSOCKET_API HP_TcpPullAgent __HP_CALL Create_HP_TcpPullAgent(HP_TcpPullAgentListener pListener)
{
	return (HP_TcpPullAgent)(new C_HP_TcpPullAgent((ITcpAgentListener*)pListener));
}

HPSOCKET_API HP_TcpPullClient __HP_CALL Create_HP_TcpPullClient(HP_TcpPullClientListener pListener)
{
	return (HP_TcpPullClient)(new C_HP_TcpPullClient((ITcpClientListener*)pListener));
}

HPSOCKET_API HP_TcpPackServer __HP_CALL Create_HP_TcpPackServer(HP_TcpServerListener pListener)
{
	return (HP_TcpPackServer)(new C_HP_TcpPackServer((ITcpServerListener*)pListener));
}

HPSOCKET_API HP_TcpPackAgent __HP_CALL Create_HP_TcpPackAgent(HP_TcpAgentListener pListener)
{
	return (HP_TcpPackAgent)(new C_HP_TcpPackAgent((ITcpAgentListener*)pListener));
}

HPSOCKET_API HP_TcpPackClient __HP_CALL Create_HP_TcpPackClient(HP_TcpClientListener pListener)
{
	return (HP_TcpPackClient)(new C_HP_TcpPackClient((ITcpClientListener*)pListener));
}

HPSOCKET_API void __HP_CALL Destroy_HP_TcpServer(HP_TcpServer pServer)
{
	delete (C_HP_TcpServer*)pServer;
}

HPSOCKET_API void __HP_CALL Destroy_HP_TcpAgent(HP_TcpAgent pAgent)
{
	delete (C_HP_TcpAgent*)pAgent;
}

HPSOCKET_API void __HP_CALL Destroy_HP_TcpClient(HP_TcpClient pClient)
{
	delete (C_HP_TcpClient*)pClient;
}

HPSOCKET_API void __HP_CALL Destroy_HP_TcpPullServer(HP_TcpPullServer pServer)
{
	delete (C_HP_TcpPullServer*)pServer;
}

HPSOCKET_API void __HP_CALL Destroy_HP_TcpPullAgent(HP_TcpPullAgent pAgent)
{
	delete (C_HP_TcpPullAgent*)pAgent;
}

HPSOCKET_API void __HP_CALL Destroy_HP_TcpPullClient(HP_TcpPullClient pClient)
{
	delete (C_HP_TcpPullClient*)pClient;
}

HPSOCKET_API void __HP_CALL Destroy_HP_TcpPackServer(HP_TcpPackServer pServer)
{
	delete (C_HP_TcpPackServer*)pServer;
}

HPSOCKET_API void __HP_CALL Destroy_HP_TcpPackAgent(HP_TcpPackAgent pAgent)
{
	delete (C_HP_TcpPackAgent*)pAgent;
}

HPSOCKET_API void __HP_CALL Destroy_HP_TcpPackClient(HP_TcpPackClient pClient)
{
	delete (C_HP_TcpPackClient*)pClient;
}

HPSOCKET_API HP_TcpServerListener __HP_CALL Create_HP_TcpServerListener()
{
	return (HP_TcpServerListener)(new C_HP_TcpServerListener);
}

HPSOCKET_API HP_TcpAgentListener __HP_CALL Create_HP_TcpAgentListener()
{
	return (HP_TcpAgentListener)(new C_HP_TcpAgentListener);
}

HPSOCKET_API HP_TcpClientListener __HP_CALL Create_HP_TcpClientListener()
{
	return (HP_TcpClientListener)(new C_HP_TcpClientListener);
}

HPSOCKET_API HP_TcpPullServerListener __HP_CALL Create_HP_TcpPullServerListener()
{
	return (HP_TcpPullServerListener)(new C_HP_TcpPullServerListener);
}

HPSOCKET_API HP_TcpPullAgentListener __HP_CALL Create_HP_TcpPullAgentListener()
{
	return (HP_TcpPullAgentListener)(new C_HP_TcpPullAgentListener);
}

HPSOCKET_API HP_TcpPullClientListener __HP_CALL Create_HP_TcpPullClientListener()
{
	return (HP_TcpPullClientListener)(new C_HP_TcpPullClientListener);
}

HPSOCKET_API HP_TcpPackServerListener __HP_CALL Create_HP_TcpPackServerListener()
{
	return (HP_TcpPackServerListener)(new C_HP_TcpPackServerListener);
}

HPSOCKET_API HP_TcpPackAgentListener __HP_CALL Create_HP_TcpPackAgentListener()
{
	return (HP_TcpPackAgentListener)(new C_HP_TcpPackAgentListener);
}

HPSOCKET_API HP_TcpPackClientListener __HP_CALL Create_HP_TcpPackClientListener()
{
	return (HP_TcpPackClientListener)(new C_HP_TcpPackClientListener);
}

HPSOCKET_API void __HP_CALL Destroy_HP_TcpServerListener(HP_TcpServerListener pListener)
{
	delete (C_HP_TcpServerListener*)pListener;
}

HPSOCKET_API void __HP_CALL Destroy_HP_TcpAgentListener(HP_TcpAgentListener pListener)
{
	delete (C_HP_TcpAgentListener*)pListener;
}

HPSOCKET_API void __HP_CALL Destroy_HP_TcpClientListener(HP_TcpClientListener pListener)
{
	delete (C_HP_TcpClientListener*)pListener;
}

HPSOCKET_API void __HP_CALL Destroy_HP_TcpPullServerListener(HP_TcpPullServerListener pListener)
{
	delete (C_HP_TcpPullServerListener*)pListener;
}

HPSOCKET_API void __HP_CALL Destroy_HP_TcpPullAgentListener(HP_TcpPullAgentListener pListener)
{
	delete (C_HP_TcpPullAgentListener*)pListener;
}

HPSOCKET_API void __HP_CALL Destroy_HP_TcpPullClientListener(HP_TcpPullClientListener pListener)
{
	delete (C_HP_TcpPullClientListener*)pListener;
}

HPSOCKET_API void __HP_CALL Destroy_HP_TcpPackServerListener(HP_TcpPackServerListener pListener)
{
	delete (C_HP_TcpPackServerListener*)pListener;
}

HPSOCKET_API void __HP_CALL Destroy_HP_TcpPackAgentListener(HP_TcpPackAgentListener pListener)
{
	delete (C_HP_TcpPackAgentListener*)pListener;
}

HPSOCKET_API void __HP_CALL Destroy_HP_TcpPackClientListener(HP_TcpPackClientListener pListener)
{
	delete (C_HP_TcpPackClientListener*)pListener;
}

#ifdef _UDP_SUPPORT

HPSOCKET_API HP_UdpServer __HP_CALL Create_HP_UdpServer(HP_UdpServerListener pListener)
{
	return (HP_UdpServer)(new C_HP_UdpServer((IUdpServerListener*)pListener));
}

HPSOCKET_API HP_UdpClient __HP_CALL Create_HP_UdpClient(HP_UdpClientListener pListener)
{
	return (HP_UdpClient)(new C_HP_UdpClient((IUdpClientListener*)pListener));
}

HPSOCKET_API HP_UdpCast __HP_CALL Create_HP_UdpCast(HP_UdpCastListener pListener)
{
	return (HP_UdpCast)(new C_HP_UdpCast((IUdpCastListener*)pListener));
}

HPSOCKET_API HP_UdpNode __HP_CALL Create_HP_UdpNode(HP_UdpNodeListener pListener)
{
	return (HP_UdpNode)(new C_HP_UdpNode((IUdpNodeListener*)pListener));
}

HPSOCKET_API HP_UdpArqServer __HP_CALL Create_HP_UdpArqServer(HP_UdpServerListener pListener)
{
	return (HP_UdpArqServer)(new C_HP_UdpArqServer((IUdpServerListener*)pListener));
}

HPSOCKET_API HP_UdpArqClient __HP_CALL Create_HP_UdpArqClient(HP_UdpClientListener pListener)
{
	return (HP_UdpArqClient)(new C_HP_UdpArqClient((IUdpClientListener*)pListener));
}

HPSOCKET_API void __HP_CALL Destroy_HP_UdpServer(HP_UdpServer pServer)
{
	delete (C_HP_UdpServer*)pServer;
}

HPSOCKET_API void __HP_CALL Destroy_HP_UdpClient(HP_UdpClient pClient)
{
	delete (C_HP_UdpClient*)pClient;
}

HPSOCKET_API void __HP_CALL Destroy_HP_UdpCast(HP_UdpCast pCast)
{
	delete (C_HP_UdpCast*)pCast;
}

HPSOCKET_API void __HP_CALL Destroy_HP_UdpNode(HP_UdpNode pNode)
{
	delete (C_HP_UdpNode*)pNode;
}

HPSOCKET_API void __HP_CALL Destroy_HP_UdpArqServer(HP_UdpArqServer pServer)
{
	delete (C_HP_UdpArqServer*)pServer;
}

HPSOCKET_API void __HP_CALL Destroy_HP_UdpArqClient(HP_UdpArqClient pClient)
{
	delete (C_HP_UdpArqClient*)pClient;
}

HPSOCKET_API HP_UdpServerListener __HP_CALL Create_HP_UdpServerListener()
{
	return (HP_UdpServerListener)(new C_HP_UdpServerListener);
}

HPSOCKET_API HP_UdpClientListener __HP_CALL Create_HP_UdpClientListener()
{
	return (HP_UdpClientListener)(new C_HP_UdpClientListener);
}

HPSOCKET_API HP_UdpCastListener __HP_CALL Create_HP_UdpCastListener()
{
	return (HP_UdpCastListener)(new C_HP_UdpCastListener);
}

HPSOCKET_API HP_UdpNodeListener __HP_CALL Create_HP_UdpNodeListener()
{
	return (HP_UdpNodeListener)(new C_HP_UdpNodeListener);
}

HPSOCKET_API HP_UdpArqServerListener __HP_CALL Create_HP_UdpArqServerListener()
{
	return (HP_UdpArqServerListener)(new C_HP_UdpArqServerListener);
}

HPSOCKET_API HP_UdpArqClientListener __HP_CALL Create_HP_UdpArqClientListener()
{
	return (HP_UdpArqClientListener)(new C_HP_UdpArqClientListener);
}

HPSOCKET_API void __HP_CALL Destroy_HP_UdpServerListener(HP_UdpServerListener pListener)
{
	delete (C_HP_UdpServerListener*)pListener;
}

HPSOCKET_API void __HP_CALL Destroy_HP_UdpClientListener(HP_UdpClientListener pListener)
{
	delete (C_HP_UdpClientListener*)pListener;
}

HPSOCKET_API void __HP_CALL Destroy_HP_UdpCastListener(HP_UdpCastListener pListener)
{
	delete (C_HP_UdpClientListener*)pListener;
}

HPSOCKET_API void __HP_CALL Destroy_HP_UdpNodeListener(HP_UdpNodeListener pListener)
{
	delete (C_HP_UdpNodeListener*)pListener;
}

HPSOCKET_API void __HP_CALL Destroy_HP_UdpArqServerListener(HP_UdpArqServerListener pListener)
{
	delete (C_HP_UdpArqServerListener*)pListener;
}

HPSOCKET_API void __HP_CALL Destroy_HP_UdpArqClientListener(HP_UdpArqClientListener pListener)
{
	delete (C_HP_UdpArqClientListener*)pListener;
}

#endif

/**********************************************************************************/
/***************************** Server 回调函数设置方法 *****************************/

HPSOCKET_API void __HP_CALL HP_Set_FN_Server_OnPrepareListen(HP_ServerListener pListener, HP_FN_Server_OnPrepareListen fn)
{
	((C_HP_TcpServerListener*)pListener)->m_fnOnPrepareListen = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_Server_OnAccept(HP_ServerListener pListener, HP_FN_Server_OnAccept fn)
{
	((C_HP_TcpServerListener*)pListener)->m_fnOnAccept = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_Server_OnHandShake(HP_ServerListener pListener, HP_FN_Server_OnHandShake fn)
{
	((C_HP_TcpServerListener*)pListener)->m_fnOnHandShake = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_Server_OnSend(HP_ServerListener pListener, HP_FN_Server_OnSend fn)
{
	((C_HP_TcpServerListener*)pListener)->m_fnOnSend = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_Server_OnReceive(HP_ServerListener pListener, HP_FN_Server_OnReceive fn)
{
	((C_HP_TcpServerListener*)pListener)->m_fnOnReceive = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_Server_OnPullReceive(HP_ServerListener pListener, HP_FN_Server_OnPullReceive fn)
{
	((C_HP_TcpServerListener*)pListener)->m_fnOnPullReceive = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_Server_OnClose(HP_ServerListener pListener, HP_FN_Server_OnClose fn)
{
	((C_HP_TcpServerListener*)pListener)->m_fnOnClose = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_Server_OnShutdown(HP_ServerListener pListener, HP_FN_Server_OnShutdown fn)
{
	((C_HP_TcpServerListener*)pListener)->m_fnOnShutdown = fn;
}

/**********************************************************************************/
/***************************** Agent 回调函数设置方法 *****************************/

HPSOCKET_API void __HP_CALL HP_Set_FN_Agent_OnPrepareConnect(HP_AgentListener pListener, HP_FN_Agent_OnPrepareConnect fn)
{
	((C_HP_TcpAgentListener*)pListener)->m_fnOnPrepareConnect = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_Agent_OnConnect(HP_AgentListener pListener, HP_FN_Agent_OnConnect fn)
{
	((C_HP_TcpAgentListener*)pListener)->m_fnOnConnect = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_Agent_OnHandShake(HP_AgentListener pListener, HP_FN_Agent_OnHandShake fn)
{
	((C_HP_TcpAgentListener*)pListener)->m_fnOnHandShake = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_Agent_OnSend(HP_AgentListener pListener, HP_FN_Agent_OnSend fn)
{
	((C_HP_TcpAgentListener*)pListener)->m_fnOnSend = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_Agent_OnReceive(HP_AgentListener pListener, HP_FN_Agent_OnReceive fn)
{
	((C_HP_TcpAgentListener*)pListener)->m_fnOnReceive = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_Agent_OnPullReceive(HP_AgentListener pListener, HP_FN_Agent_OnPullReceive fn)
{
	((C_HP_TcpAgentListener*)pListener)->m_fnOnPullReceive = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_Agent_OnClose(HP_AgentListener pListener, HP_FN_Agent_OnClose fn)
{
	((C_HP_TcpAgentListener*)pListener)->m_fnOnClose = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_Agent_OnShutdown(HP_AgentListener pListener, HP_FN_Agent_OnShutdown fn)
{
	((C_HP_TcpAgentListener*)pListener)->m_fnOnShutdown = fn;
}

/**********************************************************************************/
/***************************** Client 回调函数设置方法 *****************************/

HPSOCKET_API void __HP_CALL HP_Set_FN_Client_OnPrepareConnect(HP_ClientListener pListener, HP_FN_Client_OnPrepareConnect fn)
{
	((C_HP_TcpClientListener*)pListener)->m_fnOnPrepareConnect = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_Client_OnConnect(HP_ClientListener pListener, HP_FN_Client_OnConnect fn)
{
	((C_HP_TcpClientListener*)pListener)->m_fnOnConnect = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_Client_OnHandShake(HP_ClientListener pListener, HP_FN_Client_OnHandShake fn)
{
	((C_HP_TcpClientListener*)pListener)->m_fnOnHandShake = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_Client_OnSend(HP_ClientListener pListener, HP_FN_Client_OnSend fn)
{
	((C_HP_TcpClientListener*)pListener)->m_fnOnSend = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_Client_OnReceive(HP_ClientListener pListener, HP_FN_Client_OnReceive fn)
{
	((C_HP_TcpClientListener*)pListener)->m_fnOnReceive = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_Client_OnPullReceive(HP_ClientListener pListener, HP_FN_Client_OnPullReceive fn)
{
	((C_HP_TcpClientListener*)pListener)->m_fnOnPullReceive = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_Client_OnClose(HP_ClientListener pListener, HP_FN_Client_OnClose fn)
{
	((C_HP_TcpClientListener*)pListener)->m_fnOnClose = fn;
}

#ifdef _UDP_SUPPORT

/**********************************************************************************/
/***************************** UdpNode 回调函数设置方法 *****************************/


HPSOCKET_API void __HP_CALL HP_Set_FN_UdpNode_OnPrepareListen(HP_UdpNodeListener pListener, HP_FN_UdpNode_OnPrepareListen fn)
{
	((C_HP_UdpNodeListener*)pListener)->m_fnOnPrepareListen = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_UdpNode_OnSend(HP_UdpNodeListener pListener, HP_FN_UdpNode_OnSend fn)
{
	((C_HP_UdpNodeListener*)pListener)->m_fnOnSend = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_UdpNode_OnReceive(HP_UdpNodeListener pListener, HP_FN_UdpNode_OnReceive fn)
{
	((C_HP_UdpNodeListener*)pListener)->m_fnOnReceive = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_UdpNode_OnError(HP_UdpNodeListener pListener, HP_FN_UdpNode_OnError fn)
{
	((C_HP_UdpNodeListener*)pListener)->m_fnOnError = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_UdpNode_OnShutdown(HP_UdpNodeListener pListener, HP_FN_UdpNode_OnShutdown fn)
{
	((C_HP_UdpNodeListener*)pListener)->m_fnOnShutdown = fn;
}

#endif

/**************************************************************************/
/***************************** Server 操作方法 *****************************/

HPSOCKET_API BOOL __HP_CALL HP_Server_Start(HP_Server pServer, LPCTSTR lpszBindAddress, USHORT usPort)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->Start(lpszBindAddress, usPort);
}

HPSOCKET_API BOOL __HP_CALL HP_Server_Stop(HP_Server pServer)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->Stop();
}

HPSOCKET_API BOOL __HP_CALL HP_Server_Wait(HP_Server pServer, DWORD dwMilliseconds)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->Wait(dwMilliseconds);
}

HPSOCKET_API BOOL __HP_CALL HP_Server_Send(HP_Server pServer, HP_CONNID dwConnID, const BYTE* pBuffer, int iLength)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->Send(dwConnID, pBuffer, iLength);
}

HPSOCKET_API BOOL __HP_CALL HP_Server_SendPart(HP_Server pServer, HP_CONNID dwConnID, const BYTE* pBuffer, int iLength, int iOffset)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->Send(dwConnID, pBuffer, iLength, iOffset);
}

HPSOCKET_API BOOL __HP_CALL HP_Server_SendPackets(HP_Server pServer, HP_CONNID dwConnID, const WSABUF pBuffers[], int iCount)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->SendPackets(dwConnID, pBuffers, iCount);
}

HPSOCKET_API BOOL __HP_CALL HP_Server_PauseReceive(HP_Server pServer, HP_CONNID dwConnID, BOOL bPause)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->PauseReceive(dwConnID, bPause);
}

HPSOCKET_API BOOL __HP_CALL HP_Server_Disconnect(HP_Server pServer, HP_CONNID dwConnID, BOOL bForce)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->Disconnect(dwConnID, bForce);
}

HPSOCKET_API BOOL __HP_CALL HP_Server_DisconnectLongConnections(HP_Server pServer, DWORD dwPeriod, BOOL bForce)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->DisconnectLongConnections(dwPeriod, bForce);
}

HPSOCKET_API BOOL __HP_CALL HP_Server_DisconnectSilenceConnections(HP_Server pServer, DWORD dwPeriod, BOOL bForce)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->DisconnectSilenceConnections(dwPeriod, bForce);
}

/******************************************************************************/
/***************************** Server 属性访问方法 *****************************/

HPSOCKET_API BOOL __HP_CALL HP_Server_SetConnectionExtra(HP_Server pServer, HP_CONNID dwConnID, PVOID pExtra)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->SetConnectionExtra(dwConnID, pExtra);
}

HPSOCKET_API BOOL __HP_CALL HP_Server_GetConnectionExtra(HP_Server pServer, HP_CONNID dwConnID, PVOID* ppExtra)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->GetConnectionExtra(dwConnID, ppExtra);
}

HPSOCKET_API BOOL __HP_CALL HP_Server_IsSecure(HP_Server pServer)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->IsSecure();
}

HPSOCKET_API BOOL __HP_CALL HP_Server_HasStarted(HP_Server pServer)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->HasStarted();
}

HPSOCKET_API En_HP_ServiceState __HP_CALL HP_Server_GetState(HP_Server pServer)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->GetState();
}

HPSOCKET_API En_HP_SocketError __HP_CALL HP_Server_GetLastError(HP_Server pServer)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->GetLastError();
}

HPSOCKET_API LPCTSTR __HP_CALL HP_Server_GetLastErrorDesc(HP_Server pServer)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->GetLastErrorDesc();
}

HPSOCKET_API BOOL __HP_CALL HP_Server_GetPendingDataLength(HP_Server pServer, HP_CONNID dwConnID, int* piPending)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->GetPendingDataLength(dwConnID, *piPending);
}

HPSOCKET_API BOOL __HP_CALL HP_Server_IsPauseReceive(HP_Server pServer, HP_CONNID dwConnID, BOOL* pbPaused)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->IsPauseReceive(dwConnID, *pbPaused);
}

HPSOCKET_API BOOL __HP_CALL HP_Server_IsConnected(HP_Server pServer, HP_CONNID dwConnID)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->IsConnected(dwConnID);
}

HPSOCKET_API DWORD __HP_CALL HP_Server_GetConnectionCount(HP_Server pServer)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->GetConnectionCount();
}

HPSOCKET_API BOOL __HP_CALL HP_Server_GetAllConnectionIDs(HP_Server pServer, HP_CONNID pIDs[], DWORD* pdwCount)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->GetAllConnectionIDs(pIDs, *pdwCount);
}

HPSOCKET_API BOOL __HP_CALL HP_Server_GetConnectPeriod(HP_Server pServer, HP_CONNID dwConnID, DWORD* pdwPeriod)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->GetConnectPeriod(dwConnID, *pdwPeriod);
}

HPSOCKET_API BOOL __HP_CALL HP_Server_GetSilencePeriod(HP_Server pServer, HP_CONNID dwConnID, DWORD* pdwPeriod)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->GetSilencePeriod(dwConnID, *pdwPeriod);
}

HPSOCKET_API BOOL __HP_CALL HP_Server_GetListenAddress(HP_Server pServer, TCHAR lpszAddress[], int* piAddressLen, USHORT* pusPort)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->GetListenAddress(lpszAddress, *piAddressLen, *pusPort);
}

HPSOCKET_API BOOL __HP_CALL HP_Server_GetLocalAddress(HP_Server pServer, HP_CONNID dwConnID, TCHAR lpszAddress[], int* piAddressLen, USHORT* pusPort)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->GetLocalAddress(dwConnID, lpszAddress, *piAddressLen, *pusPort);
}

HPSOCKET_API BOOL __HP_CALL HP_Server_GetRemoteAddress(HP_Server pServer, HP_CONNID dwConnID, TCHAR lpszAddress[], int* piAddressLen, USHORT* pusPort)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->GetRemoteAddress(dwConnID, lpszAddress, *piAddressLen, *pusPort);
}

HPSOCKET_API void __HP_CALL HP_Server_SetReuseAddressPolicy(HP_Server pServer, En_HP_ReuseAddressPolicy enReusePolicy)
{
	C_HP_Object::ToSecond<IServer>(pServer)->SetReuseAddressPolicy(enReusePolicy);
}

HPSOCKET_API void __HP_CALL HP_Server_SetSendPolicy(HP_Server pServer, En_HP_SendPolicy enSendPolicy)
{
	C_HP_Object::ToSecond<IServer>(pServer)->SetSendPolicy(enSendPolicy);
}

HPSOCKET_API void __HP_CALL HP_Server_SetOnSendSyncPolicy(HP_Server pServer, En_HP_OnSendSyncPolicy enSyncPolicy)
{
	C_HP_Object::ToSecond<IServer>(pServer)->SetOnSendSyncPolicy(enSyncPolicy);
}

HPSOCKET_API void __HP_CALL HP_Server_SetFreeSocketObjLockTime(HP_Server pServer, DWORD dwFreeSocketObjLockTime)
{
	C_HP_Object::ToSecond<IServer>(pServer)->SetFreeSocketObjLockTime(dwFreeSocketObjLockTime);
}

HPSOCKET_API void __HP_CALL HP_Server_SetFreeSocketObjPool(HP_Server pServer, DWORD dwFreeSocketObjPool)
{
	C_HP_Object::ToSecond<IServer>(pServer)->SetFreeSocketObjPool(dwFreeSocketObjPool);
}

HPSOCKET_API void __HP_CALL HP_Server_SetFreeBufferObjPool(HP_Server pServer, DWORD dwFreeBufferObjPool)
{
	C_HP_Object::ToSecond<IServer>(pServer)->SetFreeBufferObjPool(dwFreeBufferObjPool);
}

HPSOCKET_API void __HP_CALL HP_Server_SetFreeSocketObjHold(HP_Server pServer, DWORD dwFreeSocketObjHold)
{
	C_HP_Object::ToSecond<IServer>(pServer)->SetFreeSocketObjHold(dwFreeSocketObjHold);
}

HPSOCKET_API void __HP_CALL HP_Server_SetFreeBufferObjHold(HP_Server pServer, DWORD dwFreeBufferObjHold)
{
	C_HP_Object::ToSecond<IServer>(pServer)->SetFreeBufferObjHold(dwFreeBufferObjHold);
}

HPSOCKET_API void __HP_CALL HP_Server_SetMaxConnectionCount(HP_Server pServer, DWORD dwMaxConnectionCount)
{
	C_HP_Object::ToSecond<IServer>(pServer)->SetMaxConnectionCount(dwMaxConnectionCount);
}

HPSOCKET_API void __HP_CALL HP_Server_SetWorkerThreadCount(HP_Server pServer, DWORD dwWorkerThreadCount)
{
	C_HP_Object::ToSecond<IServer>(pServer)->SetWorkerThreadCount(dwWorkerThreadCount);
}

HPSOCKET_API void __HP_CALL HP_Server_SetMarkSilence(HP_Server pServer, BOOL bMarkSilence)
{
	C_HP_Object::ToSecond<IServer>(pServer)->SetMarkSilence(bMarkSilence);
}

HPSOCKET_API En_HP_ReuseAddressPolicy __HP_CALL HP_Server_GetReuseAddressPolicy(HP_Server pServer)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->GetReuseAddressPolicy();
}

HPSOCKET_API En_HP_SendPolicy __HP_CALL HP_Server_GetSendPolicy(HP_Server pServer)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->GetSendPolicy();
}

HPSOCKET_API En_HP_OnSendSyncPolicy __HP_CALL HP_Server_GetOnSendSyncPolicy(HP_Server pServer)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->GetOnSendSyncPolicy();
}

HPSOCKET_API DWORD __HP_CALL HP_Server_GetFreeSocketObjLockTime(HP_Server pServer)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->GetFreeSocketObjLockTime();
}

HPSOCKET_API DWORD __HP_CALL HP_Server_GetFreeSocketObjPool(HP_Server pServer)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->GetFreeSocketObjPool();
}

HPSOCKET_API DWORD __HP_CALL HP_Server_GetFreeBufferObjPool(HP_Server pServer)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->GetFreeBufferObjPool();
}

HPSOCKET_API DWORD __HP_CALL HP_Server_GetFreeSocketObjHold(HP_Server pServer)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->GetFreeSocketObjHold();
}

HPSOCKET_API DWORD __HP_CALL HP_Server_GetFreeBufferObjHold(HP_Server pServer)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->GetFreeBufferObjHold();
}

HPSOCKET_API DWORD __HP_CALL HP_Server_GetMaxConnectionCount(HP_Server pServer)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->GetMaxConnectionCount();
}

HPSOCKET_API DWORD __HP_CALL HP_Server_GetWorkerThreadCount(HP_Server pServer)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->GetWorkerThreadCount();
}

HPSOCKET_API BOOL __HP_CALL HP_Server_IsMarkSilence(HP_Server pServer)
{
	return C_HP_Object::ToSecond<IServer>(pServer)->IsMarkSilence();
}

/**********************************************************************************/
/******************************* TCP Server 操作方法 *******************************/

HPSOCKET_API BOOL __HP_CALL HP_TcpServer_SendSmallFile(HP_Server pServer, HP_CONNID dwConnID, LPCTSTR lpszFileName, const LPWSABUF pHead, const LPWSABUF pTail)
{
	return C_HP_Object::ToSecond<ITcpServer>(pServer)->SendSmallFile(dwConnID, lpszFileName, pHead, pTail);
}

/**********************************************************************************/
/***************************** TCP Server 属性访问方法 *****************************/

HPSOCKET_API void __HP_CALL HP_TcpServer_SetAcceptSocketCount(HP_TcpServer pServer, DWORD dwAcceptSocketCount)
{
	C_HP_Object::ToSecond<ITcpServer>(pServer)->SetAcceptSocketCount(dwAcceptSocketCount);
}

HPSOCKET_API void __HP_CALL HP_TcpServer_SetSocketBufferSize(HP_TcpServer pServer, DWORD dwSocketBufferSize)
{
	C_HP_Object::ToSecond<ITcpServer>(pServer)->SetSocketBufferSize(dwSocketBufferSize);
}

HPSOCKET_API void __HP_CALL HP_TcpServer_SetSocketListenQueue(HP_TcpServer pServer, DWORD dwSocketListenQueue)
{
	C_HP_Object::ToSecond<ITcpServer>(pServer)->SetSocketListenQueue(dwSocketListenQueue);
}

HPSOCKET_API void __HP_CALL HP_TcpServer_SetKeepAliveTime(HP_TcpServer pServer, DWORD dwKeepAliveTime)
{
	C_HP_Object::ToSecond<ITcpServer>(pServer)->SetKeepAliveTime(dwKeepAliveTime);
}

HPSOCKET_API void __HP_CALL HP_TcpServer_SetKeepAliveInterval(HP_TcpServer pServer, DWORD dwKeepAliveInterval)
{
	C_HP_Object::ToSecond<ITcpServer>(pServer)->SetKeepAliveInterval(dwKeepAliveInterval);
}

HPSOCKET_API void __HP_CALL HP_TcpServer_SetNoDelay(HP_TcpServer pServer, BOOL bNoDelay)
{
	C_HP_Object::ToSecond<ITcpServer>(pServer)->SetNoDelay(bNoDelay);
}

HPSOCKET_API DWORD __HP_CALL HP_TcpServer_GetAcceptSocketCount(HP_TcpServer pServer)
{
	return C_HP_Object::ToSecond<ITcpServer>(pServer)->GetAcceptSocketCount();
}

HPSOCKET_API DWORD __HP_CALL HP_TcpServer_GetSocketBufferSize(HP_TcpServer pServer)
{
	return C_HP_Object::ToSecond<ITcpServer>(pServer)->GetSocketBufferSize();
}

HPSOCKET_API DWORD __HP_CALL HP_TcpServer_GetSocketListenQueue(HP_TcpServer pServer)
{
	return C_HP_Object::ToSecond<ITcpServer>(pServer)->GetSocketListenQueue();
}

HPSOCKET_API DWORD __HP_CALL HP_TcpServer_GetKeepAliveTime(HP_TcpServer pServer)
{
	return C_HP_Object::ToSecond<ITcpServer>(pServer)->GetKeepAliveTime();
}

HPSOCKET_API DWORD __HP_CALL HP_TcpServer_GetKeepAliveInterval(HP_TcpServer pServer)
{
	return C_HP_Object::ToSecond<ITcpServer>(pServer)->GetKeepAliveInterval();
}

HPSOCKET_API BOOL __HP_CALL HP_TcpServer_IsNoDelay(HP_TcpServer pServer)
{
	return C_HP_Object::ToSecond<ITcpServer>(pServer)->IsNoDelay();
}

#ifdef _UDP_SUPPORT

/**********************************************************************************/
/***************************** UDP Server 属性访问方法 *****************************/

HPSOCKET_API void __HP_CALL HP_UdpServer_SetMaxDatagramSize(HP_UdpServer pServer, DWORD dwMaxDatagramSize)
{
	C_HP_Object::ToSecond<IUdpServer>(pServer)->SetMaxDatagramSize(dwMaxDatagramSize);
}

HPSOCKET_API DWORD __HP_CALL HP_UdpServer_GetMaxDatagramSize(HP_UdpServer pServer)
{
	return C_HP_Object::ToSecond<IUdpServer>(pServer)->GetMaxDatagramSize();
}

HPSOCKET_API void __HP_CALL HP_UdpServer_SetPostReceiveCount(HP_UdpServer pServer, DWORD dwPostReceiveCount)
{
	C_HP_Object::ToSecond<IUdpServer>(pServer)->SetPostReceiveCount(dwPostReceiveCount);
}

HPSOCKET_API DWORD __HP_CALL HP_UdpServer_GetPostReceiveCount(HP_UdpServer pServer)
{
	return C_HP_Object::ToSecond<IUdpServer>(pServer)->GetPostReceiveCount();
}

HPSOCKET_API void __HP_CALL HP_UdpServer_SetDetectAttempts(HP_UdpServer pServer, DWORD dwDetectAttempts)
{
	C_HP_Object::ToSecond<IUdpServer>(pServer)->SetDetectAttempts(dwDetectAttempts);
}

HPSOCKET_API void __HP_CALL HP_UdpServer_SetDetectInterval(HP_UdpServer pServer, DWORD dwDetectInterval)
{
	C_HP_Object::ToSecond<IUdpServer>(pServer)->SetDetectInterval(dwDetectInterval);
}

HPSOCKET_API DWORD __HP_CALL HP_UdpServer_GetDetectAttempts(HP_UdpServer pServer)
{
	return C_HP_Object::ToSecond<IUdpServer>(pServer)->GetDetectAttempts();
}

HPSOCKET_API DWORD __HP_CALL HP_UdpServer_GetDetectInterval(HP_UdpServer pServer)
{
	return C_HP_Object::ToSecond<IUdpServer>(pServer)->GetDetectInterval();
}

/**********************************************************************************/
/*************************** UDP ARQ Server 属性访问方法 ***************************/

HPSOCKET_API void __HP_CALL HP_UdpArqServer_SetNoDelay(HP_UdpArqServer pServer, BOOL bNoDelay)
{
	C_HP_Object::ToFirst<IArqSocket>(pServer)->SetNoDelay(bNoDelay);
}

HPSOCKET_API void __HP_CALL HP_UdpArqServer_SetTurnoffCongestCtrl(HP_UdpArqServer pServer, BOOL bTurnOff)
{
	C_HP_Object::ToFirst<IArqSocket>(pServer)->SetTurnoffCongestCtrl(bTurnOff);
}

HPSOCKET_API void __HP_CALL HP_UdpArqServer_SetFlushInterval(HP_UdpArqServer pServer, DWORD dwFlushInterval)
{
	C_HP_Object::ToFirst<IArqSocket>(pServer)->SetFlushInterval(dwFlushInterval);
}

HPSOCKET_API void __HP_CALL HP_UdpArqServer_SetResendByAcks(HP_UdpArqServer pServer, DWORD dwResendByAcks)
{
	C_HP_Object::ToFirst<IArqSocket>(pServer)->SetResendByAcks(dwResendByAcks);
}

HPSOCKET_API void __HP_CALL HP_UdpArqServer_SetSendWndSize(HP_UdpArqServer pServer, DWORD dwSendWndSize)
{
	C_HP_Object::ToFirst<IArqSocket>(pServer)->SetSendWndSize(dwSendWndSize);
}

HPSOCKET_API void __HP_CALL HP_UdpArqServer_SetRecvWndSize(HP_UdpArqServer pServer, DWORD dwRecvWndSize)
{
	C_HP_Object::ToFirst<IArqSocket>(pServer)->SetRecvWndSize(dwRecvWndSize);
}

HPSOCKET_API void __HP_CALL HP_UdpArqServer_SetMinRto(HP_UdpArqServer pServer, DWORD dwMinRto)
{
	C_HP_Object::ToFirst<IArqSocket>(pServer)->SetMinRto(dwMinRto);
}

HPSOCKET_API void __HP_CALL HP_UdpArqServer_SetFastLimit(HP_UdpArqServer pServer, DWORD dwFastLimit)
{
	C_HP_Object::ToFirst<IArqSocket>(pServer)->SetFastLimit(dwFastLimit);
}

HPSOCKET_API void __HP_CALL HP_UdpArqServer_SetMaxTransUnit(HP_UdpArqServer pServer, DWORD dwMaxTransUnit)
{
	C_HP_Object::ToFirst<IArqSocket>(pServer)->SetMaxTransUnit(dwMaxTransUnit);
}

HPSOCKET_API void __HP_CALL HP_UdpArqServer_SetMaxMessageSize(HP_UdpArqServer pServer, DWORD dwMaxMessageSize)
{
	C_HP_Object::ToFirst<IArqSocket>(pServer)->SetMaxMessageSize(dwMaxMessageSize);
}

HPSOCKET_API void __HP_CALL HP_UdpArqServer_SetHandShakeTimeout(HP_UdpArqServer pServer, DWORD dwHandShakeTimeout)
{
	C_HP_Object::ToFirst<IArqSocket>(pServer)->SetHandShakeTimeout(dwHandShakeTimeout);
}

HPSOCKET_API BOOL __HP_CALL HP_UdpArqServer_IsNoDelay(HP_UdpArqServer pServer)
{
	return C_HP_Object::ToFirst<IArqSocket>(pServer)->IsNoDelay();
}

HPSOCKET_API BOOL __HP_CALL HP_UdpArqServer_IsTurnoffCongestCtrl(HP_UdpArqServer pServer)
{
	return C_HP_Object::ToFirst<IArqSocket>(pServer)->IsTurnoffCongestCtrl();
}

HPSOCKET_API DWORD __HP_CALL HP_UdpArqServer_GetFlushInterval(HP_UdpArqServer pServer)
{
	return C_HP_Object::ToFirst<IArqSocket>(pServer)->GetFlushInterval();
}

HPSOCKET_API DWORD __HP_CALL HP_UdpArqServer_GetResendByAcks(HP_UdpArqServer pServer)
{
	return C_HP_Object::ToFirst<IArqSocket>(pServer)->GetResendByAcks();
}

HPSOCKET_API DWORD __HP_CALL HP_UdpArqServer_GetSendWndSize(HP_UdpArqServer pServer)
{
	return C_HP_Object::ToFirst<IArqSocket>(pServer)->GetSendWndSize();
}

HPSOCKET_API DWORD __HP_CALL HP_UdpArqServer_GetRecvWndSize(HP_UdpArqServer pServer)
{
	return C_HP_Object::ToFirst<IArqSocket>(pServer)->GetRecvWndSize();
}

HPSOCKET_API DWORD __HP_CALL HP_UdpArqServer_GetMinRto(HP_UdpArqServer pServer)
{
	return C_HP_Object::ToFirst<IArqSocket>(pServer)->GetMinRto();
}

HPSOCKET_API DWORD __HP_CALL HP_UdpArqServer_GetFastLimit(HP_UdpArqServer pServer)
{
	return C_HP_Object::ToFirst<IArqSocket>(pServer)->GetFastLimit();
}

HPSOCKET_API DWORD __HP_CALL HP_UdpArqServer_GetMaxTransUnit(HP_UdpArqServer pServer)
{
	return C_HP_Object::ToFirst<IArqSocket>(pServer)->GetMaxTransUnit();
}

HPSOCKET_API DWORD __HP_CALL HP_UdpArqServer_GetMaxMessageSize(HP_UdpArqServer pServer)
{
	return C_HP_Object::ToFirst<IArqSocket>(pServer)->GetMaxMessageSize();
}

HPSOCKET_API DWORD __HP_CALL HP_UdpArqServer_GetHandShakeTimeout(HP_UdpArqServer pServer)
{
	return C_HP_Object::ToFirst<IArqSocket>(pServer)->GetHandShakeTimeout();
}

HPSOCKET_API BOOL __HP_CALL HP_UdpArqServer_GetWaitingSendMessageCount(HP_UdpArqServer pServer, HP_CONNID dwConnID, int* piCount)
{
	return C_HP_Object::ToFirst<IArqSocket>(pServer)->GetWaitingSendMessageCount(dwConnID, *piCount);
}

#endif

/**************************************************************************/
/***************************** Agent 操作方法 *****************************/

HPSOCKET_API BOOL __HP_CALL HP_Agent_Start(HP_Agent pAgent, LPCTSTR lpszBindAddress, BOOL bAsyncConnect)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->Start(lpszBindAddress, bAsyncConnect);
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_Stop(HP_Agent pAgent)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->Stop();
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_Wait(HP_Agent pAgent, DWORD dwMilliseconds)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->Wait(dwMilliseconds);
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_Connect(HP_Agent pAgent, LPCTSTR lpszRemoteAddress, USHORT usPort, HP_CONNID* pdwConnID)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->Connect(lpszRemoteAddress, usPort, pdwConnID);
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_ConnectWithExtra(HP_Agent pAgent, LPCTSTR lpszRemoteAddress, USHORT usPort, HP_CONNID* pdwConnID, PVOID pExtra)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->Connect(lpszRemoteAddress, usPort, pdwConnID, pExtra);
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_ConnectWithLocalPort(HP_Agent pAgent, LPCTSTR lpszRemoteAddress, USHORT usPort, HP_CONNID* pdwConnID, USHORT usLocalPort)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->Connect(lpszRemoteAddress, usPort, pdwConnID, nullptr, usLocalPort);
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_ConnectWithLocalAddress(HP_Agent pAgent, LPCTSTR lpszRemoteAddress, USHORT usPort, HP_CONNID* pdwConnID, LPCTSTR lpszLocalAddress)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->Connect(lpszRemoteAddress, usPort, pdwConnID, nullptr, 0, lpszLocalAddress);
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_ConnectWithExtraAndLocalPort(HP_Agent pAgent, LPCTSTR lpszRemoteAddress, USHORT usPort, HP_CONNID* pdwConnID, PVOID pExtra, USHORT usLocalPort)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->Connect(lpszRemoteAddress, usPort, pdwConnID, pExtra, usLocalPort);
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_ConnectWithExtraAndLocalAddressPort(HP_Agent pAgent, LPCTSTR lpszRemoteAddress, USHORT usPort, HP_CONNID* pdwConnID, PVOID pExtra, USHORT usLocalPort, LPCTSTR lpszLocalAddress)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->Connect(lpszRemoteAddress, usPort, pdwConnID, pExtra, usLocalPort, lpszLocalAddress);
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_Send(HP_Agent pAgent, HP_CONNID dwConnID, const BYTE* pBuffer, int iLength)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->Send(dwConnID, pBuffer, iLength);
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_SendPart(HP_Agent pAgent, HP_CONNID dwConnID, const BYTE* pBuffer, int iLength, int iOffset)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->Send(dwConnID, pBuffer, iLength, iOffset);
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_SendPackets(HP_Agent pAgent, HP_CONNID dwConnID, const WSABUF pBuffers[], int iCount)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->SendPackets(dwConnID, pBuffers, iCount);
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_PauseReceive(HP_Agent pAgent, HP_CONNID dwConnID, BOOL bPause)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->PauseReceive(dwConnID, bPause);
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_Disconnect(HP_Agent pAgent, HP_CONNID dwConnID, BOOL bForce)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->Disconnect(dwConnID, bForce);
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_DisconnectLongConnections(HP_Agent pAgent, DWORD dwPeriod, BOOL bForce)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->DisconnectLongConnections(dwPeriod, bForce);
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_DisconnectSilenceConnections(HP_Agent pAgent, DWORD dwPeriod, BOOL bForce)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->DisconnectSilenceConnections(dwPeriod, bForce);
}

/******************************************************************************/
/***************************** Agent 属性访问方法 *****************************/

HPSOCKET_API BOOL __HP_CALL HP_Agent_SetConnectionExtra(HP_Agent pAgent, HP_CONNID dwConnID, PVOID pExtra)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->SetConnectionExtra(dwConnID, pExtra);
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_GetConnectionExtra(HP_Agent pAgent, HP_CONNID dwConnID, PVOID* ppExtra)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->GetConnectionExtra(dwConnID, ppExtra);
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_IsSecure(HP_Agent pAgent)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->IsSecure();
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_HasStarted(HP_Agent pAgent)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->HasStarted();
}

HPSOCKET_API En_HP_ServiceState __HP_CALL HP_Agent_GetState(HP_Agent pAgent)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->GetState();
}

HPSOCKET_API En_HP_SocketError __HP_CALL HP_Agent_GetLastError(HP_Agent pAgent)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->GetLastError();
}

HPSOCKET_API LPCTSTR __HP_CALL HP_Agent_GetLastErrorDesc(HP_Agent pAgent)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->GetLastErrorDesc();
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_GetPendingDataLength(HP_Agent pAgent, HP_CONNID dwConnID, int* piPending)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->GetPendingDataLength(dwConnID, *piPending);
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_IsPauseReceive(HP_Agent pAgent, HP_CONNID dwConnID, BOOL* pbPaused)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->IsPauseReceive(dwConnID, *pbPaused);
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_IsConnected(HP_Agent pAgent, HP_CONNID dwConnID)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->IsConnected(dwConnID);
}

HPSOCKET_API DWORD __HP_CALL HP_Agent_GetConnectionCount(HP_Agent pAgent)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->GetConnectionCount();
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_GetAllConnectionIDs(HP_Agent pAgent, HP_CONNID pIDs[], DWORD* pdwCount)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->GetAllConnectionIDs(pIDs, *pdwCount);
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_GetConnectPeriod(HP_Agent pAgent, HP_CONNID dwConnID, DWORD* pdwPeriod)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->GetConnectPeriod(dwConnID, *pdwPeriod);
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_GetSilencePeriod(HP_Agent pAgent, HP_CONNID dwConnID, DWORD* pdwPeriod)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->GetSilencePeriod(dwConnID, *pdwPeriod);
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_GetLocalAddress(HP_Agent pAgent, HP_CONNID dwConnID, TCHAR lpszAddress[], int* piAddressLen, USHORT* pusPort)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->GetLocalAddress(dwConnID, lpszAddress, *piAddressLen, *pusPort);
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_GetRemoteAddress(HP_Agent pAgent, HP_CONNID dwConnID, TCHAR lpszAddress[], int* piAddressLen, USHORT* pusPort)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->GetRemoteAddress(dwConnID, lpszAddress, *piAddressLen, *pusPort);
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_GetRemoteHost(HP_Agent pAgent, HP_CONNID dwConnID, TCHAR lpszHost[], int* piHostLen, USHORT* pusPort)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->GetRemoteHost(dwConnID, lpszHost, *piHostLen, *pusPort);
}

HPSOCKET_API void __HP_CALL HP_Agent_SetReuseAddressPolicy(HP_Agent pAgent, En_HP_ReuseAddressPolicy enReusePolicy)
{
	C_HP_Object::ToSecond<IAgent>(pAgent)->SetReuseAddressPolicy(enReusePolicy);
}

HPSOCKET_API void __HP_CALL HP_Agent_SetSendPolicy(HP_Agent pAgent, En_HP_SendPolicy enSendPolicy)
{
	C_HP_Object::ToSecond<IAgent>(pAgent)->SetSendPolicy(enSendPolicy);
}

HPSOCKET_API void __HP_CALL HP_Agent_SetOnSendSyncPolicy(HP_Agent pAgent, En_HP_OnSendSyncPolicy enSyncPolicy)
{
	C_HP_Object::ToSecond<IAgent>(pAgent)->SetOnSendSyncPolicy(enSyncPolicy);
}

HPSOCKET_API void __HP_CALL HP_Agent_SetFreeSocketObjLockTime(HP_Agent pAgent, DWORD dwFreeSocketObjLockTime)
{
	C_HP_Object::ToSecond<IAgent>(pAgent)->SetFreeSocketObjLockTime(dwFreeSocketObjLockTime);
}

HPSOCKET_API void __HP_CALL HP_Agent_SetFreeSocketObjPool(HP_Agent pAgent, DWORD dwFreeSocketObjPool)
{
	C_HP_Object::ToSecond<IAgent>(pAgent)->SetFreeSocketObjPool(dwFreeSocketObjPool);
}

HPSOCKET_API void __HP_CALL HP_Agent_SetFreeBufferObjPool(HP_Agent pAgent, DWORD dwFreeBufferObjPool)
{
	C_HP_Object::ToSecond<IAgent>(pAgent)->SetFreeBufferObjPool(dwFreeBufferObjPool);
}

HPSOCKET_API void __HP_CALL HP_Agent_SetFreeSocketObjHold(HP_Agent pAgent, DWORD dwFreeSocketObjHold)
{
	C_HP_Object::ToSecond<IAgent>(pAgent)->SetFreeSocketObjHold(dwFreeSocketObjHold);
}

HPSOCKET_API void __HP_CALL HP_Agent_SetFreeBufferObjHold(HP_Agent pAgent, DWORD dwFreeBufferObjHold)
{
	C_HP_Object::ToSecond<IAgent>(pAgent)->SetFreeBufferObjHold(dwFreeBufferObjHold);
}

HPSOCKET_API void __HP_CALL HP_Agent_SetMaxConnectionCount(HP_Agent pAgent, DWORD dwMaxConnectionCount)
{
	C_HP_Object::ToSecond<IAgent>(pAgent)->SetMaxConnectionCount(dwMaxConnectionCount);
}

HPSOCKET_API void __HP_CALL HP_Agent_SetWorkerThreadCount(HP_Agent pAgent, DWORD dwWorkerThreadCount)
{
	C_HP_Object::ToSecond<IAgent>(pAgent)->SetWorkerThreadCount(dwWorkerThreadCount);
}

HPSOCKET_API void __HP_CALL HP_Agent_SetMarkSilence(HP_Agent pAgent, BOOL bMarkSilence)
{
	C_HP_Object::ToSecond<IAgent>(pAgent)->SetMarkSilence(bMarkSilence);
}

HPSOCKET_API En_HP_ReuseAddressPolicy __HP_CALL HP_Agent_GetReuseAddressPolicy(HP_Agent pAgent)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->GetReuseAddressPolicy();
}

HPSOCKET_API En_HP_SendPolicy __HP_CALL HP_Agent_GetSendPolicy(HP_Agent pAgent)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->GetSendPolicy();
}

HPSOCKET_API En_HP_OnSendSyncPolicy __HP_CALL HP_Agent_GetOnSendSyncPolicy(HP_Agent pAgent)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->GetOnSendSyncPolicy();
}

HPSOCKET_API DWORD __HP_CALL HP_Agent_GetFreeSocketObjLockTime(HP_Agent pAgent)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->GetFreeSocketObjLockTime();
}

HPSOCKET_API DWORD __HP_CALL HP_Agent_GetFreeSocketObjPool(HP_Agent pAgent)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->GetFreeSocketObjPool();
}

HPSOCKET_API DWORD __HP_CALL HP_Agent_GetFreeBufferObjPool(HP_Agent pAgent)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->GetFreeBufferObjPool();
}

HPSOCKET_API DWORD __HP_CALL HP_Agent_GetFreeSocketObjHold(HP_Agent pAgent)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->GetFreeSocketObjHold();
}

HPSOCKET_API DWORD __HP_CALL HP_Agent_GetFreeBufferObjHold(HP_Agent pAgent)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->GetFreeBufferObjHold();
}

HPSOCKET_API DWORD __HP_CALL HP_Agent_GetMaxConnectionCount(HP_Agent pAgent)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->GetMaxConnectionCount();
}

HPSOCKET_API DWORD __HP_CALL HP_Agent_GetWorkerThreadCount(HP_Agent pAgent)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->GetWorkerThreadCount();
}

HPSOCKET_API BOOL __HP_CALL HP_Agent_IsMarkSilence(HP_Agent pAgent)
{
	return C_HP_Object::ToSecond<IAgent>(pAgent)->IsMarkSilence();
}

/**********************************************************************************/
/******************************* TCP Agent 操作方法 *******************************/

HPSOCKET_API BOOL __HP_CALL HP_TcpAgent_SendSmallFile(HP_Agent pAgent, HP_CONNID dwConnID, LPCTSTR lpszFileName, const LPWSABUF pHead, const LPWSABUF pTail)
{
	return C_HP_Object::ToSecond<ITcpAgent>(pAgent)->SendSmallFile(dwConnID, lpszFileName, pHead, pTail);
}

/**********************************************************************************/
/***************************** TCP Agent 属性访问方法 *****************************/

HPSOCKET_API void __HP_CALL HP_TcpAgent_SetSocketBufferSize(HP_TcpAgent pAgent, DWORD dwSocketBufferSize)
{
	C_HP_Object::ToSecond<ITcpAgent>(pAgent)->SetSocketBufferSize(dwSocketBufferSize);
}

HPSOCKET_API void __HP_CALL HP_TcpAgent_SetKeepAliveTime(HP_TcpAgent pAgent, DWORD dwKeepAliveTime)
{
	C_HP_Object::ToSecond<ITcpAgent>(pAgent)->SetKeepAliveTime(dwKeepAliveTime);
}

HPSOCKET_API void __HP_CALL HP_TcpAgent_SetKeepAliveInterval(HP_TcpAgent pAgent, DWORD dwKeepAliveInterval)
{
	C_HP_Object::ToSecond<ITcpAgent>(pAgent)->SetKeepAliveInterval(dwKeepAliveInterval);
}

HPSOCKET_API void __HP_CALL HP_TcpAgent_SetNoDelay(HP_TcpAgent pAgent, BOOL bNoDelay)
{
	C_HP_Object::ToSecond<ITcpAgent>(pAgent)->SetNoDelay(bNoDelay);
}

HPSOCKET_API DWORD __HP_CALL HP_TcpAgent_GetSocketBufferSize(HP_TcpAgent pAgent)
{
	return C_HP_Object::ToSecond<ITcpAgent>(pAgent)->GetSocketBufferSize();
}

HPSOCKET_API DWORD __HP_CALL HP_TcpAgent_GetKeepAliveTime(HP_TcpAgent pAgent)
{
	return C_HP_Object::ToSecond<ITcpAgent>(pAgent)->GetKeepAliveTime();
}

HPSOCKET_API DWORD __HP_CALL HP_TcpAgent_GetKeepAliveInterval(HP_TcpAgent pAgent)
{
	return C_HP_Object::ToSecond<ITcpAgent>(pAgent)->GetKeepAliveInterval();
}

HPSOCKET_API BOOL __HP_CALL HP_TcpAgent_IsNoDelay(HP_TcpAgent pAgent)
{
	return C_HP_Object::ToSecond<ITcpAgent>(pAgent)->IsNoDelay();
}

/******************************************************************************/
/***************************** Client 组件操作方法 *****************************/

HPSOCKET_API BOOL __HP_CALL HP_Client_Start(HP_Client pClient, LPCTSTR lpszRemoteAddress, USHORT usPort, BOOL bAsyncConnect)
{
	return C_HP_Object::ToSecond<IClient>(pClient)->Start(lpszRemoteAddress, usPort, bAsyncConnect);
}

HPSOCKET_API BOOL __HP_CALL HP_Client_StartWithBindAddress(HP_Client pClient, LPCTSTR lpszRemoteAddress, USHORT usPort, BOOL bAsyncConnect, LPCTSTR lpszBindAddress)
{
	return C_HP_Object::ToSecond<IClient>(pClient)->Start(lpszRemoteAddress, usPort, bAsyncConnect, lpszBindAddress);
}

HPSOCKET_API BOOL __HP_CALL HP_Client_StartWithBindAddressAndLocalPort(HP_Client pClient, LPCTSTR lpszRemoteAddress, USHORT usPort, BOOL bAsyncConnect, LPCTSTR lpszBindAddress, USHORT usLocalPort)
{
	return C_HP_Object::ToSecond<IClient>(pClient)->Start(lpszRemoteAddress, usPort, bAsyncConnect, lpszBindAddress, usLocalPort);
}

HPSOCKET_API BOOL __HP_CALL HP_Client_Stop(HP_Client pClient)
{
	return C_HP_Object::ToSecond<IClient>(pClient)->Stop();
}

HPSOCKET_API BOOL __HP_CALL HP_Client_Send(HP_Client pClient, const BYTE* pBuffer, int iLength)
{
	return C_HP_Object::ToSecond<IClient>(pClient)->Send(pBuffer, iLength);
}

HPSOCKET_API BOOL __HP_CALL HP_Client_SendPart(HP_Client pClient, const BYTE* pBuffer, int iLength, int iOffset)
{
	return C_HP_Object::ToSecond<IClient>(pClient)->Send(pBuffer, iLength, iOffset);
}

HPSOCKET_API BOOL __HP_CALL HP_Client_SendPackets(HP_Client pClient, const WSABUF pBuffers[], int iCount)
{
	return C_HP_Object::ToSecond<IClient>(pClient)->SendPackets(pBuffers, iCount);
}

HPSOCKET_API BOOL __HP_CALL HP_Client_PauseReceive(HP_Client pClient, BOOL bPause)
{
	return C_HP_Object::ToSecond<IClient>(pClient)->PauseReceive(bPause);
}

HPSOCKET_API BOOL __HP_CALL HP_Client_Wait(HP_Client pClient, DWORD dwMilliseconds)
{
	return C_HP_Object::ToSecond<IClient>(pClient)->Wait(dwMilliseconds);
}

/******************************************************************************/
/***************************** Client 属性访问方法 *****************************/

HPSOCKET_API void __HP_CALL HP_Client_SetExtra(HP_Client pClient, PVOID pExtra)
{
	C_HP_Object::ToSecond<IClient>(pClient)->SetExtra(pExtra);
}

HPSOCKET_API PVOID __HP_CALL HP_Client_GetExtra(HP_Client pClient)
{
	return C_HP_Object::ToSecond<IClient>(pClient)->GetExtra();
}

HPSOCKET_API BOOL __HP_CALL HP_Client_IsSecure(HP_Client pClient)
{
	return C_HP_Object::ToSecond<IClient>(pClient)->IsSecure();
}

HPSOCKET_API BOOL __HP_CALL HP_Client_HasStarted(HP_Client pClient)
{
	return C_HP_Object::ToSecond<IClient>(pClient)->HasStarted();
}

HPSOCKET_API En_HP_ServiceState	__HP_CALL HP_Client_GetState(HP_Client pClient)
{
	return C_HP_Object::ToSecond<IClient>(pClient)->GetState();
}

HPSOCKET_API En_HP_SocketError	__HP_CALL HP_Client_GetLastError(HP_Client pClient)
{
	return C_HP_Object::ToSecond<IClient>(pClient)->GetLastError();
}

HPSOCKET_API LPCTSTR __HP_CALL HP_Client_GetLastErrorDesc(HP_Client pClient)
{
	return C_HP_Object::ToSecond<IClient>(pClient)->GetLastErrorDesc();
}

HPSOCKET_API HP_CONNID __HP_CALL HP_Client_GetConnectionID(HP_Client pClient)
{
	return C_HP_Object::ToSecond<IClient>(pClient)->GetConnectionID();
}

HPSOCKET_API BOOL __HP_CALL HP_Client_GetLocalAddress(HP_Client pClient, TCHAR lpszAddress[], int* piAddressLen, USHORT* pusPort)
{
	return C_HP_Object::ToSecond<IClient>(pClient)->GetLocalAddress(lpszAddress, *piAddressLen, *pusPort);
}

HPSOCKET_API BOOL __HP_CALL HP_Client_GetRemoteHost(HP_Client pClient, TCHAR lpszHost[], int* piHostLen, USHORT* pusPort)
{
	return C_HP_Object::ToSecond<IClient>(pClient)->GetRemoteHost(lpszHost, *piHostLen, *pusPort);
}

HPSOCKET_API BOOL __HP_CALL HP_Client_GetPendingDataLength(HP_Client pClient, int* piPending)
{
	return C_HP_Object::ToSecond<IClient>(pClient)->GetPendingDataLength(*piPending);
}

HPSOCKET_API BOOL __HP_CALL HP_Client_IsPauseReceive(HP_Client pClient, BOOL* pbPaused)
{
	return C_HP_Object::ToSecond<IClient>(pClient)->IsPauseReceive(*pbPaused);
}

HPSOCKET_API BOOL __HP_CALL HP_Client_IsConnected(HP_Client pClient)
{
	return C_HP_Object::ToSecond<IClient>(pClient)->IsConnected();
}

HPSOCKET_API void __HP_CALL HP_Client_SetReuseAddressPolicy(HP_Client pClient, En_HP_ReuseAddressPolicy enReusePolicy)
{
	C_HP_Object::ToSecond<IClient>(pClient)->SetReuseAddressPolicy(enReusePolicy);
}

HPSOCKET_API void __HP_CALL HP_Client_SetFreeBufferPoolSize(HP_Client pClient, DWORD dwFreeBufferPoolSize)
{
	C_HP_Object::ToSecond<IClient>(pClient)->SetFreeBufferPoolSize(dwFreeBufferPoolSize);
}

HPSOCKET_API void __HP_CALL HP_Client_SetFreeBufferPoolHold(HP_Client pClient, DWORD dwFreeBufferPoolHold)
{
	C_HP_Object::ToSecond<IClient>(pClient)->SetFreeBufferPoolHold(dwFreeBufferPoolHold);
}

HPSOCKET_API En_HP_ReuseAddressPolicy __HP_CALL HP_Client_GetReuseAddressPolicy(HP_Client pClient)
{
	return C_HP_Object::ToSecond<IClient>(pClient)->GetReuseAddressPolicy();
}

HPSOCKET_API DWORD __HP_CALL HP_Client_GetFreeBufferPoolSize(HP_Client pClient)
{
	return C_HP_Object::ToSecond<IClient>(pClient)->GetFreeBufferPoolSize();
}

HPSOCKET_API DWORD __HP_CALL HP_Client_GetFreeBufferPoolHold(HP_Client pClient)
{
	return C_HP_Object::ToSecond<IClient>(pClient)->GetFreeBufferPoolHold();
}

/**********************************************************************************/
/******************************* TCP Client 操作方法 *******************************/

HPSOCKET_API BOOL __HP_CALL HP_TcpClient_SendSmallFile(HP_Client pClient, LPCTSTR lpszFileName, const LPWSABUF pHead, const LPWSABUF pTail)
{
	return C_HP_Object::ToSecond<ITcpClient>(pClient)->SendSmallFile(lpszFileName, pHead, pTail);
}

/**********************************************************************************/
/***************************** TCP Client 属性访问方法 *****************************/

HPSOCKET_API void __HP_CALL HP_TcpClient_SetSocketBufferSize(HP_TcpClient pClient, DWORD dwSocketBufferSize)
{
	C_HP_Object::ToSecond<ITcpClient>(pClient)->SetSocketBufferSize(dwSocketBufferSize);
}

HPSOCKET_API void __HP_CALL HP_TcpClient_SetKeepAliveTime(HP_TcpClient pClient, DWORD dwKeepAliveTime)
{
	C_HP_Object::ToSecond<ITcpClient>(pClient)->SetKeepAliveTime(dwKeepAliveTime);
}

HPSOCKET_API void __HP_CALL HP_TcpClient_SetKeepAliveInterval(HP_TcpClient pClient, DWORD dwKeepAliveInterval)
{
	C_HP_Object::ToSecond<ITcpClient>(pClient)->SetKeepAliveInterval(dwKeepAliveInterval);
}

HPSOCKET_API void __HP_CALL HP_TcpClient_SetNoDelay(HP_TcpClient pClient, BOOL bNoDelay)
{
	C_HP_Object::ToSecond<ITcpClient>(pClient)->SetNoDelay(bNoDelay);
}

HPSOCKET_API DWORD __HP_CALL HP_TcpClient_GetSocketBufferSize(HP_TcpClient pClient)
{
	return C_HP_Object::ToSecond<ITcpClient>(pClient)->GetSocketBufferSize();
}

HPSOCKET_API DWORD __HP_CALL HP_TcpClient_GetKeepAliveTime(HP_TcpClient pClient)
{
	return C_HP_Object::ToSecond<ITcpClient>(pClient)->GetKeepAliveTime();
}

HPSOCKET_API DWORD __HP_CALL HP_TcpClient_GetKeepAliveInterval(HP_TcpClient pClient)
{
	return C_HP_Object::ToSecond<ITcpClient>(pClient)->GetKeepAliveInterval();
}

HPSOCKET_API BOOL __HP_CALL HP_TcpClient_IsNoDelay(HP_TcpClient pClient)
{
	return C_HP_Object::ToSecond<ITcpClient>(pClient)->IsNoDelay();
}

#ifdef _UDP_SUPPORT

/**********************************************************************************/
/***************************** UDP Client 属性访问方法 *****************************/

HPSOCKET_API void __HP_CALL HP_UdpClient_SetMaxDatagramSize(HP_UdpClient pClient, DWORD dwMaxDatagramSize)
{
	C_HP_Object::ToSecond<IUdpClient>(pClient)->SetMaxDatagramSize(dwMaxDatagramSize);
}

HPSOCKET_API DWORD __HP_CALL HP_UdpClient_GetMaxDatagramSize(HP_UdpClient pClient)
{
	return C_HP_Object::ToSecond<IUdpClient>(pClient)->GetMaxDatagramSize();
}

HPSOCKET_API void __HP_CALL HP_UdpClient_SetDetectAttempts(HP_UdpClient pClient, DWORD dwDetectAttempts)
{
	C_HP_Object::ToSecond<IUdpClient>(pClient)->SetDetectAttempts(dwDetectAttempts);
}

HPSOCKET_API void __HP_CALL HP_UdpClient_SetDetectInterval(HP_UdpClient pClient, DWORD dwDetectInterval)
{
	C_HP_Object::ToSecond<IUdpClient>(pClient)->SetDetectInterval(dwDetectInterval);
}

HPSOCKET_API DWORD __HP_CALL HP_UdpClient_GetDetectAttempts(HP_UdpClient pClient)
{
	return C_HP_Object::ToSecond<IUdpClient>(pClient)->GetDetectAttempts();
}

HPSOCKET_API DWORD __HP_CALL HP_UdpClient_GetDetectInterval(HP_UdpClient pClient)
{
	return C_HP_Object::ToSecond<IUdpClient>(pClient)->GetDetectInterval();
}

/**********************************************************************************/
/*************************** UDP ARQ Client 属性访问方法 ***************************/

HPSOCKET_API void __HP_CALL HP_UdpArqClient_SetNoDelay(HP_UdpArqClient pClient, BOOL bNoDelay)
{
	C_HP_Object::ToFirst<IArqClient>(pClient)->SetNoDelay(bNoDelay);
}

HPSOCKET_API void __HP_CALL HP_UdpArqClient_SetTurnoffCongestCtrl(HP_UdpArqClient pClient, BOOL bTurnOff)
{
	C_HP_Object::ToFirst<IArqClient>(pClient)->SetTurnoffCongestCtrl(bTurnOff);
}

HPSOCKET_API void __HP_CALL HP_UdpArqClient_SetFlushInterval(HP_UdpArqClient pClient, DWORD dwFlushInterval)
{
	C_HP_Object::ToFirst<IArqClient>(pClient)->SetFlushInterval(dwFlushInterval);
}

HPSOCKET_API void __HP_CALL HP_UdpArqClient_SetResendByAcks(HP_UdpArqClient pClient, DWORD dwResendByAcks)
{
	C_HP_Object::ToFirst<IArqClient>(pClient)->SetResendByAcks(dwResendByAcks);
}

HPSOCKET_API void __HP_CALL HP_UdpArqClient_SetSendWndSize(HP_UdpArqClient pClient, DWORD dwSendWndSize)
{
	C_HP_Object::ToFirst<IArqClient>(pClient)->SetSendWndSize(dwSendWndSize);
}

HPSOCKET_API void __HP_CALL HP_UdpArqClient_SetRecvWndSize(HP_UdpArqClient pClient, DWORD dwRecvWndSize)
{
	C_HP_Object::ToFirst<IArqClient>(pClient)->SetRecvWndSize(dwRecvWndSize);
}

HPSOCKET_API void __HP_CALL HP_UdpArqClient_SetMinRto(HP_UdpArqClient pClient, DWORD dwMinRto)
{
	C_HP_Object::ToFirst<IArqClient>(pClient)->SetMinRto(dwMinRto);
}

HPSOCKET_API void __HP_CALL HP_UdpArqClient_SetFastLimit(HP_UdpArqClient pClient, DWORD dwFastLimit)
{
	C_HP_Object::ToFirst<IArqClient>(pClient)->SetFastLimit(dwFastLimit);
}

HPSOCKET_API void __HP_CALL HP_UdpArqClient_SetMaxTransUnit(HP_UdpArqClient pClient, DWORD dwMaxTransUnit)
{
	C_HP_Object::ToFirst<IArqClient>(pClient)->SetMaxTransUnit(dwMaxTransUnit);
}

HPSOCKET_API void __HP_CALL HP_UdpArqClient_SetMaxMessageSize(HP_UdpArqClient pClient, DWORD dwMaxMessageSize)
{
	C_HP_Object::ToFirst<IArqClient>(pClient)->SetMaxMessageSize(dwMaxMessageSize);
}

HPSOCKET_API void __HP_CALL HP_UdpArqClient_SetHandShakeTimeout(HP_UdpArqClient pClient, DWORD dwHandShakeTimeout)
{
	C_HP_Object::ToFirst<IArqClient>(pClient)->SetHandShakeTimeout(dwHandShakeTimeout);
}

HPSOCKET_API BOOL __HP_CALL HP_UdpArqClient_IsNoDelay(HP_UdpArqClient pClient)
{
	return C_HP_Object::ToFirst<IArqClient>(pClient)->IsNoDelay();
}

HPSOCKET_API BOOL __HP_CALL HP_UdpArqClient_IsTurnoffCongestCtrl(HP_UdpArqClient pClient)
{
	return C_HP_Object::ToFirst<IArqClient>(pClient)->IsTurnoffCongestCtrl();
}

HPSOCKET_API DWORD __HP_CALL HP_UdpArqClient_GetFlushInterval(HP_UdpArqClient pClient)
{
	return C_HP_Object::ToFirst<IArqClient>(pClient)->GetFlushInterval();
}

HPSOCKET_API DWORD __HP_CALL HP_UdpArqClient_GetResendByAcks(HP_UdpArqClient pClient)
{
	return C_HP_Object::ToFirst<IArqClient>(pClient)->GetResendByAcks();
}

HPSOCKET_API DWORD __HP_CALL HP_UdpArqClient_GetSendWndSize(HP_UdpArqClient pClient)
{
	return C_HP_Object::ToFirst<IArqClient>(pClient)->GetSendWndSize();
}

HPSOCKET_API DWORD __HP_CALL HP_UdpArqClient_GetRecvWndSize(HP_UdpArqClient pClient)
{
	return C_HP_Object::ToFirst<IArqClient>(pClient)->GetRecvWndSize();
}

HPSOCKET_API DWORD __HP_CALL HP_UdpArqClient_GetMinRto(HP_UdpArqClient pClient)
{
	return C_HP_Object::ToFirst<IArqClient>(pClient)->GetMinRto();
}

HPSOCKET_API DWORD __HP_CALL HP_UdpArqClient_GetFastLimit(HP_UdpArqClient pClient)
{
	return C_HP_Object::ToFirst<IArqClient>(pClient)->GetFastLimit();
}

HPSOCKET_API DWORD __HP_CALL HP_UdpArqClient_GetMaxTransUnit(HP_UdpArqClient pClient)
{
	return C_HP_Object::ToFirst<IArqClient>(pClient)->GetMaxTransUnit();
}

HPSOCKET_API DWORD __HP_CALL HP_UdpArqClient_GetMaxMessageSize(HP_UdpArqClient pClient)
{
	return C_HP_Object::ToFirst<IArqClient>(pClient)->GetMaxMessageSize();
}

HPSOCKET_API DWORD __HP_CALL HP_UdpArqClient_GetHandShakeTimeout(HP_UdpArqClient pClient)
{
	return C_HP_Object::ToFirst<IArqClient>(pClient)->GetHandShakeTimeout();
}

HPSOCKET_API BOOL __HP_CALL HP_UdpArqClient_GetWaitingSendMessageCount(HP_UdpArqClient pClient, int* piCount)
{
	return C_HP_Object::ToFirst<IArqClient>(pClient)->GetWaitingSendMessageCount(*piCount);
}

/**********************************************************************************/
/****************************** UDP Cast 属性访问方法 ******************************/

HPSOCKET_API void __HP_CALL HP_UdpCast_SetMaxDatagramSize(HP_UdpCast pCast, DWORD dwMaxDatagramSize)
{
	C_HP_Object::ToSecond<IUdpCast>(pCast)->SetMaxDatagramSize(dwMaxDatagramSize);
}

HPSOCKET_API DWORD __HP_CALL HP_UdpCast_GetMaxDatagramSize(HP_UdpCast pCast)
{
	return C_HP_Object::ToSecond<IUdpCast>(pCast)->GetMaxDatagramSize();
}

HPSOCKET_API BOOL __HP_CALL HP_UdpCast_GetRemoteAddress(HP_UdpCast pCast, TCHAR lpszAddress[], int* piAddressLen, USHORT* pusPort)
{
	return C_HP_Object::ToSecond<IUdpCast>(pCast)->GetRemoteAddress(lpszAddress, *piAddressLen, *pusPort);
}

HPSOCKET_API void __HP_CALL HP_UdpCast_SetCastMode(HP_UdpCast pCast, En_HP_CastMode enCastMode)
{
	C_HP_Object::ToSecond<IUdpCast>(pCast)->SetCastMode(enCastMode);
}

HPSOCKET_API En_HP_CastMode __HP_CALL HP_UdpCast_GetCastMode(HP_UdpCast pCast)
{
	return C_HP_Object::ToSecond<IUdpCast>(pCast)->GetCastMode();
}

HPSOCKET_API void __HP_CALL HP_UdpCast_SetMultiCastTtl(HP_UdpCast pCast, int iMCTtl)
{
	C_HP_Object::ToSecond<IUdpCast>(pCast)->SetMultiCastTtl(iMCTtl);
}

HPSOCKET_API int __HP_CALL HP_UdpCast_GetMultiCastTtl(HP_UdpCast pCast)
{
	return C_HP_Object::ToSecond<IUdpCast>(pCast)->GetMultiCastTtl();
}

HPSOCKET_API void __HP_CALL HP_UdpCast_SetMultiCastLoop(HP_UdpCast pCast, BOOL bMCLoop)
{
	C_HP_Object::ToSecond<IUdpCast>(pCast)->SetMultiCastLoop(bMCLoop);
}

HPSOCKET_API BOOL __HP_CALL HP_UdpCast_IsMultiCastLoop(HP_UdpCast pCast)
{
	return C_HP_Object::ToSecond<IUdpCast>(pCast)->IsMultiCastLoop();
}

/**********************************************************************************/
/****************************** UDP Node 组件操作方法 ******************************/

HPSOCKET_API BOOL __HP_CALL HP_UdpNode_Start(HP_UdpNode pNode, LPCTSTR lpszBindAddress, USHORT usPort)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->Start(lpszBindAddress, usPort);
}

HPSOCKET_API BOOL __HP_CALL HP_UdpNode_StartWithCast(HP_UdpNode pNode, LPCTSTR lpszBindAddress, USHORT usPort, En_HP_CastMode enCastMode, LPCTSTR lpszCastAddress)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->Start(lpszBindAddress, usPort, enCastMode, lpszCastAddress);
}

HPSOCKET_API BOOL __HP_CALL HP_UdpNode_Stop(HP_UdpNode pNode)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->Stop();
}

HPSOCKET_API BOOL __HP_CALL HP_UdpNode_Wait(HP_UdpNode pNode, DWORD dwMilliseconds)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->Wait(dwMilliseconds);
}

HPSOCKET_API BOOL __HP_CALL HP_UdpNode_Send(HP_UdpNode pNode, LPCTSTR lpszRemoteAddress, USHORT usRemotePort, const BYTE* pBuffer, int iLength)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->Send(lpszRemoteAddress, usRemotePort, pBuffer, iLength);
}

HPSOCKET_API BOOL __HP_CALL HP_UdpNode_SendPart(HP_UdpNode pNode, LPCTSTR lpszRemoteAddress, USHORT usRemotePort, const BYTE* pBuffer, int iLength, int iOffset)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->Send(lpszRemoteAddress, usRemotePort, pBuffer, iLength, iOffset);
}

HPSOCKET_API BOOL __HP_CALL HP_UdpNode_SendPackets(HP_UdpNode pNode, LPCTSTR lpszRemoteAddress, USHORT usRemotePort, const WSABUF pBuffers[], int iCount)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->SendPackets(lpszRemoteAddress, usRemotePort, pBuffers, iCount);
}

HPSOCKET_API BOOL __HP_CALL HP_UdpNode_SendCast(HP_UdpNode pNode, const BYTE* pBuffer, int iLength)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->SendCast(pBuffer, iLength);
}

HPSOCKET_API BOOL __HP_CALL HP_UdpNode_SendCastPart(HP_UdpNode pNode, const BYTE* pBuffer, int iLength, int iOffset)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->SendCast(pBuffer, iLength, iOffset);
}

HPSOCKET_API BOOL __HP_CALL HP_UdpNode_SendCastPackets(HP_UdpNode pNode, const WSABUF pBuffers[], int iCount)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->SendCastPackets(pBuffers, iCount);
}

/**********************************************************************************/
/****************************** UDP Node 属性访问方法 ******************************/

HPSOCKET_API void __HP_CALL HP_UdpNode_SetExtra(HP_UdpNode pNode, PVOID pExtra)
{
	C_HP_Object::ToSecond<IUdpNode>(pNode)->SetExtra(pExtra);
}

HPSOCKET_API PVOID __HP_CALL HP_UdpNode_GetExtra(HP_UdpNode pNode)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->GetExtra();
}

HPSOCKET_API BOOL __HP_CALL HP_UdpNode_HasStarted(HP_UdpNode pNode)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->HasStarted();
}

HPSOCKET_API En_HP_ServiceState __HP_CALL HP_UdpNode_GetState(HP_UdpNode pNode)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->GetState();
}

HPSOCKET_API En_HP_SocketError __HP_CALL HP_UdpNode_GetLastError(HP_UdpNode pNode)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->GetLastError();
}

HPSOCKET_API LPCTSTR __HP_CALL HP_UdpNode_GetLastErrorDesc(HP_UdpNode pNode)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->GetLastErrorDesc();
}

HPSOCKET_API BOOL __HP_CALL HP_UdpNode_GetLocalAddress(HP_UdpNode pNode, TCHAR lpszAddress[], int* piAddressLen, USHORT* pusPort)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->GetLocalAddress(lpszAddress, *piAddressLen, *pusPort);
}

HPSOCKET_API BOOL __HP_CALL HP_UdpNode_GetCastAddress(HP_UdpNode pNode, TCHAR lpszAddress[], int* piAddressLen, USHORT* pusPort)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->GetCastAddress(lpszAddress, *piAddressLen, *pusPort);
}

HPSOCKET_API En_HP_CastMode __HP_CALL HP_UdpNode_GetCastMode(HP_UdpNode pNode)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->GetCastMode();
}

HPSOCKET_API BOOL __HP_CALL HP_UdpNode_GetPendingDataLength(HP_UdpNode pNode, int* piPending)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->GetPendingDataLength(*piPending);
}

HPSOCKET_API void __HP_CALL HP_UdpNode_SetMaxDatagramSize(HP_UdpNode pNode, DWORD dwMaxDatagramSize)
{
	C_HP_Object::ToSecond<IUdpNode>(pNode)->SetMaxDatagramSize(dwMaxDatagramSize);
}

HPSOCKET_API DWORD __HP_CALL HP_UdpNode_GetMaxDatagramSize(HP_UdpNode pNode)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->GetMaxDatagramSize();
}

HPSOCKET_API void __HP_CALL HP_UdpNode_SetMultiCastTtl(HP_UdpNode pNode, int iMCTtl)
{
	C_HP_Object::ToSecond<IUdpNode>(pNode)->SetMultiCastTtl(iMCTtl);
}

HPSOCKET_API int __HP_CALL HP_UdpNode_GetMultiCastTtl(HP_UdpNode pNode)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->GetMultiCastTtl();
}

HPSOCKET_API void __HP_CALL HP_UdpNode_SetMultiCastLoop(HP_UdpNode pNode, BOOL bMCLoop)
{
	C_HP_Object::ToSecond<IUdpNode>(pNode)->SetMultiCastLoop(bMCLoop);
}

HPSOCKET_API BOOL __HP_CALL HP_UdpNode_IsMultiCastLoop(HP_UdpNode pNode)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->IsMultiCastLoop();
}

HPSOCKET_API void __HP_CALL HP_UdpNode_SetReuseAddressPolicy(HP_UdpNode pNode, En_HP_ReuseAddressPolicy enReusePolicy)
{
	C_HP_Object::ToSecond<IUdpNode>(pNode)->SetReuseAddressPolicy(enReusePolicy);
}

HPSOCKET_API void __HP_CALL HP_UdpNode_SetWorkerThreadCount(HP_UdpNode pNode, DWORD dwWorkerThreadCount)
{
	C_HP_Object::ToSecond<IUdpNode>(pNode)->SetWorkerThreadCount(dwWorkerThreadCount);
}

HPSOCKET_API void __HP_CALL HP_UdpNode_SetPostReceiveCount(HP_UdpNode pNode, DWORD dwPostReceiveCount)
{
	C_HP_Object::ToSecond<IUdpNode>(pNode)->SetPostReceiveCount(dwPostReceiveCount);
}

HPSOCKET_API void __HP_CALL HP_UdpNode_SetFreeBufferPoolSize(HP_UdpNode pNode, DWORD dwFreeBufferPoolSize)
{
	C_HP_Object::ToSecond<IUdpNode>(pNode)->SetFreeBufferPoolSize(dwFreeBufferPoolSize);
}

HPSOCKET_API void __HP_CALL HP_UdpNode_SetFreeBufferPoolHold(HP_UdpNode pNode, DWORD dwFreeBufferPoolHold)
{
	C_HP_Object::ToSecond<IUdpNode>(pNode)->SetFreeBufferPoolHold(dwFreeBufferPoolHold);
}

HPSOCKET_API En_HP_ReuseAddressPolicy __HP_CALL HP_UdpNode_GetReuseAddressPolicy(HP_UdpNode pNode)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->GetReuseAddressPolicy();
}

HPSOCKET_API DWORD __HP_CALL HP_UdpNode_GetWorkerThreadCount(HP_UdpNode pNode)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->GetWorkerThreadCount();
}

HPSOCKET_API DWORD __HP_CALL HP_UdpNode_GetPostReceiveCount(HP_UdpNode pNode)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->GetPostReceiveCount();
}

HPSOCKET_API DWORD __HP_CALL HP_UdpNode_GetFreeBufferPoolSize(HP_UdpNode pNode)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->GetFreeBufferPoolSize();
}

HPSOCKET_API DWORD __HP_CALL HP_UdpNode_GetFreeBufferPoolHold(HP_UdpNode pNode)
{
	return C_HP_Object::ToSecond<IUdpNode>(pNode)->GetFreeBufferPoolHold();
}

#endif

/***************************************************************************************/
/***************************** TCP Pull Server 组件操作方法 *****************************/

HPSOCKET_API En_HP_FetchResult __HP_CALL HP_TcpPullServer_Fetch(HP_TcpPullServer pServer, HP_CONNID dwConnID, BYTE* pData, int iLength)
{
	return C_HP_Object::ToFirst<IPullSocket>(pServer)->Fetch(dwConnID, pData, iLength);
}

HPSOCKET_API En_HP_FetchResult __HP_CALL HP_TcpPullServer_Peek(HP_TcpPullServer pServer, HP_CONNID dwConnID, BYTE* pData, int iLength)
{
	return C_HP_Object::ToFirst<IPullSocket>(pServer)->Peek(dwConnID, pData, iLength);
}

/***************************************************************************************/
/***************************** TCP Pull Server 属性访问方法 *****************************/

/***************************************************************************************/
/***************************** TCP Pull Agent 组件操作方法 *****************************/

HPSOCKET_API En_HP_FetchResult __HP_CALL HP_TcpPullAgent_Fetch(HP_TcpPullAgent pAgent, HP_CONNID dwConnID, BYTE* pData, int iLength)
{
	return C_HP_Object::ToFirst<IPullSocket>(pAgent)->Fetch(dwConnID, pData, iLength);
}

HPSOCKET_API En_HP_FetchResult __HP_CALL HP_TcpPullAgent_Peek(HP_TcpPullAgent pAgent, HP_CONNID dwConnID, BYTE* pData, int iLength)
{
	return C_HP_Object::ToFirst<IPullSocket>(pAgent)->Peek(dwConnID, pData, iLength);
}

/***************************************************************************************/
/***************************** TCP Pull Agent 属性访问方法 *****************************/

/***************************************************************************************/
/***************************** TCP Pull Client 组件操作方法 *****************************/

HPSOCKET_API En_HP_FetchResult __HP_CALL HP_TcpPullClient_Fetch(HP_TcpPullClient pClient, BYTE* pData, int iLength)
{
	return C_HP_Object::ToFirst<IPullClient>(pClient)->Fetch(pData, iLength);
}

HPSOCKET_API En_HP_FetchResult __HP_CALL HP_TcpPullClient_Peek(HP_TcpPullClient pClient, BYTE* pData, int iLength)
{
	return C_HP_Object::ToFirst<IPullClient>(pClient)->Peek(pData, iLength);
}

/***************************************************************************************/
/***************************** TCP Pull Client 属性访问方法 *****************************/

/***************************************************************************************/
/***************************** TCP Pack Server 组件操作方法 *****************************/

/***************************************************************************************/
/***************************** TCP Pack Server 属性访问方法 *****************************/

HPSOCKET_API void __HP_CALL HP_TcpPackServer_SetMaxPackSize(HP_TcpPackServer pServer, DWORD dwMaxPackSize)
{
	C_HP_Object::ToFirst<IPackSocket>(pServer)->SetMaxPackSize(dwMaxPackSize);
}

HPSOCKET_API void __HP_CALL HP_TcpPackServer_SetPackHeaderFlag(HP_TcpPackServer pServer, USHORT usPackHeaderFlag)
{
	C_HP_Object::ToFirst<IPackSocket>(pServer)->SetPackHeaderFlag(usPackHeaderFlag);
}

HPSOCKET_API DWORD __HP_CALL HP_TcpPackServer_GetMaxPackSize(HP_TcpPackServer pServer)
{
	return C_HP_Object::ToFirst<IPackSocket>(pServer)->GetMaxPackSize();
}

HPSOCKET_API USHORT __HP_CALL HP_TcpPackServer_GetPackHeaderFlag(HP_TcpPackServer pServer)
{
	return C_HP_Object::ToFirst<IPackSocket>(pServer)->GetPackHeaderFlag();
}

/***************************************************************************************/
/***************************** TCP Pack Agent 组件操作方法 *****************************/

/***************************************************************************************/
/***************************** TCP Pack Agent 属性访问方法 *****************************/

HPSOCKET_API void __HP_CALL HP_TcpPackAgent_SetMaxPackSize(HP_TcpPackAgent pAgent, DWORD dwMaxPackSize)
{
	C_HP_Object::ToFirst<IPackSocket>(pAgent)->SetMaxPackSize(dwMaxPackSize);
}

HPSOCKET_API void __HP_CALL HP_TcpPackAgent_SetPackHeaderFlag(HP_TcpPackAgent pAgent, USHORT usPackHeaderFlag)
{
	C_HP_Object::ToFirst<IPackSocket>(pAgent)->SetPackHeaderFlag(usPackHeaderFlag);
}

HPSOCKET_API DWORD __HP_CALL HP_TcpPackAgent_GetMaxPackSize(HP_TcpPackAgent pAgent)
{
	return C_HP_Object::ToFirst<IPackSocket>(pAgent)->GetMaxPackSize();
}

HPSOCKET_API USHORT __HP_CALL HP_TcpPackAgent_GetPackHeaderFlag(HP_TcpPackAgent pAgent)
{
	return C_HP_Object::ToFirst<IPackSocket>(pAgent)->GetPackHeaderFlag();
}

/***************************************************************************************/
/***************************** TCP Pack Client 组件操作方法 *****************************/

/***************************************************************************************/
/***************************** TCP Pack Client 属性访问方法 *****************************/

HPSOCKET_API void __HP_CALL HP_TcpPackClient_SetMaxPackSize(HP_TcpPackClient pClient, DWORD dwMaxPackSize)
{
	C_HP_Object::ToFirst<IPackClient>(pClient)->SetMaxPackSize(dwMaxPackSize);
}

HPSOCKET_API void __HP_CALL HP_TcpPackClient_SetPackHeaderFlag(HP_TcpPackClient pClient, USHORT usPackHeaderFlag)
{
	C_HP_Object::ToFirst<IPackClient>(pClient)->SetPackHeaderFlag(usPackHeaderFlag);
}

HPSOCKET_API DWORD __HP_CALL HP_TcpPackClient_GetMaxPackSize(HP_TcpPackClient pClient)
{
	return C_HP_Object::ToFirst<IPackClient>(pClient)->GetMaxPackSize();
}

HPSOCKET_API USHORT __HP_CALL HP_TcpPackClient_GetPackHeaderFlag(HP_TcpPackClient pClient)
{
	return C_HP_Object::ToFirst<IPackClient>(pClient)->GetPackHeaderFlag();
}

/*****************************************************************************************************************************************************/
/*************************************************************** Global Function Exports *************************************************************/
/*****************************************************************************************************************************************************/

HPSOCKET_API DWORD __HP_CALL HP_GetHPSocketVersion()
{
	return ::GetHPSocketVersion();
}

HPSOCKET_API LPCTSTR __HP_CALL HP_GetSocketErrorDesc(En_HP_SocketError enCode)
{
	return ::GetSocketErrorDesc(enCode);
}

HPSOCKET_API DWORD __HP_CALL SYS_GetLastError()
{
	return ::GetLastError();
}

HPSOCKET_API LPCSTR __HP_CALL SYS_GetLastErrorStr()
{
	return ::GetLastErrorStr();
}

HPSOCKET_API int __HP_CALL SYS_SetSocketOption(SOCKET sock, int level, int name, LPVOID val, int len)
{
	return ::SSO_SetSocketOption(sock, level, name, val, len);
}

HPSOCKET_API int __HP_CALL SYS_GetSocketOption(SOCKET sock, int level, int name, LPVOID val, int* len)
{
	return ::SSO_GetSocketOption(sock, level, name, val, len);
}

HPSOCKET_API int __HP_CALL SYS_IoctlSocket(SOCKET sock, long cmd, ULONG* arg)
{
	return ::SSO_IoctlSocket(sock, cmd, arg);
}

HPSOCKET_API BOOL __HP_CALL SYS_fcntl_SETFL(FD fd, INT fl, BOOL bSet)
{
	return ::fcntl_SETFL(fd, fl, bSet);
}

HPSOCKET_API int __HP_CALL SYS_SSO_NoBlock(SOCKET sock, BOOL bNoBlock)
{
	return ::SSO_NoBlock(sock, bNoBlock);
}

HPSOCKET_API int __HP_CALL SYS_SSO_NoDelay(SOCKET sock, BOOL bNoDelay)
{
	return ::SSO_NoDelay(sock, bNoDelay);
}

HPSOCKET_API int __HP_CALL SYS_SSO_DontLinger(SOCKET sock, BOOL bDont)
{
	return ::SSO_DontLinger(sock, bDont);
}

HPSOCKET_API int __HP_CALL SYS_SSO_Linger(SOCKET sock, USHORT l_onoff, USHORT l_linger)
{
	return ::SSO_Linger(sock, l_onoff, l_linger);
}

HPSOCKET_API int __HP_CALL SYS_SSO_RecvBuffSize(SOCKET sock, int size)
{
	return ::SSO_RecvBuffSize(sock, size);
}

HPSOCKET_API int __HP_CALL SYS_SSO_SendBuffSize(SOCKET sock, int size)
{
	return ::SSO_SendBuffSize(sock, size);
}

HPSOCKET_API int __HP_CALL SYS_SSO_RecvTimeOut(SOCKET sock, int ms)
{
	return ::SSO_RecvTimeOut(sock, ms);
}

HPSOCKET_API int __HP_CALL SYS_SSO_SendTimeOut(SOCKET sock, int ms)
{
	return ::SSO_SendTimeOut(sock, ms);
}

HPSOCKET_API int __HP_CALL SYS_SSO_ReuseAddress(SOCKET sock, En_HP_ReuseAddressPolicy opt)
{
	return ::SSO_ReuseAddress(sock, opt);
}

HPSOCKET_API BOOL __HP_CALL SYS_GetSocketLocalAddress(SOCKET socket, TCHAR lpszAddress[], int* piAddressLen, USHORT* pusPort)
{
	return ::GetSocketLocalAddress(socket, lpszAddress, *piAddressLen, *pusPort);
}

HPSOCKET_API BOOL __HP_CALL SYS_GetSocketRemoteAddress(SOCKET socket, TCHAR lpszAddress[], int* piAddressLen, USHORT* pusPort)
{
	return ::GetSocketRemoteAddress(socket, lpszAddress, *piAddressLen, *pusPort);
}

HPSOCKET_API BOOL __HP_CALL SYS_EnumHostIPAddresses(LPCTSTR lpszHost, En_HP_IPAddrType enType, HP_LPTIPAddr** lpppIPAddr, int* piIPAddrCount)
{
	return ::EnumHostIPAddresses(lpszHost, enType, lpppIPAddr, *piIPAddrCount);
}

HPSOCKET_API BOOL __HP_CALL SYS_FreeHostIPAddresses(HP_LPTIPAddr* lppIPAddr)
{
	return ::FreeHostIPAddresses(lppIPAddr);
}

HPSOCKET_API BOOL __HP_CALL SYS_IsIPAddress(LPCTSTR lpszAddress, En_HP_IPAddrType* penType)
{
	return ::IsIPAddress(lpszAddress, penType);
}

HPSOCKET_API BOOL __HP_CALL SYS_GetIPAddress(LPCTSTR lpszHost, TCHAR lpszIP[], int* piIPLenth, En_HP_IPAddrType* penType)
{
	return ::GetIPAddress(lpszHost, lpszIP, *piIPLenth, *penType);
}

HPSOCKET_API ULONGLONG __HP_CALL SYS_NToH64(ULONGLONG value)
{
	return ::NToH64(value);
}

HPSOCKET_API ULONGLONG __HP_CALL SYS_HToN64(ULONGLONG value)
{
	return ::HToN64(value);
}

HPSOCKET_API USHORT __HP_CALL SYS_SwapEndian16(USHORT value)
{
	return ENDIAN_SWAP_16(value);
}

HPSOCKET_API DWORD __HP_CALL SYS_SwapEndian32(DWORD value)
{
	return ENDIAN_SWAP_32(value);
}

HPSOCKET_API BOOL __HP_CALL SYS_IsLittleEndian()
{
	return ::IsLittleEndian();
}

HPSOCKET_API LPBYTE __HP_CALL SYS_Malloc(int size)
{
	return MALLOC(BYTE, size);
}

HPSOCKET_API LPBYTE __HP_CALL SYS_Realloc(LPBYTE p, int size)
{
	return REALLOC(BYTE, p, size);
}

HPSOCKET_API VOID __HP_CALL SYS_Free(LPBYTE p)
{
	FREE(p);
}

HPSOCKET_API LPVOID __HP_CALL SYS_Calloc(int number, int size)
{
	return CALLOC(number, size);
}

HPSOCKET_API DWORD __HP_CALL SYS_GuessBase64EncodeBound(DWORD dwSrcLen)
{
	return ::GuessBase64EncodeBound(dwSrcLen);
}

HPSOCKET_API DWORD __HP_CALL SYS_GuessBase64DecodeBound(const BYTE* lpszSrc, DWORD dwSrcLen)
{
	return ::GuessBase64DecodeBound(lpszSrc, dwSrcLen);
}

HPSOCKET_API int __HP_CALL SYS_Base64Encode(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD* pdwDestLen)
{
	return ::Base64Encode(lpszSrc, dwSrcLen, lpszDest, *pdwDestLen);
}

HPSOCKET_API int __HP_CALL SYS_Base64Decode(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD* pdwDestLen)
{
	return ::Base64Decode(lpszSrc, dwSrcLen, lpszDest, *pdwDestLen);
}

HPSOCKET_API DWORD __HP_CALL SYS_GuessUrlEncodeBound(const BYTE* lpszSrc, DWORD dwSrcLen)
{
	return ::GuessUrlEncodeBound(lpszSrc, dwSrcLen);
}

HPSOCKET_API DWORD __HP_CALL SYS_GuessUrlDecodeBound(const BYTE* lpszSrc, DWORD dwSrcLen)
{
	return ::GuessUrlDecodeBound(lpszSrc, dwSrcLen);
}

HPSOCKET_API int __HP_CALL SYS_UrlEncode(BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD* pdwDestLen)
{
	return ::UrlEncode(lpszSrc, dwSrcLen, lpszDest, *pdwDestLen);
}

HPSOCKET_API int __HP_CALL SYS_UrlDecode(BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD* pdwDestLen)
{
	return ::UrlDecode(lpszSrc, dwSrcLen, lpszDest, *pdwDestLen);
}

#ifdef _ZLIB_SUPPORT

HPSOCKET_API int __HP_CALL SYS_Compress(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD* pdwDestLen)
{
	return ::Compress(lpszSrc, dwSrcLen, lpszDest, *pdwDestLen);
}

HPSOCKET_API int __HP_CALL SYS_CompressEx(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD* pdwDestLen, int iLevel, int iMethod, int iWindowBits, int iMemLevel, int iStrategy)
{
	return ::CompressEx(lpszSrc, dwSrcLen, lpszDest, *pdwDestLen, iLevel, iMethod, iWindowBits, iMemLevel, iStrategy);
}

HPSOCKET_API int __HP_CALL SYS_Uncompress(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD* pdwDestLen)
{
	return ::Uncompress(lpszSrc, dwSrcLen, lpszDest, *pdwDestLen);
}

HPSOCKET_API int __HP_CALL SYS_UncompressEx(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD* pdwDestLen, int iWindowBits)
{
	return ::UncompressEx(lpszSrc, dwSrcLen, lpszDest, *pdwDestLen, iWindowBits);
}

HPSOCKET_API DWORD __HP_CALL SYS_GuessCompressBound(DWORD dwSrcLen, BOOL bGZip)
{
	return ::GuessCompressBound(dwSrcLen, bGZip);
}

HPSOCKET_API int __HP_CALL SYS_GZipCompress(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD* pdwDestLen)
{
	return ::GZipCompress(lpszSrc, dwSrcLen, lpszDest, *pdwDestLen);
}

HPSOCKET_API int __HP_CALL SYS_GZipUncompress(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD* pdwDestLen)
{
	return ::GZipUncompress(lpszSrc, dwSrcLen, lpszDest, *pdwDestLen);
}

HPSOCKET_API DWORD __HP_CALL SYS_GZipGuessUncompressBound(const BYTE* lpszSrc, DWORD dwSrcLen)
{
	return ::GZipGuessUncompressBound(lpszSrc, dwSrcLen);
}

#endif

#ifdef _BROTLI_SUPPORT

HPSOCKET_API int __HP_CALL SYS_BrotliCompress(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD* pdwDestLen)
{
	return ::BrotliCompress(lpszSrc, dwSrcLen, lpszDest, *pdwDestLen);
}

HPSOCKET_API int __HP_CALL SYS_BrotliCompressEx(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD* pdwDestLen, int iQuality, int iWindow, int iMode)
{
	return ::BrotliCompressEx(lpszSrc, dwSrcLen, lpszDest, *pdwDestLen, iQuality, iWindow , iMode);
}

HPSOCKET_API int __HP_CALL SYS_BrotliUncompress(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD* pdwDestLen)
{
	return ::BrotliUncompress(lpszSrc, dwSrcLen, lpszDest, *pdwDestLen);
}

HPSOCKET_API DWORD __HP_CALL SYS_BrotliGuessCompressBound(DWORD dwSrcLen)
{
	return ::BrotliGuessCompressBound(dwSrcLen);
}

#endif

#ifdef _ICONV_SUPPORT

HPSOCKET_API BOOL __HP_CALL SYS_CharsetConvert(LPCSTR lpszFromCharset, LPCSTR lpszToCharset, LPCSTR lpszInBuf, int iInBufLen, LPSTR lpszOutBuf, int* piOutBufLen)
{
	return ::CharsetConvert(lpszFromCharset, lpszToCharset, lpszInBuf, iInBufLen, lpszOutBuf, *piOutBufLen);
}

HPSOCKET_API BOOL __HP_CALL SYS_GbkToUnicodeEx(const char szSrc[], int iSrcLength, WCHAR szDest[], int* piDestLength)
{
	return ::GbkToUnicodeEx(szSrc, iSrcLength, szDest, *piDestLength);
}

HPSOCKET_API BOOL __HP_CALL SYS_UnicodeToGbkEx(const WCHAR szSrc[], int iSrcLength, char szDest[], int* piDestLength)
{
	return ::UnicodeToGbkEx(szSrc, iSrcLength, szDest, *piDestLength);
}

HPSOCKET_API BOOL __HP_CALL SYS_Utf8ToUnicodeEx(const char szSrc[], int iSrcLength, WCHAR szDest[], int* piDestLength)
{
	return ::Utf8ToUnicodeEx(szSrc, iSrcLength, szDest, *piDestLength);
}

HPSOCKET_API BOOL __HP_CALL SYS_UnicodeToUtf8Ex(const WCHAR szSrc[], int iSrcLength, char szDest[], int* piDestLength)
{
	return ::UnicodeToUtf8Ex(szSrc, iSrcLength, szDest, *piDestLength);
}

HPSOCKET_API BOOL __HP_CALL SYS_GbkToUtf8Ex(const char szSrc[], int iSrcLength, char szDest[], int* piDestLength)
{
	return ::GbkToUtf8Ex(szSrc, iSrcLength, szDest, *piDestLength);
}

HPSOCKET_API BOOL __HP_CALL SYS_Utf8ToGbkEx(const char szSrc[], int iSrcLength, char szDest[], int* piDestLength)
{
	return ::Utf8ToGbkEx(szSrc, iSrcLength, szDest, *piDestLength);
}

HPSOCKET_API BOOL __HP_CALL SYS_GbkToUnicode(const char szSrc[], WCHAR szDest[], int* piDestLength)
{
	return ::GbkToUnicode(szSrc, szDest, *piDestLength);
}

HPSOCKET_API BOOL __HP_CALL SYS_UnicodeToGbk(const WCHAR szSrc[], char szDest[], int* piDestLength)
{
	return ::UnicodeToGbk(szSrc, szDest, *piDestLength);
}

HPSOCKET_API BOOL __HP_CALL SYS_Utf8ToUnicode(const char szSrc[], WCHAR szDest[], int* piDestLength)
{
	return ::Utf8ToUnicode(szSrc, szDest, *piDestLength);
}

HPSOCKET_API BOOL __HP_CALL SYS_UnicodeToUtf8(const WCHAR szSrc[], char szDest[], int* piDestLength)
{
	return ::UnicodeToUtf8(szSrc, szDest, *piDestLength);
}

HPSOCKET_API BOOL __HP_CALL SYS_GbkToUtf8(const char szSrc[], char szDest[], int* piDestLength)
{
	return ::GbkToUtf8(szSrc, szDest, *piDestLength);
}

HPSOCKET_API BOOL __HP_CALL SYS_Utf8ToGbk(const char szSrc[], char szDest[], int* piDestLength)
{
	return ::Utf8ToGbk(szSrc, szDest, *piDestLength);
}

#endif

/*****************************************************************************************************************************************************/
/******************************************************************** HTTP Exports *******************************************************************/
/*****************************************************************************************************************************************************/

#ifdef _HTTP_SUPPORT

typedef C_HP_ObjectT<CHttpServer, IHttpServerListener, sizeof(IComplexHttpResponder)>	C_HP_HttpServer;
typedef C_HP_ObjectT<CHttpAgent, IHttpAgentListener, sizeof(IComplexHttpRequester)>		C_HP_HttpAgent;
typedef C_HP_ObjectT<CHttpClient, IHttpClientListener, sizeof(IHttpRequester)>			C_HP_HttpClient;
typedef C_HP_ObjectT<CHttpSyncClient, IHttpClientListener, sizeof(IHttpSyncRequester)>	C_HP_HttpSyncClient;

/****************************************************/
/***************** HTTP 对象创建函数 *****************/

HPSOCKET_API HP_HttpServer __HP_CALL Create_HP_HttpServer(HP_HttpServerListener pListener)
{
	return (HP_HttpServer)(new C_HP_HttpServer((IHttpServerListener*)pListener));
}

HPSOCKET_API HP_HttpAgent __HP_CALL Create_HP_HttpAgent(HP_HttpAgentListener pListener)
{
	return (HP_HttpAgent)(new C_HP_HttpAgent((IHttpAgentListener*)pListener));
}

HPSOCKET_API HP_HttpClient __HP_CALL Create_HP_HttpClient(HP_HttpClientListener pListener)
{
	return (HP_HttpClient)(new C_HP_HttpClient((IHttpClientListener*)pListener));
}

HPSOCKET_API HP_HttpSyncClient __HP_CALL Create_HP_HttpSyncClient(HP_HttpClientListener pListener)
{
	return (HP_HttpSyncClient)(new C_HP_HttpSyncClient((IHttpClientListener*)pListener));
}

HPSOCKET_API void __HP_CALL Destroy_HP_HttpServer(HP_HttpServer pServer)
{
	delete (C_HP_HttpServer*)pServer;
}

HPSOCKET_API void __HP_CALL Destroy_HP_HttpAgent(HP_HttpAgent pAgent)
{
	delete (C_HP_HttpAgent*)pAgent;
}

HPSOCKET_API void __HP_CALL Destroy_HP_HttpClient(HP_HttpClient pClient)
{
	delete (C_HP_HttpClient*)pClient;
}

HPSOCKET_API void __HP_CALL Destroy_HP_HttpSyncClient(HP_HttpSyncClient pClient)
{
	delete (C_HP_HttpSyncClient*)pClient;
}

HPSOCKET_API HP_HttpServerListener __HP_CALL Create_HP_HttpServerListener()
{
	return (HP_HttpServerListener)(new C_HP_HttpServerListener);
}

HPSOCKET_API HP_HttpAgentListener __HP_CALL Create_HP_HttpAgentListener()
{
	return (HP_HttpAgentListener)(new C_HP_HttpAgentListener);
}

HPSOCKET_API HP_HttpClientListener __HP_CALL Create_HP_HttpClientListener()
{
	return (HP_HttpClientListener)(new C_HP_HttpClientListener);
}

HPSOCKET_API void __HP_CALL Destroy_HP_HttpServerListener(HP_HttpServerListener pListener)
{
	delete (C_HP_HttpServerListener*)pListener;
}

HPSOCKET_API void __HP_CALL Destroy_HP_HttpAgentListener(HP_HttpAgentListener pListener)
{
	delete (C_HP_HttpAgentListener*)pListener;
}

HPSOCKET_API void __HP_CALL Destroy_HP_HttpClientListener(HP_HttpClientListener pListener)
{
	delete (C_HP_HttpClientListener*)pListener;
}

/**********************************************************************************/
/*************************** HTTP Server 回调函数设置方法 **************************/

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpServer_OnMessageBegin(HP_HttpServerListener pListener, HP_FN_HttpServer_OnMessageBegin fn)
{
	((C_HP_HttpServerListener*)pListener)->m_lsnHttp.m_fnOnMessageBegin = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpServer_OnRequestLine(HP_HttpServerListener pListener, HP_FN_HttpServer_OnRequestLine fn)
{
	((C_HP_HttpServerListener*)pListener)->m_lsnHttp.m_fnOnRequestLine = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpServer_OnHeader(HP_HttpServerListener pListener, HP_FN_HttpServer_OnHeader fn)
{
	((C_HP_HttpServerListener*)pListener)->m_lsnHttp.m_fnOnHeader = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpServer_OnHeadersComplete(HP_HttpServerListener pListener, HP_FN_HttpServer_OnHeadersComplete fn)
{
	((C_HP_HttpServerListener*)pListener)->m_lsnHttp.m_fnOnHeadersComplete = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpServer_OnBody(HP_HttpServerListener pListener, HP_FN_HttpServer_OnBody fn)
{
	((C_HP_HttpServerListener*)pListener)->m_lsnHttp.m_fnOnBody = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpServer_OnChunkHeader(HP_HttpServerListener pListener, HP_FN_HttpServer_OnChunkHeader fn)
{
	((C_HP_HttpServerListener*)pListener)->m_lsnHttp.m_fnOnChunkHeader = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpServer_OnChunkComplete(HP_HttpServerListener pListener, HP_FN_HttpServer_OnChunkComplete fn)
{
	((C_HP_HttpServerListener*)pListener)->m_lsnHttp.m_fnOnChunkComplete = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpServer_OnMessageComplete(HP_HttpServerListener pListener, HP_FN_HttpServer_OnMessageComplete fn)
{
	((C_HP_HttpServerListener*)pListener)->m_lsnHttp.m_fnOnMessageComplete = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpServer_OnUpgrade(HP_HttpServerListener pListener, HP_FN_HttpServer_OnUpgrade fn)
{
	((C_HP_HttpServerListener*)pListener)->m_lsnHttp.m_fnOnUpgrade = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpServer_OnParseError(HP_HttpServerListener pListener, HP_FN_HttpServer_OnParseError fn)
{
	((C_HP_HttpServerListener*)pListener)->m_lsnHttp.m_fnOnParseError = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpServer_OnWSMessageHeader(HP_HttpServerListener pListener, HP_FN_HttpServer_OnWSMessageHeader fn)
{
	((C_HP_HttpServerListener*)pListener)->m_lsnHttp.m_fnOnWSMessageHeader = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpServer_OnWSMessageBody(HP_HttpServerListener pListener, HP_FN_HttpServer_OnWSMessageBody fn)
{
	((C_HP_HttpServerListener*)pListener)->m_lsnHttp.m_fnOnWSMessageBody = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpServer_OnWSMessageComplete(HP_HttpServerListener pListener, HP_FN_HttpServer_OnWSMessageComplete fn)
{
	((C_HP_HttpServerListener*)pListener)->m_lsnHttp.m_fnOnWSMessageComplete = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpServer_OnPrepareListen(HP_HttpServerListener pListener, HP_FN_HttpServer_OnPrepareListen fn)
{
	((C_HP_HttpServerListener*)pListener)->m_lsnServer.m_fnOnPrepareListen = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpServer_OnAccept(HP_HttpServerListener pListener, HP_FN_HttpServer_OnAccept fn)
{
	((C_HP_HttpServerListener*)pListener)->m_lsnServer.m_fnOnAccept = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpServer_OnHandShake(HP_HttpServerListener pListener, HP_FN_HttpServer_OnHandShake fn)
{
	((C_HP_HttpServerListener*)pListener)->m_lsnServer.m_fnOnHandShake = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpServer_OnSend(HP_HttpServerListener pListener, HP_FN_HttpServer_OnSend fn)
{
	((C_HP_HttpServerListener*)pListener)->m_lsnServer.m_fnOnSend = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpServer_OnReceive(HP_HttpServerListener pListener, HP_FN_HttpServer_OnReceive fn)
{
	((C_HP_HttpServerListener*)pListener)->m_lsnServer.m_fnOnReceive = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpServer_OnClose(HP_HttpServerListener pListener, HP_FN_HttpServer_OnClose fn)
{
	((C_HP_HttpServerListener*)pListener)->m_lsnServer.m_fnOnClose = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpServer_OnShutdown(HP_HttpServerListener pListener, HP_FN_HttpServer_OnShutdown fn)
{
	((C_HP_HttpServerListener*)pListener)->m_lsnServer.m_fnOnShutdown = fn;
}

/**********************************************************************************/
/**************************** HTTP Agent 回调函数设置方法 **************************/

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpAgent_OnMessageBegin(HP_HttpAgentListener pListener, HP_FN_HttpAgent_OnMessageBegin fn)
{
	((C_HP_HttpAgentListener*)pListener)->m_lsnHttp.m_fnOnMessageBegin = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpAgent_OnStatusLine(HP_HttpAgentListener pListener, HP_FN_HttpAgent_OnStatusLine fn)
{
	((C_HP_HttpAgentListener*)pListener)->m_lsnHttp.m_fnOnStatusLine = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpAgent_OnHeader(HP_HttpAgentListener pListener, HP_FN_HttpAgent_OnHeader fn)
{
	((C_HP_HttpAgentListener*)pListener)->m_lsnHttp.m_fnOnHeader = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpAgent_OnHeadersComplete(HP_HttpAgentListener pListener, HP_FN_HttpAgent_OnHeadersComplete fn)
{
	((C_HP_HttpAgentListener*)pListener)->m_lsnHttp.m_fnOnHeadersComplete = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpAgent_OnBody(HP_HttpAgentListener pListener, HP_FN_HttpAgent_OnBody fn)
{
	((C_HP_HttpAgentListener*)pListener)->m_lsnHttp.m_fnOnBody = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpAgent_OnChunkHeader(HP_HttpAgentListener pListener, HP_FN_HttpAgent_OnChunkHeader fn)
{
	((C_HP_HttpAgentListener*)pListener)->m_lsnHttp.m_fnOnChunkHeader = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpAgent_OnChunkComplete(HP_HttpAgentListener pListener, HP_FN_HttpAgent_OnChunkComplete fn)
{
	((C_HP_HttpAgentListener*)pListener)->m_lsnHttp.m_fnOnChunkComplete = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpAgent_OnMessageComplete(HP_HttpAgentListener pListener, HP_FN_HttpAgent_OnMessageComplete fn)
{
	((C_HP_HttpAgentListener*)pListener)->m_lsnHttp.m_fnOnMessageComplete = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpAgent_OnUpgrade(HP_HttpAgentListener pListener, HP_FN_HttpAgent_OnUpgrade fn)
{
	((C_HP_HttpAgentListener*)pListener)->m_lsnHttp.m_fnOnUpgrade = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpAgent_OnParseError(HP_HttpAgentListener pListener, HP_FN_HttpAgent_OnParseError fn)
{
	((C_HP_HttpAgentListener*)pListener)->m_lsnHttp.m_fnOnParseError = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpAgent_OnWSMessageHeader(HP_HttpAgentListener pListener, HP_FN_HttpAgent_OnWSMessageHeader fn)
{
	((C_HP_HttpAgentListener*)pListener)->m_lsnHttp.m_fnOnWSMessageHeader = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpAgent_OnWSMessageBody(HP_HttpAgentListener pListener, HP_FN_HttpAgent_OnWSMessageBody fn)
{
	((C_HP_HttpAgentListener*)pListener)->m_lsnHttp.m_fnOnWSMessageBody = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpAgent_OnWSMessageComplete(HP_HttpAgentListener pListener, HP_FN_HttpAgent_OnWSMessageComplete fn)
{
	((C_HP_HttpAgentListener*)pListener)->m_lsnHttp.m_fnOnWSMessageComplete = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpAgent_OnPrepareConnect(HP_HttpAgentListener pListener, HP_FN_HttpAgent_OnPrepareConnect fn)
{
	((C_HP_HttpAgentListener*)pListener)->m_lsnAgent.m_fnOnPrepareConnect = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpAgent_OnConnect(HP_HttpAgentListener pListener, HP_FN_HttpAgent_OnConnect fn)
{
	((C_HP_HttpAgentListener*)pListener)->m_lsnAgent.m_fnOnConnect = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpAgent_OnHandShake(HP_HttpAgentListener pListener, HP_FN_HttpAgent_OnHandShake fn)
{
	((C_HP_HttpAgentListener*)pListener)->m_lsnAgent.m_fnOnHandShake = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpAgent_OnReceive(HP_HttpAgentListener pListener, HP_FN_HttpAgent_OnReceive fn)
{
	((C_HP_HttpAgentListener*)pListener)->m_lsnAgent.m_fnOnReceive = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpAgent_OnSend(HP_HttpAgentListener pListener, HP_FN_HttpAgent_OnSend fn)
{
	((C_HP_HttpAgentListener*)pListener)->m_lsnAgent.m_fnOnSend = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpAgent_OnClose(HP_HttpAgentListener pListener, HP_FN_HttpAgent_OnClose fn)
{
	((C_HP_HttpAgentListener*)pListener)->m_lsnAgent.m_fnOnClose = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpAgent_OnShutdown(HP_HttpAgentListener pListener, HP_FN_HttpAgent_OnShutdown fn)
{
	((C_HP_HttpAgentListener*)pListener)->m_lsnAgent.m_fnOnShutdown = fn;
}

/**********************************************************************************/
/*************************** HTTP Client 回调函数设置方法 **************************/

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpClient_OnMessageBegin(HP_HttpClientListener pListener, HP_FN_HttpClient_OnMessageBegin fn)
{
	((C_HP_HttpClientListener*)pListener)->m_lsnHttp.m_fnOnMessageBegin = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpClient_OnStatusLine(HP_HttpClientListener pListener, HP_FN_HttpClient_OnStatusLine fn)
{
	((C_HP_HttpClientListener*)pListener)->m_lsnHttp.m_fnOnStatusLine = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpClient_OnHeader(HP_HttpClientListener pListener, HP_FN_HttpClient_OnHeader fn)
{
	((C_HP_HttpClientListener*)pListener)->m_lsnHttp.m_fnOnHeader = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpClient_OnHeadersComplete(HP_HttpClientListener pListener, HP_FN_HttpClient_OnHeadersComplete fn)
{
	((C_HP_HttpClientListener*)pListener)->m_lsnHttp.m_fnOnHeadersComplete = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpClient_OnBody(HP_HttpClientListener pListener, HP_FN_HttpClient_OnBody fn)
{
	((C_HP_HttpClientListener*)pListener)->m_lsnHttp.m_fnOnBody = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpClient_OnChunkHeader(HP_HttpClientListener pListener, HP_FN_HttpClient_OnChunkHeader fn)
{
	((C_HP_HttpClientListener*)pListener)->m_lsnHttp.m_fnOnChunkHeader = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpClient_OnChunkComplete(HP_HttpClientListener pListener, HP_FN_HttpClient_OnChunkComplete fn)
{
	((C_HP_HttpClientListener*)pListener)->m_lsnHttp.m_fnOnChunkComplete = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpClient_OnMessageComplete(HP_HttpClientListener pListener, HP_FN_HttpClient_OnMessageComplete fn)
{
	((C_HP_HttpClientListener*)pListener)->m_lsnHttp.m_fnOnMessageComplete = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpClient_OnUpgrade(HP_HttpClientListener pListener, HP_FN_HttpClient_OnUpgrade fn)
{
	((C_HP_HttpClientListener*)pListener)->m_lsnHttp.m_fnOnUpgrade = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpClient_OnParseError(HP_HttpClientListener pListener, HP_FN_HttpClient_OnParseError fn)
{
	((C_HP_HttpClientListener*)pListener)->m_lsnHttp.m_fnOnParseError = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpClient_OnWSMessageHeader(HP_HttpClientListener pListener, HP_FN_HttpClient_OnWSMessageHeader fn)
{
	((C_HP_HttpClientListener*)pListener)->m_lsnHttp.m_fnOnWSMessageHeader = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpClient_OnWSMessageBody(HP_HttpClientListener pListener, HP_FN_HttpClient_OnWSMessageBody fn)
{
	((C_HP_HttpClientListener*)pListener)->m_lsnHttp.m_fnOnWSMessageBody = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpClient_OnWSMessageComplete(HP_HttpClientListener pListener, HP_FN_HttpClient_OnWSMessageComplete fn)
{
	((C_HP_HttpClientListener*)pListener)->m_lsnHttp.m_fnOnWSMessageComplete = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpClient_OnPrepareConnect(HP_HttpClientListener pListener, HP_FN_HttpClient_OnPrepareConnect fn)
{
	((C_HP_HttpClientListener*)pListener)->m_lsnClient.m_fnOnPrepareConnect = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpClient_OnConnect(HP_HttpClientListener pListener, HP_FN_HttpClient_OnConnect fn)
{
	((C_HP_HttpClientListener*)pListener)->m_lsnClient.m_fnOnConnect = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpClient_OnHandShake(HP_HttpClientListener pListener, HP_FN_HttpClient_OnHandShake fn)
{
	((C_HP_HttpClientListener*)pListener)->m_lsnClient.m_fnOnHandShake = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpClient_OnReceive(HP_HttpClientListener pListener, HP_FN_HttpClient_OnReceive fn)
{
	((C_HP_HttpClientListener*)pListener)->m_lsnClient.m_fnOnReceive = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpClient_OnSend(HP_HttpClientListener pListener, HP_FN_HttpClient_OnSend fn)
{
	((C_HP_HttpClientListener*)pListener)->m_lsnClient.m_fnOnSend = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_HttpClient_OnClose(HP_HttpClientListener pListener, HP_FN_HttpClient_OnClose fn)
{
	((C_HP_HttpClientListener*)pListener)->m_lsnClient.m_fnOnClose = fn;
}

/**************************************************************************/
/*************************** HTTP Server 操作方法 **************************/

HPSOCKET_API BOOL __HP_CALL HP_HttpServer_SendResponse(HP_HttpServer pServer, HP_CONNID dwConnID, USHORT usStatusCode, LPCSTR lpszDesc, const HP_THeader lpHeaders[], int iHeaderCount, const BYTE* pData, int iLength)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->SendResponse(dwConnID, usStatusCode, lpszDesc, lpHeaders, iHeaderCount, pData, iLength);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpServer_SendLocalFile(HP_HttpServer pServer, HP_CONNID dwConnID, LPCSTR lpszFileName, USHORT usStatusCode, LPCSTR lpszDesc, const HP_THeader lpHeaders[], int iHeaderCount)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->SendLocalFile(dwConnID, lpszFileName, usStatusCode, lpszDesc, lpHeaders, iHeaderCount);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpServer_SendChunkData(HP_HttpServer pServer, HP_CONNID dwConnID, const BYTE* pData, int iLength, LPCSTR lpszExtensions)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->SendChunkData(dwConnID, pData, iLength, lpszExtensions);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpServer_SendWSMessage(HP_HttpServer pServer, HP_CONNID dwConnID, BOOL bFinal, BYTE iReserved, BYTE iOperationCode, const BYTE* pData, int iLength, ULONGLONG ullBodyLen)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->SendWSMessage(dwConnID, bFinal, iReserved, iOperationCode, pData, iLength, ullBodyLen);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpServer_Release(HP_HttpServer pServer, HP_CONNID dwConnID)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->Release(dwConnID);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpServer_StartHttp(HP_HttpServer pServer, HP_CONNID dwConnID)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->StartHttp(dwConnID);
}

/******************************************************************************/
/*************************** HTTP Server 属性访问方法 **************************/

HPSOCKET_API void __HP_CALL HP_HttpServer_SetReleaseDelay(HP_HttpServer pServer, DWORD dwReleaseDelay)
{
	C_HP_Object::ToFirst<IHttpServer>(pServer)->SetReleaseDelay(dwReleaseDelay);
}

HPSOCKET_API DWORD __HP_CALL HP_HttpServer_GetReleaseDelay(HP_HttpServer pServer)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->GetReleaseDelay();
}

HPSOCKET_API USHORT __HP_CALL HP_HttpServer_GetUrlFieldSet(HP_HttpServer pServer, HP_CONNID dwConnID)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->GetUrlFieldSet(dwConnID);
}


HPSOCKET_API LPCSTR __HP_CALL HP_HttpServer_GetUrlField(HP_HttpServer pServer, HP_CONNID dwConnID, En_HP_HttpUrlField enField)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->GetUrlField(dwConnID, enField);
}

HPSOCKET_API LPCSTR __HP_CALL HP_HttpServer_GetMethod(HP_HttpServer pServer, HP_CONNID dwConnID)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->GetMethod(dwConnID);
}

HPSOCKET_API void __HP_CALL HP_HttpServer_SetLocalVersion(HP_HttpServer pServer, En_HP_HttpVersion usVersion)
{
	C_HP_Object::ToFirst<IHttpServer>(pServer)->SetLocalVersion(usVersion);
}

HPSOCKET_API En_HP_HttpVersion __HP_CALL HP_HttpServer_GetLocalVersion(HP_HttpServer pServer)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->GetLocalVersion();
}

HPSOCKET_API BOOL __HP_CALL HP_HttpServer_IsUpgrade(HP_HttpServer pServer, HP_CONNID dwConnID)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->IsUpgrade(dwConnID);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpServer_IsKeepAlive(HP_HttpServer pServer, HP_CONNID dwConnID)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->IsKeepAlive(dwConnID);
}

HPSOCKET_API USHORT __HP_CALL HP_HttpServer_GetVersion(HP_HttpServer pServer, HP_CONNID dwConnID)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->GetVersion(dwConnID);
}

HPSOCKET_API LPCSTR __HP_CALL HP_HttpServer_GetHost(HP_HttpServer pServer, HP_CONNID dwConnID)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->GetHost(dwConnID);
}

HPSOCKET_API ULONGLONG __HP_CALL HP_HttpServer_GetContentLength(HP_HttpServer pServer, HP_CONNID dwConnID)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->GetContentLength(dwConnID);
}

HPSOCKET_API LPCSTR __HP_CALL HP_HttpServer_GetContentType(HP_HttpServer pServer, HP_CONNID dwConnID)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->GetContentType(dwConnID);
}

HPSOCKET_API LPCSTR __HP_CALL HP_HttpServer_GetContentEncoding(HP_HttpServer pServer, HP_CONNID dwConnID)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->GetContentEncoding(dwConnID);
}

HPSOCKET_API LPCSTR __HP_CALL HP_HttpServer_GetTransferEncoding(HP_HttpServer pServer, HP_CONNID dwConnID)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->GetTransferEncoding(dwConnID);
}

HPSOCKET_API En_HP_HttpUpgradeType __HP_CALL HP_HttpServer_GetUpgradeType(HP_HttpServer pServer, HP_CONNID dwConnID)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->GetUpgradeType(dwConnID);
}

HPSOCKET_API USHORT __HP_CALL HP_HttpServer_GetParseErrorCode(HP_HttpServer pServer, HP_CONNID dwConnID, LPCSTR* lpszErrorDesc)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->GetParseErrorCode(dwConnID, lpszErrorDesc);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpServer_GetHeader(HP_HttpServer pServer, HP_CONNID dwConnID, LPCSTR lpszName, LPCSTR* lpszValue)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->GetHeader(dwConnID, lpszName, lpszValue);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpServer_GetHeaders(HP_HttpServer pServer, HP_CONNID dwConnID, LPCSTR lpszName, LPCSTR lpszValue[], DWORD* pdwCount)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->GetHeaders(dwConnID, lpszName, lpszValue, *pdwCount);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpServer_GetAllHeaders(HP_HttpServer pServer, HP_CONNID dwConnID, HP_THeader lpHeaders[], DWORD* pdwCount)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->GetAllHeaders(dwConnID, lpHeaders, *pdwCount);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpServer_GetAllHeaderNames(HP_HttpServer pServer, HP_CONNID dwConnID, LPCSTR lpszName[], DWORD* pdwCount)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->GetAllHeaderNames(dwConnID, lpszName, *pdwCount);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpServer_GetCookie(HP_HttpServer pServer, HP_CONNID dwConnID, LPCSTR lpszName, LPCSTR* lpszValue)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->GetCookie(dwConnID, lpszName, lpszValue);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpServer_GetAllCookies(HP_HttpServer pServer, HP_CONNID dwConnID, HP_TCookie lpCookies[], DWORD* pdwCount)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->GetAllCookies(dwConnID, lpCookies, *pdwCount);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpServer_GetWSMessageState(HP_HttpServer pServer, HP_CONNID dwConnID, BOOL* lpbFinal, BYTE* lpiReserved, BYTE* lpiOperationCode, LPCBYTE* lpszMask, ULONGLONG* lpullBodyLen, ULONGLONG* lpullBodyRemain)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->GetWSMessageState(dwConnID, lpbFinal, lpiReserved, lpiOperationCode, lpszMask, lpullBodyLen, lpullBodyRemain);
}

HPSOCKET_API void __HP_CALL HP_HttpServer_SetHttpAutoStart(HP_HttpServer pServer, BOOL bAutoStart)
{
	C_HP_Object::ToFirst<IHttpServer>(pServer)->SetHttpAutoStart(bAutoStart);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpServer_IsHttpAutoStart(HP_HttpServer pServer)
{
	return C_HP_Object::ToFirst<IHttpServer>(pServer)->IsHttpAutoStart();
}

/**************************************************************************/
/*************************** HTTP Agent 操作方法 ***************************/

HPSOCKET_API BOOL __HP_CALL HP_HttpAgent_SendRequest(HP_HttpAgent pAgent, HP_CONNID dwConnID, LPCSTR lpszMethod, LPCSTR lpszPath, const HP_THeader lpHeaders[], int iHeaderCount, const BYTE* pBody, int iLength)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->SendRequest(dwConnID, lpszMethod, lpszPath, lpHeaders, iHeaderCount, pBody, iLength);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpAgent_SendLocalFile(HP_HttpAgent pAgent, HP_CONNID dwConnID, LPCSTR lpszFileName, LPCSTR lpszMethod, LPCSTR lpszPath, const HP_THeader lpHeaders[], int iHeaderCount)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->SendLocalFile(dwConnID, lpszFileName, lpszMethod, lpszPath, lpHeaders, iHeaderCount);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpAgent_SendChunkData(HP_HttpAgent pAgent, HP_CONNID dwConnID, const BYTE* pData, int iLength, LPCSTR lpszExtensions)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->SendChunkData(dwConnID, pData, iLength, lpszExtensions);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpAgent_SendPost(HP_HttpAgent pAgent, HP_CONNID dwConnID, LPCSTR lpszPath, const HP_THeader lpHeaders[], int iHeaderCount, const BYTE* pBody, int iLength)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->SendPost(dwConnID, lpszPath, lpHeaders, iHeaderCount, pBody, iLength);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpAgent_SendPut(HP_HttpAgent pAgent, HP_CONNID dwConnID, LPCSTR lpszPath, const HP_THeader lpHeaders[], int iHeaderCount, const BYTE* pBody, int iLength)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->SendPut(dwConnID, lpszPath, lpHeaders, iHeaderCount, pBody, iLength);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpAgent_SendPatch(HP_HttpAgent pAgent, HP_CONNID dwConnID, LPCSTR lpszPath, const HP_THeader lpHeaders[], int iHeaderCount, const BYTE* pBody, int iLength)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->SendPatch(dwConnID, lpszPath, lpHeaders, iHeaderCount, pBody, iLength);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpAgent_SendGet(HP_HttpAgent pAgent, HP_CONNID dwConnID, LPCSTR lpszPath, const HP_THeader lpHeaders[], int iHeaderCount)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->SendGet(dwConnID, lpszPath, lpHeaders, iHeaderCount);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpAgent_SendDelete(HP_HttpAgent pAgent, HP_CONNID dwConnID, LPCSTR lpszPath, const HP_THeader lpHeaders[], int iHeaderCount)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->SendDelete(dwConnID, lpszPath, lpHeaders, iHeaderCount);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpAgent_SendHead(HP_HttpAgent pAgent, HP_CONNID dwConnID, LPCSTR lpszPath, const HP_THeader lpHeaders[], int iHeaderCount)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->SendHead(dwConnID, lpszPath, lpHeaders, iHeaderCount);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpAgent_SendTrace(HP_HttpAgent pAgent, HP_CONNID dwConnID, LPCSTR lpszPath, const HP_THeader lpHeaders[], int iHeaderCount)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->SendTrace(dwConnID, lpszPath, lpHeaders, iHeaderCount);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpAgent_SendOptions(HP_HttpAgent pAgent, HP_CONNID dwConnID, LPCSTR lpszPath, const HP_THeader lpHeaders[], int iHeaderCount)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->SendOptions(dwConnID, lpszPath, lpHeaders, iHeaderCount);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpAgent_SendConnect(HP_HttpAgent pAgent, HP_CONNID dwConnID, LPCSTR lpszHost, const HP_THeader lpHeaders[], int iHeaderCount)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->SendConnect(dwConnID, lpszHost, lpHeaders, iHeaderCount);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpAgent_SendWSMessage(HP_HttpAgent pAgent, HP_CONNID dwConnID, BOOL bFinal, BYTE iReserved, BYTE iOperationCode, const BYTE lpszMask[4], const BYTE* pData, int iLength, ULONGLONG ullBodyLen)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->SendWSMessage(dwConnID, bFinal, iReserved, iOperationCode, lpszMask, pData, iLength, ullBodyLen);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpAgent_StartHttp(HP_HttpAgent pAgent, HP_CONNID dwConnID)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->StartHttp(dwConnID);
}

/******************************************************************************/
/*************************** HTTP Agent 属性访问方法 ***************************/

HPSOCKET_API USHORT __HP_CALL HP_HttpAgent_GetStatusCode(HP_HttpAgent pAgent, HP_CONNID dwConnID)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->GetStatusCode(dwConnID);
}

HPSOCKET_API void __HP_CALL HP_HttpAgent_SetLocalVersion(HP_HttpAgent pAgent, En_HP_HttpVersion usVersion)
{
	C_HP_Object::ToFirst<IHttpAgent>(pAgent)->SetLocalVersion(usVersion);
}

HPSOCKET_API En_HP_HttpVersion __HP_CALL HP_HttpAgent_GetLocalVersion(HP_HttpAgent pAgent)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->GetLocalVersion();
}

HPSOCKET_API BOOL __HP_CALL HP_HttpAgent_IsUpgrade(HP_HttpAgent pAgent, HP_CONNID dwConnID)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->IsUpgrade(dwConnID);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpAgent_IsKeepAlive(HP_HttpAgent pAgent, HP_CONNID dwConnID)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->IsKeepAlive(dwConnID);
}

HPSOCKET_API USHORT __HP_CALL HP_HttpAgent_GetVersion(HP_HttpAgent pAgent, HP_CONNID dwConnID)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->GetVersion(dwConnID);
}

HPSOCKET_API ULONGLONG __HP_CALL HP_HttpAgent_GetContentLength(HP_HttpAgent pAgent, HP_CONNID dwConnID)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->GetContentLength(dwConnID);
}

HPSOCKET_API LPCSTR __HP_CALL HP_HttpAgent_GetContentType(HP_HttpAgent pAgent, HP_CONNID dwConnID)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->GetContentType(dwConnID);
}

HPSOCKET_API LPCSTR __HP_CALL HP_HttpAgent_GetContentEncoding(HP_HttpAgent pAgent, HP_CONNID dwConnID)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->GetContentEncoding(dwConnID);
}

HPSOCKET_API LPCSTR __HP_CALL HP_HttpAgent_GetTransferEncoding(HP_HttpAgent pAgent, HP_CONNID dwConnID)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->GetTransferEncoding(dwConnID);
}

HPSOCKET_API En_HP_HttpUpgradeType __HP_CALL HP_HttpAgent_GetUpgradeType(HP_HttpAgent pAgent, HP_CONNID dwConnID)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->GetUpgradeType(dwConnID);
}

HPSOCKET_API USHORT __HP_CALL HP_HttpAgent_GetParseErrorCode(HP_HttpAgent pAgent, HP_CONNID dwConnID, LPCSTR* lpszErrorDesc)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->GetParseErrorCode(dwConnID, lpszErrorDesc);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpAgent_GetHeader(HP_HttpAgent pAgent, HP_CONNID dwConnID, LPCSTR lpszName, LPCSTR* lpszValue)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->GetHeader(dwConnID, lpszName, lpszValue);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpAgent_GetHeaders(HP_HttpAgent pAgent, HP_CONNID dwConnID, LPCSTR lpszName, LPCSTR lpszValue[], DWORD* pdwCount)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->GetHeaders(dwConnID, lpszName, lpszValue, *pdwCount);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpAgent_GetAllHeaders(HP_HttpAgent pAgent, HP_CONNID dwConnID, HP_THeader lpHeaders[], DWORD* pdwCount)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->GetAllHeaders(dwConnID, lpHeaders, *pdwCount);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpAgent_GetAllHeaderNames(HP_HttpAgent pAgent, HP_CONNID dwConnID, LPCSTR lpszName[], DWORD* pdwCount)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->GetAllHeaderNames(dwConnID, lpszName, *pdwCount);
}

HPSOCKET_API void __HP_CALL HP_HttpAgent_SetUseCookie(HP_HttpAgent pAgent, BOOL bUseCookie)
{
	C_HP_Object::ToFirst<IHttpAgent>(pAgent)->SetUseCookie(bUseCookie);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpAgent_IsUseCookie(HP_HttpAgent pAgent)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->IsUseCookie();
}

HPSOCKET_API BOOL __HP_CALL HP_HttpAgent_GetCookie(HP_HttpAgent pAgent, HP_CONNID dwConnID, LPCSTR lpszName, LPCSTR* lpszValue)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->GetCookie(dwConnID, lpszName, lpszValue);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpAgent_GetAllCookies(HP_HttpAgent pAgent, HP_CONNID dwConnID, HP_TCookie lpCookies[], DWORD* pdwCount)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->GetAllCookies(dwConnID, lpCookies, *pdwCount);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpAgent_GetWSMessageState(HP_HttpAgent pAgent, HP_CONNID dwConnID, BOOL* lpbFinal, BYTE* lpiReserved, BYTE* lpiOperationCode, LPCBYTE* lpszMask, ULONGLONG* lpullBodyLen, ULONGLONG* lpullBodyRemain)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->GetWSMessageState(dwConnID, lpbFinal, lpiReserved, lpiOperationCode, lpszMask, lpullBodyLen, lpullBodyRemain);
}

HPSOCKET_API void __HP_CALL HP_HttpAgent_SetHttpAutoStart(HP_HttpAgent pAgent, BOOL bAutoStart)
{
	C_HP_Object::ToFirst<IHttpAgent>(pAgent)->SetHttpAutoStart(bAutoStart);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpAgent_IsHttpAutoStart(HP_HttpAgent pAgent)
{
	return C_HP_Object::ToFirst<IHttpAgent>(pAgent)->IsHttpAutoStart();
}

/**************************************************************************/
/*************************** HTTP Client 操作方法 **************************/

HPSOCKET_API BOOL __HP_CALL HP_HttpClient_SendRequest(HP_HttpClient pClient, LPCSTR lpszMethod, LPCSTR lpszPath, const HP_THeader lpHeaders[], int iHeaderCount, const BYTE* pBody, int iLength)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->SendRequest(lpszMethod, lpszPath, lpHeaders, iHeaderCount, pBody, iLength);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpClient_SendLocalFile(HP_HttpClient pClient, LPCSTR lpszFileName, LPCSTR lpszMethod, LPCSTR lpszPath, const HP_THeader lpHeaders[], int iHeaderCount)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->SendLocalFile(lpszFileName, lpszMethod, lpszPath, lpHeaders, iHeaderCount);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpClient_SendChunkData(HP_HttpClient pClient, const BYTE* pData, int iLength, LPCSTR lpszExtensions)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->SendChunkData(pData, iLength, lpszExtensions);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpClient_SendPost(HP_HttpClient pClient, LPCSTR lpszPath, const HP_THeader lpHeaders[], int iHeaderCount, const BYTE* pBody, int iLength)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->SendPost(lpszPath, lpHeaders, iHeaderCount, pBody, iLength);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpClient_SendPut(HP_HttpClient pClient, LPCSTR lpszPath, const HP_THeader lpHeaders[], int iHeaderCount, const BYTE* pBody, int iLength)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->SendPut(lpszPath, lpHeaders, iHeaderCount, pBody, iLength);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpClient_SendPatch(HP_HttpClient pClient, LPCSTR lpszPath, const HP_THeader lpHeaders[], int iHeaderCount, const BYTE* pBody, int iLength)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->SendPatch(lpszPath, lpHeaders, iHeaderCount, pBody, iLength);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpClient_SendGet(HP_HttpClient pClient, LPCSTR lpszPath, const HP_THeader lpHeaders[], int iHeaderCount)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->SendGet(lpszPath, lpHeaders, iHeaderCount);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpClient_SendDelete(HP_HttpClient pClient, LPCSTR lpszPath, const HP_THeader lpHeaders[], int iHeaderCount)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->SendDelete(lpszPath, lpHeaders, iHeaderCount);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpClient_SendHead(HP_HttpClient pClient, LPCSTR lpszPath, const HP_THeader lpHeaders[], int iHeaderCount)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->SendHead(lpszPath, lpHeaders, iHeaderCount);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpClient_SendTrace(HP_HttpClient pClient, LPCSTR lpszPath, const HP_THeader lpHeaders[], int iHeaderCount)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->SendTrace(lpszPath, lpHeaders, iHeaderCount);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpClient_SendOptions(HP_HttpClient pClient, LPCSTR lpszPath, const HP_THeader lpHeaders[], int iHeaderCount)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->SendOptions(lpszPath, lpHeaders, iHeaderCount);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpClient_SendConnect(HP_HttpClient pClient, LPCSTR lpszHost, const HP_THeader lpHeaders[], int iHeaderCount)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->SendConnect(lpszHost, lpHeaders, iHeaderCount);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpClient_SendWSMessage(HP_HttpClient pClient, BOOL bFinal, BYTE iReserved, BYTE iOperationCode, const BYTE lpszMask[4], const BYTE* pData, int iLength, ULONGLONG ullBodyLen)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->SendWSMessage(bFinal, iReserved, iOperationCode, lpszMask, pData, iLength, ullBodyLen);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpClient_StartHttp(HP_HttpClient pClient)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->StartHttp();
}

/******************************************************************************/
/*************************** HTTP Client 属性访问方法 **************************/

HPSOCKET_API USHORT __HP_CALL HP_HttpClient_GetStatusCode(HP_HttpClient pClient)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->GetStatusCode();
}

HPSOCKET_API void __HP_CALL HP_HttpClient_SetLocalVersion(HP_HttpClient pClient, En_HP_HttpVersion usVersion)
{
	C_HP_Object::ToFirst<IHttpClient>(pClient)->SetLocalVersion(usVersion);
}

HPSOCKET_API En_HP_HttpVersion __HP_CALL HP_HttpClient_GetLocalVersion(HP_HttpClient pClient)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->GetLocalVersion();
}

HPSOCKET_API BOOL __HP_CALL HP_HttpClient_IsUpgrade(HP_HttpClient pClient)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->IsUpgrade();
}

HPSOCKET_API BOOL __HP_CALL HP_HttpClient_IsKeepAlive(HP_HttpClient pClient)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->IsKeepAlive();
}

HPSOCKET_API USHORT __HP_CALL HP_HttpClient_GetVersion(HP_HttpClient pClient)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->GetVersion();
}

HPSOCKET_API ULONGLONG __HP_CALL HP_HttpClient_GetContentLength(HP_HttpClient pClient)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->GetContentLength();
}

HPSOCKET_API LPCSTR __HP_CALL HP_HttpClient_GetContentType(HP_HttpClient pClient)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->GetContentType();
}

HPSOCKET_API LPCSTR __HP_CALL HP_HttpClient_GetContentEncoding(HP_HttpClient pClient)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->GetContentEncoding();
}

HPSOCKET_API LPCSTR __HP_CALL HP_HttpClient_GetTransferEncoding(HP_HttpClient pClient)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->GetTransferEncoding();
}

HPSOCKET_API En_HP_HttpUpgradeType __HP_CALL HP_HttpClient_GetUpgradeType(HP_HttpClient pClient)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->GetUpgradeType();
}

HPSOCKET_API USHORT __HP_CALL HP_HttpClient_GetParseErrorCode(HP_HttpClient pClient, LPCSTR* lpszErrorDesc)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->GetParseErrorCode(lpszErrorDesc);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpClient_GetHeader(HP_HttpClient pClient, LPCSTR lpszName, LPCSTR* lpszValue)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->GetHeader(lpszName, lpszValue);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpClient_GetHeaders(HP_HttpClient pClient, LPCSTR lpszName, LPCSTR lpszValue[], DWORD* pdwCount)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->GetHeaders(lpszName, lpszValue, *pdwCount);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpClient_GetAllHeaders(HP_HttpClient pClient, HP_THeader lpHeaders[], DWORD* pdwCount)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->GetAllHeaders(lpHeaders, *pdwCount);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpClient_GetAllHeaderNames(HP_HttpClient pClient, LPCSTR lpszName[], DWORD* pdwCount)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->GetAllHeaderNames(lpszName, *pdwCount);
}

HPSOCKET_API void __HP_CALL HP_HttpClient_SetUseCookie(HP_HttpClient pClient, BOOL bUseCookie)
{
	C_HP_Object::ToFirst<IHttpClient>(pClient)->SetUseCookie(bUseCookie);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpClient_IsUseCookie(HP_HttpClient pClient)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->IsUseCookie();
}

HPSOCKET_API BOOL __HP_CALL HP_HttpClient_GetCookie(HP_HttpClient pClient, LPCSTR lpszName, LPCSTR* lpszValue)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->GetCookie(lpszName, lpszValue);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpClient_GetAllCookies(HP_HttpClient pClient, HP_TCookie lpCookies[], DWORD* pdwCount)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->GetAllCookies(lpCookies, *pdwCount);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpClient_GetWSMessageState(HP_HttpClient pClient, BOOL* lpbFinal, BYTE* lpiReserved, BYTE* lpiOperationCode, LPCBYTE* lpszMask, ULONGLONG* lpullBodyLen, ULONGLONG* lpullBodyRemain)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->GetWSMessageState(lpbFinal, lpiReserved, lpiOperationCode, lpszMask, lpullBodyLen, lpullBodyRemain);
}

HPSOCKET_API void __HP_CALL HP_HttpClient_SetHttpAutoStart(HP_HttpClient pClient, BOOL bAutoStart)
{
	C_HP_Object::ToFirst<IHttpClient>(pClient)->SetHttpAutoStart(bAutoStart);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpClient_IsHttpAutoStart(HP_HttpClient pClient)
{
	return C_HP_Object::ToFirst<IHttpClient>(pClient)->IsHttpAutoStart();
}

/**************************************************************************/
/************************ HTTP Sync Client 操作方法 ************************/

HPSOCKET_API BOOL __HP_CALL HP_HttpSyncClient_OpenUrl(HP_HttpSyncClient pClient, LPCSTR lpszMethod, LPCSTR lpszUrl, const HP_THeader lpHeaders[], int iHeaderCount, const BYTE* pBody, int iLength, BOOL bForceReconnect)
{
	return C_HP_Object::ToFirst<IHttpSyncClient>(pClient)->OpenUrl(lpszMethod, lpszUrl, lpHeaders, iHeaderCount, pBody, iLength, bForceReconnect);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpSyncClient_CleanupRequestResult(HP_HttpSyncClient pClient)
{
	return C_HP_Object::ToFirst<IHttpSyncClient>(pClient)->CleanupRequestResult();
}

/**************************************************************************/
/********************** HTTP Sync Client 属性访问方法 **********************/

HPSOCKET_API void __HP_CALL HP_HttpSyncClient_SetConnectTimeout(HP_HttpSyncClient pClient, DWORD dwConnectTimeout)
{
	C_HP_Object::ToFirst<IHttpSyncClient>(pClient)->SetConnectTimeout(dwConnectTimeout);
}

HPSOCKET_API void __HP_CALL HP_HttpSyncClient_SetRequestTimeout(HP_HttpSyncClient pClient, DWORD dwRequestTimeout)
{
	C_HP_Object::ToFirst<IHttpSyncClient>(pClient)->SetRequestTimeout(dwRequestTimeout);
}

HPSOCKET_API DWORD __HP_CALL HP_HttpSyncClient_GetConnectTimeout(HP_HttpSyncClient pClient)
{
	return C_HP_Object::ToFirst<IHttpSyncClient>(pClient)->GetConnectTimeout();
}

HPSOCKET_API DWORD __HP_CALL HP_HttpSyncClient_GetRequestTimeout(HP_HttpSyncClient pClient)
{
	return C_HP_Object::ToFirst<IHttpSyncClient>(pClient)->GetRequestTimeout();
}

HPSOCKET_API BOOL __HP_CALL HP_HttpSyncClient_GetResponseBody(HP_HttpSyncClient pClient, LPCBYTE* lpszBody, int* piLength)
{
	return C_HP_Object::ToFirst<IHttpSyncClient>(pClient)->GetResponseBody(lpszBody, piLength);
}

/**************************************************************************/
/*************************** HTTP Cookie 管理方法 **************************/

HPSOCKET_API BOOL __HP_CALL HP_HttpCookie_MGR_LoadFromFile(LPCSTR lpszFile, BOOL bKeepExists)
{
	return g_CookieMgr.LoadFromFile(lpszFile, bKeepExists);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpCookie_MGR_SaveToFile(LPCSTR lpszFile, BOOL bKeepExists)
{
	return g_CookieMgr.SaveToFile(lpszFile, bKeepExists);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpCookie_MGR_ClearCookies(LPCSTR lpszDomain, LPCSTR lpszPath)
{
	return g_CookieMgr.ClearCookies(lpszDomain, lpszPath);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpCookie_MGR_RemoveExpiredCookies(LPCSTR lpszDomain, LPCSTR lpszPath)
{
	return g_CookieMgr.RemoveExpiredCookies(lpszDomain, lpszPath);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpCookie_MGR_SetCookie(LPCSTR lpszName, LPCSTR lpszValue, LPCSTR lpszDomain, LPCSTR lpszPath, int iMaxAge, BOOL bHttpOnly, BOOL bSecure, int enSameSite, BOOL bOnlyUpdateValueIfExists)
{
	return g_CookieMgr.SetCookie(lpszName, lpszValue, lpszDomain, lpszPath, iMaxAge, bHttpOnly, bSecure, (CCookie::EnSameSite)enSameSite, bOnlyUpdateValueIfExists);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpCookie_MGR_DeleteCookie(LPCSTR lpszDomain, LPCSTR lpszPath, LPCSTR lpszName)
{
	return g_CookieMgr.DeleteCookie(lpszDomain, lpszPath, lpszName);
}

HPSOCKET_API void __HP_CALL HP_HttpCookie_MGR_SetEnableThirdPartyCookie(BOOL bEnableThirdPartyCookie)
{
	g_CookieMgr.SetEnableThirdPartyCookie(bEnableThirdPartyCookie);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpCookie_MGR_IsEnableThirdPartyCookie()
{
	return g_CookieMgr.IsEnableThirdPartyCookie();
}

HPSOCKET_API BOOL __HP_CALL HP_HttpCookie_HLP_ParseExpires(LPCSTR lpszExpires, __time64_t* ptmExpires)
{
	return CCookie::ParseExpires(lpszExpires, *ptmExpires);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpCookie_HLP_MakeExpiresStr(char lpszBuff[], int* piBuffLen, __time64_t tmExpires)
{
	return CCookie::MakeExpiresStr(lpszBuff, *piBuffLen, tmExpires);
}

HPSOCKET_API BOOL __HP_CALL HP_HttpCookie_HLP_ToString(char lpszBuff[], int* piBuffLen, LPCSTR lpszName, LPCSTR lpszValue, LPCSTR lpszDomain, LPCSTR lpszPath, int iMaxAge, BOOL bHttpOnly, BOOL bSecure, int enSameSite)
{
	return CCookie::ToString(lpszBuff, *piBuffLen, lpszName, lpszValue, lpszDomain, lpszPath, iMaxAge, bHttpOnly, bSecure, (CCookie::EnSameSite)enSameSite);
}

HPSOCKET_API __time64_t __HP_CALL HP_HttpCookie_HLP_CurrentUTCTime()
{
	return CCookie::CurrentUTCTime();
}

HPSOCKET_API __time64_t __HP_CALL HP_HttpCookie_HLP_MaxAgeToExpires(int iMaxAge)
{
	return CCookie::MaxAgeToExpires(iMaxAge);
}

HPSOCKET_API int __HP_CALL HP_HttpCookie_HLP_ExpiresToMaxAge(__time64_t tmExpires)
{
	return CCookie::ExpiresToMaxAge(tmExpires);
}

/*****************************************************************************************************************************************************/
/************************************************************* HTTP Global Function Exports **********************************************************/
/*****************************************************************************************************************************************************/

#endif

/*****************************************************************************************************************************************************/
/**************************************************************** Thread Pool Exports ****************************************************************/
/*****************************************************************************************************************************************************/

/**********************************************************************************/
/************************** Thread Pool 回调函数设置方法 ***************************/

HPSOCKET_API void __HP_CALL HP_Set_FN_ThreadPool_OnStartup(HP_ThreadPoolListener pListener, HP_FN_ThreadPool_OnStartup fn)
{
	((C_HP_ThreadPoolListener*)pListener)->m_fnOnStartup = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_ThreadPool_OnShutdown(HP_ThreadPoolListener pListener, HP_FN_ThreadPool_OnShutdown fn)
{
	((C_HP_ThreadPoolListener*)pListener)->m_fnOnShutdown = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_ThreadPool_OnWorkerThreadStart(HP_ThreadPoolListener pListener, HP_FN_ThreadPool_OnWorkerThreadStart fn)
{
	((C_HP_ThreadPoolListener*)pListener)->m_fnOnWorkerThreadStart = fn;
}

HPSOCKET_API void __HP_CALL HP_Set_FN_ThreadPool_OnWorkerThreadEnd(HP_ThreadPoolListener pListener, HP_FN_ThreadPool_OnWorkerThreadEnd fn)
{
	((C_HP_ThreadPoolListener*)pListener)->m_fnOnWorkerThreadEnd = fn;
}

/****************************************************/
/******************* 对象创建函数 ********************/

HPSOCKET_API HP_ThreadPoolListener __HP_CALL Create_HP_ThreadPoolListener()
{
	return (HP_ThreadPoolListener)(new C_HP_ThreadPoolListener);
}

HPSOCKET_API void __HP_CALL Destroy_HP_ThreadPoolListener(HP_ThreadPoolListener pListener)
{
	delete (IHPThreadPoolListener*)pListener;
}

HPSOCKET_API HP_ThreadPool __HP_CALL Create_HP_ThreadPool(HP_ThreadPoolListener pListener)
{
	return (HP_ThreadPool)(new CHPThreadPool((IHPThreadPoolListener*)pListener));
}

HPSOCKET_API void __HP_CALL Destroy_HP_ThreadPool(HP_ThreadPool pThreadPool)
{
	delete (IHPThreadPool*)pThreadPool;
}

HPSOCKET_API LPTSocketTask __HP_CALL Create_HP_SocketTaskObj(Fn_SocketTaskProc fnTaskProc, PVOID pSender, HP_CONNID dwConnID, LPCBYTE pBuffer, INT iBuffLen, En_HP_TaskBufferType enBuffType, WPARAM wParam, LPARAM lParam)
{
	return ::CreateSocketTaskObj(fnTaskProc, pSender, dwConnID, pBuffer, iBuffLen, enBuffType, wParam, lParam);
}

HPSOCKET_API void __HP_CALL Destroy_HP_SocketTaskObj(LPTSocketTask pTask)
{
	::DestroySocketTaskObj(pTask);
}

/***********************************************************************/
/***************************** 组件操作方法 *****************************/

HPSOCKET_API BOOL __HP_CALL HP_ThreadPool_Start(HP_ThreadPool pThreadPool, DWORD dwThreadCount, DWORD dwMaxQueueSize, En_HP_RejectedPolicy enRejectedPolicy, DWORD dwStackSize)
{
	return ((IHPThreadPool*)pThreadPool)->Start(dwThreadCount, dwMaxQueueSize, enRejectedPolicy, dwStackSize);
}

HPSOCKET_API BOOL __HP_CALL HP_ThreadPool_Stop(HP_ThreadPool pThreadPool, DWORD dwMaxWait)
{
	return ((IHPThreadPool*)pThreadPool)->Stop(dwMaxWait);
}

HPSOCKET_API BOOL __HP_CALL HP_ThreadPool_Submit(HP_ThreadPool pThreadPool, HP_Fn_TaskProc fnTaskProc, PVOID pvArg, DWORD dwMaxWait)
{
	return ((IHPThreadPool*)pThreadPool)->Submit(fnTaskProc, pvArg, dwMaxWait);
}

HPSOCKET_API BOOL __HP_CALL HP_ThreadPool_Submit_Task(HP_ThreadPool pThreadPool, HP_LPTSocketTask pTask, DWORD dwMaxWait)
{
	return ((IHPThreadPool*)pThreadPool)->Submit(pTask, dwMaxWait);
}

HPSOCKET_API BOOL __HP_CALL HP_ThreadPool_AdjustThreadCount(HP_ThreadPool pThreadPool, DWORD dwNewThreadCount)
{
	return ((IHPThreadPool*)pThreadPool)->AdjustThreadCount(dwNewThreadCount);
}

HPSOCKET_API BOOL __HP_CALL HP_ThreadPool_Wait(HP_ThreadPool pThreadPool, DWORD dwMilliseconds)
{
	return ((IHPThreadPool*)pThreadPool)->Wait(dwMilliseconds);
}

/***********************************************************************/
/***************************** 属性访问方法 *****************************/

HPSOCKET_API BOOL __HP_CALL HP_ThreadPool_HasStarted(HP_ThreadPool pThreadPool)
{
	return ((IHPThreadPool*)pThreadPool)->HasStarted();
}

HPSOCKET_API En_HP_ServiceState	__HP_CALL HP_ThreadPool_GetState(HP_ThreadPool pThreadPool)
{
	return ((IHPThreadPool*)pThreadPool)->GetState();
}

HPSOCKET_API DWORD __HP_CALL HP_ThreadPool_GetQueueSize(HP_ThreadPool pThreadPool)
{
	return ((IHPThreadPool*)pThreadPool)->GetQueueSize();
}

HPSOCKET_API DWORD __HP_CALL HP_ThreadPool_GetTaskCount(HP_ThreadPool pThreadPool)
{
	return ((IHPThreadPool*)pThreadPool)->GetTaskCount();
}

HPSOCKET_API DWORD __HP_CALL HP_ThreadPool_GetThreadCount(HP_ThreadPool pThreadPool)
{
	return ((IHPThreadPool*)pThreadPool)->GetThreadCount();
}

HPSOCKET_API DWORD __HP_CALL HP_ThreadPool_GetMaxQueueSize(HP_ThreadPool pThreadPool)
{
	return ((IHPThreadPool*)pThreadPool)->GetMaxQueueSize();
}

HPSOCKET_API En_HP_RejectedPolicy __HP_CALL HP_ThreadPool_GetRejectedPolicy(HP_ThreadPool pThreadPool)
{
	return ((IHPThreadPool*)pThreadPool)->GetRejectedPolicy();
}

/*****************************************************************************************************************************************************/
/********************************************************* Compressor / Decompressor Exports *********************************************************/
/*****************************************************************************************************************************************************/

/****************************************************/
/******************* 对象创建函数 ********************/

HPSOCKET_API void __HP_CALL Destroy_HP_Compressor(HP_Compressor pCompressor)
{
	::DestroyCompressor((IHPCompressor*)pCompressor);
}

HPSOCKET_API void __HP_CALL Destroy_HP_Decompressor(HP_Decompressor pDecompressor)
{
	::DestroyDecompressor((IHPDecompressor*)pDecompressor);
}

#ifdef _ZLIB_SUPPORT

HPSOCKET_API HP_Compressor __HP_CALL Create_HP_ZLibCompressor(HP_Fn_CompressDataCallback fnCallback)
{
	return ::CreateZLibCompressor(fnCallback);
}

HPSOCKET_API HP_Compressor __HP_CALL Create_HP_ZLibCompressorEx(HP_Fn_CompressDataCallback fnCallback, int iWindowBits, int iLevel, int iMethod, int iMemLevel, int iStrategy, DWORD dwBuffSize)
{
	return ::CreateZLibCompressor(fnCallback, iWindowBits, iLevel, iMethod, iMemLevel, iStrategy, dwBuffSize);
}

HPSOCKET_API HP_Compressor __HP_CALL Create_HP_GZipCompressor(HP_Fn_CompressDataCallback fnCallback)
{
	return ::CreateGZipCompressor(fnCallback);
}

HPSOCKET_API HP_Compressor __HP_CALL Create_HP_GZipCompressorEx(HP_Fn_CompressDataCallback fnCallback, int iLevel, int iMethod, int iMemLevel, int iStrategy, DWORD dwBuffSize)
{
	return ::CreateGZipCompressor(fnCallback, iLevel, iMethod, iMemLevel, iStrategy, dwBuffSize);
}

HPSOCKET_API HP_Decompressor __HP_CALL Create_HP_ZLibDecompressor(HP_Fn_DecompressDataCallback fnCallback)
{
	return ::CreateZLibDecompressor(fnCallback);
}

HPSOCKET_API HP_Decompressor __HP_CALL Create_HP_ZLibDecompressorEx(HP_Fn_DecompressDataCallback fnCallback, int iWindowBits, DWORD dwBuffSize)
{
	return ::CreateZLibDecompressor(fnCallback, iWindowBits, dwBuffSize);
}

HPSOCKET_API HP_Decompressor __HP_CALL Create_HP_GZipDecompressor(HP_Fn_DecompressDataCallback fnCallback)
{
	return ::CreateGZipDecompressor(fnCallback);
}

HPSOCKET_API HP_Decompressor __HP_CALL Create_HP_GZipDecompressorEx(HP_Fn_DecompressDataCallback fnCallback, DWORD dwBuffSize)
{
	return ::CreateGZipDecompressor(fnCallback, dwBuffSize);
}

#endif

#ifdef _BROTLI_SUPPORT

HPSOCKET_API HP_Compressor __HP_CALL Create_HP_BrotliCompressor(HP_Fn_CompressDataCallback fnCallback)
{
	return ::CreateBrotliCompressor(fnCallback);
}

HPSOCKET_API HP_Compressor __HP_CALL Create_HP_BrotliCompressorEx(HP_Fn_CompressDataCallback fnCallback, int iQuality, int iWindow, int iMode, DWORD dwBuffSize)
{
	return ::CreateBrotliCompressor(fnCallback, iQuality, iWindow, iMode, dwBuffSize);
}

HPSOCKET_API HP_Decompressor __HP_CALL Create_HP_BrotliDecompressor(HP_Fn_DecompressDataCallback fnCallback)
{
	return ::CreateBrotliDecompressor(fnCallback);
}

HPSOCKET_API HP_Decompressor __HP_CALL Create_HP_BrotliDecompressorEx(HP_Fn_DecompressDataCallback fnCallback, DWORD dwBuffSize)
{
	return ::CreateBrotliDecompressor(fnCallback, dwBuffSize);
}

#endif

/***********************************************************************/
/***************************** 组件操作方法 *****************************/

HPSOCKET_API BOOL __HP_CALL HP_Compressor_Process(HP_Compressor pCompressor, const BYTE* pData, int iLength, BOOL bLast, PVOID pContext)
{
	return ((IHPCompressor*)pCompressor)->Process(pData, iLength, bLast, pContext);
}

HPSOCKET_API BOOL __HP_CALL HP_Compressor_ProcessEx(HP_Compressor pCompressor, const BYTE* pData, int iLength, BOOL bLast, BOOL bFlush, PVOID pContext)
{
	return ((IHPCompressor*)pCompressor)->ProcessEx(pData, iLength, bLast, bFlush, pContext);
}

HPSOCKET_API BOOL __HP_CALL HP_Compressor_Reset(HP_Compressor pCompressor)
{
	return ((IHPCompressor*)pCompressor)->Reset();
}

HPSOCKET_API BOOL __HP_CALL HP_Decompressor_Process(HP_Decompressor pDecompressor, const BYTE* pData, int iLength, PVOID pContext)
{
	return ((IHPDecompressor*)pDecompressor)->Process(pData, iLength, pContext);
}

HPSOCKET_API BOOL __HP_CALL HP_Decompressor_Reset(HP_Decompressor pDecompressor)
{
	return ((IHPDecompressor*)pDecompressor)->Reset();
}

/***********************************************************************/
/***************************** 属性访问方法 *****************************/

HPSOCKET_API BOOL __HP_CALL HP_Compressor_IsValid(HP_Compressor pCompressor)
{
	return ((IHPCompressor*)pCompressor)->IsValid();
}

HPSOCKET_API BOOL __HP_CALL HP_Decompressor_IsValid(HP_Decompressor pDecompressor)
{
	return ((IHPDecompressor*)pDecompressor)->IsValid();
}
