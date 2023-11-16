/*!
 * \file TraderHTS.cpp
 * \project	WonderTrader
 *
 * \author Suuuunseeker
 * \date 2022/09/06
 *
 * \brief
 */

#include "TraderHTS.h"
#include "../API/HTS5.2.43.0/include/scopeguard.hpp"

#include "../Includes/WTSError.hpp"
#include "../Includes/WTSContractInfo.hpp"
#include "../Includes/WTSSessionInfo.hpp"
#include "../Includes/WTSTradeDef.hpp"
#include "../Includes/WTSDataDef.hpp"
#include "../Includes/WTSVariant.hpp"
#include "../Share/StdUtils.hpp"
#include "../Share/TimeUtils.hpp"
#include "../Includes/IBaseDataMgr.h"
#include "../Share/DLLHelper.hpp"
#include "../Share/decimal.h"
#include "../Share/StrUtil.hpp"
#include <rapidjson/document.h>
#include <map>
#include <algorithm>

#include <iostream>

#include "../Share/BoostFile.hpp"
 //By Wesley @ 2022.01.05
#include "../Share/fmtlib.h"


/*
 @brief: ����ȫ��map�����ڲ��Ҷ�ʵ������ص���Ӧ��TraderHTS����Ľ��
*/
std::map<std::string, TraderHTS*> itpdkCallMap;

std::unique_ptr<HTSCallMgr> g_callback_mgr;

template<typename... Args>
inline void write_log(ITraderSpi* sink, WTSLogLevel ll, const char* format, const Args&... args)
{
	if (sink == NULL)
		return;

	const char* buffer = fmtutil::format(format, args...);

	sink->handleTraderLog(ll, buffer);
}

void inst_hlp() {}

#ifdef _WIN32
#ifdef _WIN64
#pragma comment(lib, "../API/HTS5.2.43.0/x64/secitpdk.lib")
#else
#pragma comment(lib, "../API/HTS5.2.43.0/x86/secitpdk_x86.lib")
#endif
#include <wtypes.h>
HMODULE	g_dllModule = NULL;

BOOL APIENTRY DllMain(
	HANDLE hModule,
	DWORD  ul_reason_for_call,
	LPVOID lpReserved
)
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
		g_dllModule = (HMODULE)hModule;
		break;
	}
	return TRUE;
}
#else
#include <dlfcn.h>

char PLATFORM_NAME[] = "UNIX";

std::string	g_moduleName;

const std::string& getInstPath()
{
	static std::string moduleName;
	if (moduleName.empty())
	{
		Dl_info dl_info;
		dladdr((void *)inst_hlp, &dl_info);
		moduleName = dl_info.dli_fname;
	}

	return moduleName;
}
#endif

std::string getBinDir()
{
	static std::string _bin_dir;
	if (_bin_dir.empty())
	{


#ifdef _WIN32
		char strPath[MAX_PATH];
		GetModuleFileName(g_dllModule, strPath, MAX_PATH);

		_bin_dir = StrUtil::standardisePath(strPath, false);
#else
		_bin_dir = getInstPath();
#endif
		boost::filesystem::path p(_bin_dir);
		_bin_dir = p.branch_path().string() + "/";
	}

	return _bin_dir;
}

const char* ENTRUST_SECTION = "entrusts";
const char* ORDER_SECTION = "orders";

inline const char* exchgI2O(const char* exchg)
{
	if (strcmp(exchg, "SSE") == 0)
		return "SH";
	else if (strcmp(exchg, "SZSE") == 0)
		return "SZ";

	return "";
}

inline const char* exchgO2I(const char* exchg)
{
	if (strcmp(exchg, "SH") == 0)
		return "SSE";
	else if (strcmp(exchg, "SZ") == 0)
		return "SZSE";

	return "";
}

WTSOrderState wrapOrderState(int state)
{
	switch (state)
	{
	case 0: return WOS_Nottouched; break;
	case 1: return WOS_Submitting; break;
	case 2: return WOS_NotTraded_NotQueuing; break;
	case 5: return WOS_PartTraded_Queuing; break;
	case 6: return WOS_AllTraded; break;
	default:
		return WOS_Canceled;
		break;
	}
}

uint32_t strToTime(const char* strTime)
{
	std::string str;
	const char *pos = strTime;
	while (strlen(pos) > 0)
	{
		if (pos[0] != ':' && pos[0] != '.')
		{
			str.append(pos, 1);
		}
		pos++;
	}

	return strtoul(str.c_str(), NULL, 10);
}

extern "C"
{
	EXPORT_FLAG ITraderApi* createTrader()
	{
		TraderHTS *instance = new TraderHTS();
		return instance;
	}

	EXPORT_FLAG void deleteTrader(ITraderApi* &trader)
	{
		if (NULL != trader)
		{
			delete trader;
			trader = NULL;
		}
	}
}

std::string SECITPDK_GetLastError()
{
	char msg[256];
	SECITPDK_GetLastError(msg);            //��ȡ������Ϣ
	return msg;
}

void ConnCallBack(const char* pKhh, const char* pConnKey, int nEvent, void* pData)
{
	std::cout << "Connect event khh: " << pKhh << "event: " << (0 == nEvent ? "�ָ�" : "�Ͽ�") << std::endl;
}

TraderHTS::TraderHTS()
	: m_wrapperState(WS_NOTLOGIN)
	, m_uLastQryTime(0)
	, m_bInQuery(false)
	, m_bUseEX(false)
	, m_bASync(false)
	, m_strandIO(NULL)
	, m_lastQryTime(0)
	, m_orderRef(1)
	, m_lDate(0)
	, m_mapLives(NULL)
	, m_bdMgr(NULL)
	, m_traderSink(NULL)
{
	m_mapLives = TradeDataMap::create();

	sShGdh[13] = { 0 };                //�Ϻ��ɶ���
	sSzGdh[13] = { 0 };                //���ڹɶ���
	sShZjzh[13] = { 0 };               //�ʽ��˺�
	sSzZjzh[13] = { 0 };               //�ʽ��˺�
}

TraderHTS::~TraderHTS()
{

}

void TraderHTS::InitializeHTS(WTSVariant* params)
{
	SECITPDK_SetLogPath("./");            //��־Ŀ¼
	SECITPDK_SetProfilePath("./");           //�����ļ�Ŀ¼

	static bool bInited = false;
	bInited = g_callback_mgr->init(params);  //��ʼ���������нӿ�ʹ��ǰ���ã���·�����ýӿ���
	if (!bInited)
	{
		write_log(m_traderSink, LL_ERROR, "[TraderHTS]��ʼ��ʧ��!");
		return;
	}
}

bool TraderHTS::init(WTSVariant* params)
{
	m_strUser = params->getCString("user");
	m_strPass = params->getCString("pass");

	m_strWtfs = params->getCString("order_way");
	m_strNode = params->getCString("node");
	m_strKey = params->getCString("key");

	m_bUseEX = params->getBoolean("use_ex");
	m_bASync = params->getBoolean("use_async");

	std::string temp;
	temp = params->getCString("shGdh");
	strncpy(sShGdh, temp.c_str(), sizeof(sShGdh));
	temp = params->getCString("szGdh");
	strncpy(sSzGdh, temp.c_str(), sizeof(sShGdh));
	temp = params->getCString("shFundAcct");
	strncpy(sShZjzh, temp.c_str(), sizeof(sShGdh));
	temp = params->getCString("szFundAcct");
	strncpy(sSzZjzh, temp.c_str(), sizeof(sShGdh));

	WTSVariant* param = params->get("ddmodule");
	if (param != NULL)
	{
		m_strModule = getBinDir() + param->asCString();
	}
	else
	{
#ifdef _WIN32
#ifdef _WIN64
		m_strModule = getBinDir() + "fixapitool.dll";
		m_strModule = getBinDir() + "fixapi50.dll";
		m_strModule = getBinDir() + "secitpdk.dll";
#else
		m_strModule = getBinDir() + "fixapitool.dll";
		m_strModule = getBinDir() + "fixapi50_x86.dll";
		m_strModule = getBinDir() + "secitpdk_x86.dll";
#endif
#else
		m_strModule = getBinDir() + "libfixapitool.dll";
		m_strModule = getBinDir() + "libfixapi.so";
		m_strModule = getBinDir() + "libsecitpdk.so";
#endif
	}

	m_hInstDD = DLLHelper::load_library(m_strModule.c_str());

	// ��ʼ������
	InitializeHTS(params);

	// ���ûص�����������Ϣ
	try
	{
		//g_callback_mgr->setCallbackMsgFunc();  // ������Ϣ�����ûص�����
		g_callback_mgr->setHTSCallPtr(m_strUser.c_str(), this);  // ����ǰ������ӵ�ȫ��ӳ�����
	}
	catch (...)
	{
		std::string msg = SECITPDK_GetLastError();
		write_log(m_traderSink, LL_ERROR, "[TraderHTS]���ĳɽ��ر�ʧ�ܣ�{}({})", msg.c_str(), -1);
	}

	return true;
}

void TraderHTS::release()
{
	if (m_mapLives)
		m_mapLives->release();
}

void TraderHTS::reconnect()
{
	//if (m_traderSink)
	//{
	//	m_traderSink->handleEvent(WTE_Connect, -1);
	//	m_traderSink->handleTraderLog(LL_ERROR, "[TraderHTS]ͨѶ����ʧ��");
	//}

	//StdThreadPtr thrd(new StdThread([this]() {
	//	std::this_thread::sleep_for(std::chrono::seconds(2));
	//	if (m_traderSink)
	//	{
	//		write_log(m_traderSink, LL_WARN, "[TraderHTS]�˺�{}������������", m_strUser.c_str());  //m_traderSink->handleTraderLog(LL_WARN, "[TraderHTS]�˺�%s������������", m_strUser.c_str());
	//	}
	//	reconnect();
	//}));

	//registerSpi(this);						// ע���¼�

	if (m_traderSink) m_traderSink->handleEvent(WTE_Connect, 0);
}

void TraderHTS::connect()
{
	if (m_thrdWorker == NULL)
	{
		m_strandIO = new boost::asio::io_service::strand(m_asyncIO);
		boost::asio::io_service::work work(m_asyncIO);
		m_thrdWorker.reset(new StdThread([this]() {
			while (true)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(2));
				m_asyncIO.run_one();
			}
		}));
	}

	reconnect();
}

void TraderHTS::disconnect()
{
	m_asyncIO.post([this]() {
		release();
	});

	if (m_thrdWorker)
	{
		m_asyncIO.stop();
		m_thrdWorker->join();
		m_thrdWorker = NULL;

		delete m_strandIO;
		m_strandIO = NULL;
	}
}

void TraderHTS::htsGetCallback(const char* pTime, const char* pMsg, int nType)
{
	// ��������
	write_log(m_traderSink, LL_INFO, "[TraderHTS] ��ʼ���ûص�����");

	rapidjson::Document document;
	if (document.Parse(pMsg).HasParseError())
	{
		write_log(m_traderSink, LL_ERROR, "[TraderHTS] ��������ί��������Ϣʧ��!");
		return;
	}

	long nWTH = (long)document["WTH"].GetInt64();
	const char* sKHH = document["KHH"].GetString();
	const char* sGDH = document["GDH"].GetString();
	const char* sZQDM = document["ZQDM"].GetString();

	if (NOTIFY_PUSH_ORDER == nType)
	{
		//ȷ�ϻر�����
		//write_log(m_traderSink, LL_INFO, "[TraderHTS] Insert order confirm callback: WTH = {}, KHH = {},GDH = {},ZQDM = {}", nWTH, sKHH, sGDH, sZQDM);

		OnRtnOrder(pMsg);
	}
	else if (NOTIFY_PUSH_MATCH == nType)
	{
		//�ɽ��ر�����
		//write_log(m_traderSink, LL_INFO, "[TraderHTS] Order match confirm callback: WTH = {}, KHH = {},GDH = {},ZQDM = {}", nWTH, sKHH, sGDH, sZQDM);

		OnRtnTrade(pMsg);
	}
	else if (NOTIFY_PUSH_WITHDRAW == nType)
	{
		// �����ر�����
		//write_log(m_traderSink, LL_INFO, "[TraderHTS] Cancel order confirm callback: WTH = {}, KHH = {},GDH = {},ZQDM = {}", nWTH, sKHH, sGDH, sZQDM);

		const char* sCXBZ = document["CXBZ"].GetString();
		int64 iSBJG = document["SBJG"].GetInt();
		const char* sJGSM = document["JGSM"].GetString();
		//����ȷ�ϻر�
		//write_log(m_traderSink, LL_DEBUG, "HTSCall cancelling report message: WTH={}, KHH= {}, GDH={}, ZQDM={}, sCXBZ={}, iSBJG={} , sJGSM={}", nWTH, sKHH, sGDH, sZQDM, sCXBZ, iSBJG, sJGSM);

		OnRtnOrder(pMsg);
	}
	else if (NOTIFY_PUSH_INVALID == nType)
	{
		//�ϵ�
		write_log(m_traderSink, LL_WARN, "[TraderHTS] Abandoned order confirm callback: WTH = {}, KHH = {},GDH = {},ZQDM = {}", nWTH, sKHH, sGDH, sZQDM);
	}
}

void TraderHTS::OnRtnOrder(const char* pMsg)
{
	rapidjson::Document document;
	if (document.Parse(pMsg).HasParseError())   //����ʧ��
	{
		write_log(m_traderSink, LL_ERROR, "[TraderHTS]��������ί��������Ϣʧ��.");
		return;
	}

	std::string code = document["ZQDM"].GetString();
	std::string exchg = exchgO2I(document["JYS"].GetString());

	bool isCancel = document["CXBZ"].GetString()[0] == 'W';  // ������־�Ƿ���W /// O��ʾί��

	//ȷ������
	//std::cout << "Accept push report message: " << " code: " << code << "  exchg: " << "  wth: " << document["WTH"].GetInt64() << "  kfsbdbh: "  << document["KFSBDBH"].GetInt64() << "  wtjg: " << document["WTJG"].GetDouble() << "  wjsl: " << document["WTSL"].GetDouble() << std::endl;

	WTSContractInfo* contract = m_bdMgr->getContract(code.c_str(), exchg.c_str());

	if (contract != NULL)
	{
		WTSOrderInfo* ordInfo = NULL;
		if (!isCancel)
		{
			char buf[256] = { 0 };
			strncpy(buf, std::to_string(document["WTH"].GetInt64()).c_str(), sizeof(buf));  /// ����ʹ��KFSBDBH�� ������������ΪWTH
			ordInfo = (WTSOrderInfo*)m_mapLives->grab(buf);

			if (ordInfo == NULL)
			{
				ordInfo = WTSOrderInfo::create();
				ordInfo->setPrice(document["WTJG"].GetDouble());
				ordInfo->setVolume(document["WTSL"].GetDouble());
				ordInfo->setDirection(WDT_LONG);
				ordInfo->setPriceType(WPT_LIMITPRICE);

				ordInfo->setOffsetType(document["JYLB"].GetInt() == 1 ? WOT_OPEN : WOT_CLOSE);

				//ordInfo->setOrderDate(m_lDate);
				//len = 256;
				//Fix_GetItem(hsess, FID_WTSJ, buf, len);
				//uint32_t uTime = strToTime(buf);
				//ordInfo->setOrderTime(TimeUtils::makeTime(ordInfo->getOrderDate(), uTime));
				ordInfo->setOrderDate(TimeUtils::getCurDate());
				ordInfo->setOrderTime(TimeUtils::getLocalTimeNow());

				ordInfo->setCode(code.c_str());
				ordInfo->setExchange(contract->getExchg());

				ordInfo->setOrderID(buf);

				strncpy(buf, std::to_string(document["KFSBDBH"].GetInt64()).c_str(), 256);
				ordInfo->setEntrustID(buf);  ///ʹ�ÿ����̱��ر��

				std::string usertag = m_iniHelper.readString(ENTRUST_SECTION, ordInfo->getEntrustID(), "");
				if (usertag.empty())
				{
					ordInfo->setUserTag(ordInfo->getEntrustID());
				}
				else
				{
					ordInfo->setUserTag(usertag.c_str());

					if (strlen(ordInfo->getOrderID()) > 0)
					{
						m_iniHelper.writeString(ORDER_SECTION, StrUtil::trim(ordInfo->getOrderID()).c_str(), usertag.c_str());
						m_iniHelper.save();
					}
				}

				if (m_mapLives == NULL)
					m_mapLives = TradeDataMap::create();

				if (ordInfo->isAlive())
					m_mapLives->add(ordInfo->getOrderID(), ordInfo, true);
			}

			int state = document["SBJG"].GetInt();
			ordInfo->setOrderState(wrapOrderState(state));
			if (state == 3)
				ordInfo->setError(true);

			double total = document["WTSL"].GetDouble();
			double traded = document["CJSL"].GetDouble();
			double canceled = document["CDSL"].GetDouble();
			ordInfo->setVolume(total);

			if (ordInfo->isAlive())
				ordInfo->setVolLeft(total - canceled - traded);
			else
				ordInfo->setVolLeft(0);

			strncpy(buf, document["JGSM"].GetString(), 256);
			ordInfo->setStateMsg(buf);

			if (!ordInfo->isAlive())
				m_mapLives->remove(ordInfo->getOrderID());
		}
		else
		{
			char buf[256] = { 0 };
			strncpy(buf, std::to_string(document["CXWTH"].GetInt64()).c_str(), sizeof(buf));  // ��Ҫʹ�ó���ί�к�
			ordInfo = (WTSOrderInfo*)m_mapLives->grab(buf);
			if (ordInfo == NULL)
				return;

			m_mapLives->remove(buf);

			ordInfo->setVolLeft(0);
			ordInfo->setOrderState(WOS_Canceled);

			strncpy(buf, document["JGSM"].GetString(), 256);
			ordInfo->setStateMsg(buf);

			//std::cout << "cxwth: " << document["CXWTH"].GetInt64() << " wth: " << document["WTH"].GetInt64() << " result_info: " << document["JGSM"].GetString() << std::endl;  // ʹ��ί�к�
		}

		if (m_traderSink)
			m_traderSink->onPushOrder(ordInfo);

		ordInfo->release();
	}
}

void TraderHTS::OnRtnTrade(const char* pMsg)
{
	write_log(m_traderSink, LL_INFO, "[TraderHTS] Trade Report beginning ...");

	rapidjson::Document document;
	if (document.Parse(pMsg).HasParseError())   //����ʧ��
	{
		write_log(m_traderSink, LL_ERROR, "[TraderHTS]���������ɽ��ر���Ϣʧ��.");
		return;
	}

	std::string code = document["ZQDM"].GetString();
	std::string exchg = exchgO2I(document["JYS"].GetString());

	//�ɽ�����
	std::string tradeid = document["CJBH"].GetString();

	auto it = m_tradeids.find(tradeid);
	if (it != m_tradeids.end())
		return;

	m_tradeids.insert(tradeid);

	//���˵������ر�
	if (!decimal::eq(document["CDSL"].GetDouble(), 0.0))
		return;

	WTSContractInfo* contract = m_bdMgr->getContract(code.c_str(), exchg.c_str());

	if (contract != NULL)
	{
		std::string orderid = std::to_string(document["WTH"].GetInt64());   // ����ʹ��ί�к���Ϊorderid

		WTSCommodityInfo* commInfo = contract->getCommInfo();
		WTSTradeInfo *trdInfo = WTSTradeInfo::create(code.c_str(), exchg.c_str());
		trdInfo->setPrice(document["CJJG"].GetDouble());
		trdInfo->setVolume(document["CJSL"].GetDouble());

		WTSOrderInfo* ordInfo = (WTSOrderInfo*)m_mapLives->get(orderid);
		if (ordInfo)
		{
			ordInfo->setVolTraded(ordInfo->getVolTraded() + trdInfo->getVolume());
			ordInfo->setVolLeft(ordInfo->getVolLeft() - trdInfo->getVolume());

			if (ordInfo->getVolLeft() == 0)
			{
				ordInfo->setOrderState(WOS_AllTraded);
				if (m_traderSink)
					m_traderSink->onPushOrder(ordInfo);
			}
		}

		trdInfo->setTradeID(tradeid.c_str());

		trdInfo->setTradeDate(m_lDate);

		const int len = 256;
		char buf[len];
		strncpy(buf, document["CJSJ"].GetString(), len);
		uint32_t uTime = strToTime(buf);
		trdInfo->setTradeTime(TimeUtils::makeTime(m_lDate, uTime));

		trdInfo->setDirection(WDT_LONG);
		trdInfo->setOffsetType(document["JYLB"].GetInt() == 1 ? WOT_OPEN : WOT_CLOSE);
		trdInfo->setRefOrder(orderid.c_str());
		trdInfo->setTradeType(WTT_Common);

		trdInfo->setAmount(document["CJJE"].GetDouble());

		std::string usertag = m_iniHelper.readString(ORDER_SECTION, StrUtil::trim(trdInfo->getRefOrder()).c_str());
		if (!usertag.empty())
			trdInfo->setUserTag(usertag.c_str());

		if (m_traderSink)
			m_traderSink->onPushTrade(trdInfo);
	}
}

bool TraderHTS::makeEntrustID(char* buffer, int length)
{
	if (buffer == NULL || length == 0)
		return false;

	try
	{
		memset(buffer, 0, length);
		uint32_t orderref = m_orderRef.fetch_add(1) + 1;
		//sprintf(buffer, "%s#%u",StrUtil::fmtUInt64(m_uSessID).c_str(), orderref);
		fmt::format_to(buffer, "{}#{}", m_uSessID, orderref);
		return true;
	}
	catch (...)
	{

	}

	return false;
}

void TraderHTS::registerSpi(ITraderSpi *listener)
{
	m_traderSink = listener;
	if (m_traderSink)
	{
		m_bdMgr = listener->getBaseDataMgr();
	}
}

int TraderHTS::login(const char* user, const char* pass, const char* productInfo)
{
	m_strUser = user;
	m_strPass = pass;

	m_wrapperState = WS_LOGINING;

	doLogin();

	return 0;
}

void TraderHTS::qryGDNo()
{
	m_strandIO->post([this]() {

		std::cout << "=========== query_gudong_info ============" << std::endl;

		long nRet = 0;

		std::vector<ITPDK_GDH> argGDH;
		argGDH.reserve(5);

		if (m_bUseEX)
		{
			nRet = (long)SECITPDK_QueryAccInfoEx(cusreqinfo, argGDH);
		}
		else
		{
			nRet = (long)SECITPDK_QueryAccInfo(m_strUser.c_str(), argGDH);
		}

		if (nRet < 0)
		{
			string msg = SECITPDK_GetLastError();              //��ѯʧ�ܣ���ȡ������Ϣ
			write_log(m_traderSink, LL_ERROR, "[TraderHTS]��ѯ�ɶ���Ϣʧ�ܣ� ������Ϣ: {}({}) ", msg.c_str(), nRet);
			return;
		}

		std::cout << "SECITPDK_QueryAccInfo success. Num of results " << nRet << std::endl;

		for (auto& itr : argGDH)
		{
			printf("AccountId:%s -- Market:%s;SecuAccount:%s;FundAccount:%s;\n",
				itr.AccountId, itr.Market, itr.SecuAccount, itr.FundAccount);

			if (strlen(itr.SecuAccount) > 0)
			{
				strncpy(sShZjzh, itr.FundAccount, sizeof(sShZjzh));
				strncpy(sSzZjzh, itr.FundAccount, sizeof(sSzZjzh));
			}
			if (0 == strcmp(itr.Market, "SH"))
			{
				strncpy(sShGdh, itr.SecuAccount, sizeof(sShGdh));
			}
			if (0 == strcmp(itr.Market, "SZ"))
			{
				strncpy(sSzGdh, itr.SecuAccount, sizeof(sSzGdh));
			}
		}
	});
}

void TraderHTS::doLogin()
{
	m_strandIO->post([this]() {
		if (m_bUseEX)
		{
			strcpy(cusreqinfo.AccountId, m_strUser.c_str());  // �ͻ���
			strcpy(cusreqinfo.Password, m_strPass.c_str());  // ����
			strcpy(cusreqinfo.DevelCode, m_strDevName.c_str());  // ��������Ϣ
		}

		//std::cout << m_strKey << "  " << m_strUser << "  " << m_strPass << std::endl;

		int64 nRet = 0;
		if (m_bUseEX)
		{
			nRet = SECITPDK_TradeLoginEx(m_strKey.c_str(), cusreqinfo);     //��¼
		}
		else
		{
			nRet = SECITPDK_TradeLogin(m_strKey.c_str(), m_strUser.c_str(), m_strPass.c_str());     //��¼
		}

		if (nRet <= 0)
		{
			m_wrapperState = WS_LOGINFAILED;
			std::string msg = SECITPDK_GetLastError();  //��¼ʧ�ܣ���ȡ������Ϣ
			m_traderSink->onLoginResult(false, msg.c_str(), 0);
			write_log(m_traderSink, LL_ERROR, "[TraderHTS]��¼ʧ��, ������Ϣ: {}({})", msg.c_str(), nRet);
		}
		else
		{
			write_log(m_traderSink, LL_INFO, "[TraderHTS]��¼�ɹ��� Token: {}", nRet);

			std::stringstream ss;
			ss << "./HTSData/local/";
			std::string path = StrUtil::standardisePath(ss.str());
			if (!StdFile::exists(path.c_str()))
				boost::filesystem::create_directories(path.c_str());
			ss << m_strUser << ".dat";

			m_iniHelper.load(ss.str().c_str());
			uint32_t lastDate = m_iniHelper.readUInt("marker", "date", 0);
			if (lastDate != m_lDate)
			{
				//�����ղ�ͬ�������ԭ��������
				m_iniHelper.removeSection(ENTRUST_SECTION);
				m_iniHelper.removeSection(ORDER_SECTION);
				m_iniHelper.writeUInt("marker", "date", m_lDate);
				m_iniHelper.save();

				write_log(m_traderSink, LL_INFO, "[TraderHTS][%s]���������л�[{} -> {}]����ձ������ݻ��桭��", m_strUser.c_str(), lastDate, m_lDate);
			}

			m_wrapperState = WS_LOGINED;
			m_lDate = TimeUtils::getCurDate();

			//qryGDNo();

			write_log(m_traderSink, LL_INFO, "[TraderHTS]�˻����ݳ�ʼ�����...");
			m_wrapperState = WS_ALLREADY;
			m_traderSink->onLoginResult(true, "", m_lDate);
		}

		//cout << "=========== time_example ============" << endl;
		//cout << "SECITPDK_GetTickCount=" << SECITPDK_GetTickCount() << endl;
		//cout << "SECITPDK_GetDoubleTickCount=" << SECITPDK_GetDoubleTickCount() << endl;
		//cout << "SECITPDK_GetSystemDate=" << SECITPDK_GetSystemDate() << endl;
		//cout << "SECITPDK_GetTradeDate=" << SECITPDK_GetTradeDate() << endl;
		//cout << "SECITPDK_GetReviseTimeAsLong=" << SECITPDK_GetReviseTimeAsLong() << endl;

		//char time[128];
		//SECITPDK_GetReviseTime(time);
		//cout << "SECITPDK_GetReviseTime=" << time << endl;
	});
}

int TraderHTS::logout()
{
	ScopeGuard OnExit(
		[]() {SECITPDK_Exit(); }           //�˳�������ʹ��ITPDK�ӿ�ʱ����
	);

	return 0;
}

int TraderHTS::orderInsert(WTSEntrust* entrust)
{
	if (m_wrapperState != WS_ALLREADY)
	{
		return -1;
	}

	entrust->retain();
	m_strandIO->post([this, entrust]() {
		bool isBuy = entrust->getOffsetType() == WOT_OPEN;
		bool isSH = strcmp(entrust->getExchg(), "SSE") == 0;

		std::string exchg = isSH ? "SH" : "SZ";
		std::string gdh = isSH ? sShGdh : sSzGdh;
		int nDirection = isBuy ? JYLB_BUY : JYLB_SALE;

		if (m_bUseEX)
			strcpy(cusreqinfo.SecuAccount, gdh.c_str());

		for (int i = 0; i < 100; i++)
		{
			int64 nRet = 0;
			if (m_bASync)
			{
				if (m_bUseEX)
				{
					nRet = SECITPDK_OrderEntrustEx_ASync(cusreqinfo, exchg.c_str(), entrust->getCode(), nDirection, entrust->getVolume(), entrust->getPrice(), DDLX_XJWT);
				}
				else
				{
					nRet = SECITPDK_OrderEntrust_ASync(m_strUser.c_str(), exchg.c_str(), entrust->getCode(), nDirection, entrust->getVolume(), entrust->getPrice(), DDLX_XJWT, gdh.c_str());
				}
			}
			else
			{
				if (m_bUseEX)
				{
					nRet = SECITPDK_OrderEntrustEx(cusreqinfo, exchg.c_str(), entrust->getCode(), nDirection, entrust->getVolume(), entrust->getPrice(), DDLX_XJWT);
				}
				else
				{
					nRet = SECITPDK_OrderEntrust(m_strUser.c_str(), exchg.c_str(), entrust->getCode(), nDirection, entrust->getVolume(), entrust->getPrice(), DDLX_XJWT, gdh.c_str());
				}
			}

			if (strlen(entrust->getUserTag()) > 0)
			{
				//m_mapEntrustTag[entrust->getEntrustID()] = entrust->getUserTag();
				m_iniHelper.writeString(ENTRUST_SECTION, entrust->getEntrustID(), entrust->getUserTag());
				m_iniHelper.save();
			}

			if (nRet <= 0)
			{
				std::string msg = SECITPDK_GetLastError();
				write_log(m_traderSink, LL_ERROR, "[TraderHTS]ί��ָ���ʧ��: {}({})", msg.c_str(), nRet);

				WTSError* err = WTSError::create(WEC_ORDERINSERT, msg.c_str());
				m_traderSink->onRspEntrust(entrust, err);
			}
			else
			{
				m_traderSink->onRspEntrust(entrust, NULL);

				//�����ֶ���һ�ʻر�����Ȼ���������
				WTSOrderInfo* ordInfo = WTSOrderInfo::create(entrust);
				ordInfo->setOrderState(WOS_NotTraded_NotQueuing);
				ordInfo->setVolTraded(0);
				ordInfo->setVolLeft(ordInfo->getVolume());
				ordInfo->setOrderDate(m_lDate);
				ordInfo->setOrderTime(TimeUtils::getLocalTimeNow());

				ordInfo->setError(false);

				ordInfo->setOrderID(std::to_string(nRet).c_str());  // �˴���ί�к�

				m_iniHelper.writeString(ORDER_SECTION, StrUtil::trim(ordInfo->getOrderID()).c_str(), ordInfo->getUserTag());
				m_iniHelper.save();

				if (m_mapLives == NULL)
					m_mapLives = TradeDataMap::create();

				m_mapLives->add(ordInfo->getOrderID(), ordInfo, false);

				m_traderSink->onPushOrder(ordInfo);
			}
		}

		entrust->release();
	});

	return 0;
}

int TraderHTS::orderAction(WTSEntrustAction* action)
{
	if (m_wrapperState != WS_ALLREADY)
		return -1;

	action->retain();
	m_strandIO->post([this, action]() {
		//write_log(m_traderSink, LL_INFO, "[TraderHTS]���ó����ӿ�");

		bool isSH = strcmp(action->getExchg(), "SSE") == 0;
		int64 orderID = atol(action->getOrderID());
		std::string exchg = isSH ? "SH" : "SZ";
		int64 nRet = 0;

		/// ʹ�ÿ����̱��ر����Ϊ��������
		if (m_bASync)
		{
			if (m_bUseEX)
			{
				nRet = SECITPDK_OrderWithdrawEx_ASync(cusreqinfo, exchg.c_str(), orderID);
			}
			else
			{
				nRet = SECITPDK_OrderWithdraw_ASync(m_strUser.c_str(), exchg.c_str(), orderID);
			}
		}
		else
		{
			if (m_bUseEX)
			{
				nRet = SECITPDK_OrderWithdrawEx(cusreqinfo, exchg.c_str(), orderID);
			}
			else
			{
				nRet = SECITPDK_OrderWithdraw(m_strUser.c_str(), exchg.c_str(), orderID);
			}
		}
		;
		if (nRet <= 0)
		{
			string msg = SECITPDK_GetLastError();          //�µ�ʧ�ܣ���ȡ������Ϣ
			write_log(m_traderSink, LL_ERROR, "[TraderHTS]����ָ���ʧ��: {}({})", msg.c_str(), nRet);

			WTSError* err = WTSError::create(WEC_ORDERCANCEL, msg.c_str());
			m_traderSink->onTraderError(err);
		}
		else
		{
			write_log(m_traderSink, LL_INFO, "[TraderHTS] SECITPDK_OrderWithdraw �����ɹ��� ί�к�Wth = {}", nRet);
		}

		action->release();
	});

	return 0;
}

int TraderHTS::queryAccount()
{
	if (m_wrapperState != WS_ALLREADY)
	{
		return -1;
	}

	m_queQuery.push([this]() {
		std::vector<ITPDK_ZJZH> argFund;
		//argFund.reserve(10);
		int64 nRet = 0;

		if (m_bUseEX)
		{
			nRet = (long)SECITPDK_QueryFundInfoEx(cusreqinfo, argFund);
		}
		else
		{
			nRet = (long)SECITPDK_QueryFundInfo(m_strUser.c_str(), argFund);
		}

		if (nRet < 0)  // ��ѯʧ��
		{
			std::string msg = SECITPDK_GetLastError();
			write_log(m_traderSink, LL_ERROR, "[TraderHTS]�����ʽ��ѯʧ��: {}", msg.c_str());
		}
		else  // ��ѯ�ɹ�
		{
			WTSArray* ayFunds = WTSArray::create();

			for (auto& itr : argFund)
			{
				//printf("AccountId:%s;  FundAvl:%f;  TotalAsset:%f;  MarketValue:%f;  CurrentBalance:%f;  UnclearProfit:%f;  DiluteUnclearProfit:%f;  UncomeBalance:%f\n",
					//itr.AccountId, itr.FundAvl, itr.TotalAsset, itr.MarketValue, itr.CurrentBalance, itr.UnclearProfit, itr.DiluteUnclearProfit, itr.UncomeBalance);

				WTSAccountInfo* fundInfo = WTSAccountInfo::create();
				fundInfo->setAvailable(itr.FundAvl);   /// �����ʽ�
				fundInfo->setBalance(itr.CurrentBalance);  /// �������
				fundInfo->setDynProfit(itr.UnclearProfit);  ///����ӯ��
				fundInfo->setCloseProfit(itr.DateProfit);  /// ����ӯ��
				fundInfo->setCurrency(itr.MoneyType);  /// ����
				fundInfo->setPreBalance(itr.LastBalance);  /// �������

				ayFunds->append(fundInfo, false);
			}

			if (m_traderSink) m_traderSink->onRspAccount(ayFunds);
			ayFunds->release();
		}

		m_bInQuery = false;
	});

	triggerQuery();

	return 0;
}

int TraderHTS::queryPositions()
{
	if (m_wrapperState != WS_ALLREADY)
	{
		return -1;
	}

	StdUniqueLock lock(m_mtxQuery);
	m_queQuery.push([this]() {
		std::vector<ITPDK_ZQGL> argHolding;
		const size_t nRows = 200;
		argHolding.reserve(nRows);

		int64 nRet = 0;
		if (m_bUseEX)
		{
			nRet = (long)SECITPDK_QueryPositionsEx(cusreqinfo, SORT_TYPE_AES, nRows, 0, "", "", "", 1, argHolding);
		}
		else
		{
			nRet = SECITPDK_QueryPositions(m_strUser.c_str(), SORT_TYPE_AES, nRows, 0, "", "", "", 1, argHolding);  /// execflag�������0
		}

		if (nRet < 0)
		{
			std::string msg = SECITPDK_GetLastError();
			write_log(m_traderSink, LL_ERROR, "[TraderHTS]�û��ֲֲ�ѯʧ��: {}({})", msg.c_str(), nRet);
		}
		else
		{
			// ��������浽һ���ܵ�vector����ڷ�ҳ
			std::vector<ITPDK_ZQGL> argTotalHolding(argHolding);
			int tmp = 200;

			// ������������200�����ҳ��ѯ
			while (nRet >= nRows)
			{
				argHolding.clear();

				// ѭ����ѯ��ʱ��nBrowindex+1������һ��
				if (m_bUseEX)
				{
					nRet = (long)SECITPDK_QueryPositionsEx(cusreqinfo, SORT_TYPE_AES, nRows, tmp, "", "", "", 1, argHolding);
				}
				else
				{
					nRet = SECITPDK_QueryPositions(m_strUser.c_str(), SORT_TYPE_AES, nRows, tmp, "", "", "", 1, argHolding);  /// execflag�������0
				}

				// ����ҳ��ѯ������뵽�ܵĽ����
				if (nRet >= 0)
					argTotalHolding.insert(argTotalHolding.end(), argHolding.begin(), argHolding.end());

				tmp += 200;
			}

			WTSArray* ayPositions = WTSArray::create();

			write_log(m_traderSink, LL_INFO, "[TraderHTS]�û��ֲֲ�ѯ�ɹ������ؽ����{}", nRet);

			for (auto& itr : argTotalHolding)
			{

				const char* exchg = exchgO2I(itr.Market);
				const char* code = itr.StockCode;

				//std::cout << "query positions: " << " code: " << code << " exchange: " << exchg << " dynamic profit: " << itr.UnclearProfit << " qty: " << (long)itr.CurrentQty << " brow index: " << (long)itr.BrowIndex << std::endl;
				
				WTSContractInfo* contract = m_bdMgr->getContract(code, exchg);
				if (contract)
				{
					WTSCommodityInfo* commInfo = contract->getCommInfo();
					WTSPositionItem* pInfo = WTSPositionItem::create(code, commInfo->getCurrency(), exchg);
					pInfo->setDirection(WDT_LONG);

					double prevol = itr.PreQty;	//����ĳֲ֣������ǲ�����
					double newvol = itr.CurrentQty;	//����ĳֲ֣��������ʵ�ֲ�
					double openvol = itr.RealBuyQty;	//������������
					double closevol = itr.RealSellQty;	//������������

					pInfo->setPrePosition(prevol - closevol);
					pInfo->setNewPosition(openvol);
					pInfo->setAvailPrePos(pInfo->getPrePosition());

					pInfo->setAvgPrice(itr.CostPrice);  /// �ֲ־���
					pInfo->setDynProfit(itr.UnclearProfit);  /// ����ӯ��

					double cost = itr.CostBalance;  /// �ֲֳɱ�
					pInfo->setMargin(cost);
					pInfo->setPositionCost(cost);
					if (pInfo->getTotalPosition() > 0)
						pInfo->setAvgPrice(cost / pInfo->getTotalPosition());

					ayPositions->append(pInfo, false);
				}
			}

			if (m_traderSink) m_traderSink->onRspPosition(ayPositions);
			ayPositions->release();
		}

		m_bInQuery = false;
	});

	triggerQuery();

	return 0;
}

int TraderHTS::queryOrders()
{
	if (m_wrapperState != WS_ALLREADY)
	{
		return -1;
	}

	StdUniqueLock lock(m_mtxQuery);
	m_queQuery.push([this]() {
		std::vector<ITPDK_DRWT> argOrders;
		const int nRows = 200;
		argOrders.reserve(nRows);  //��ҪԤ�����㹻�ռ䣬��ѯ�����󷵻�200��������Ļ���Ҫ��ҳ��ѯ

		int64 nRet = 0;
		if (m_bUseEX)
		{
			nRet = (long)SECITPDK_QueryOrdersEx(cusreqinfo, 0, SORT_TYPE_AES, nRows, 0, "", "", 0, argOrders);
		}
		else
		{
			nRet = (long)SECITPDK_QueryOrders(m_strUser.c_str(), 0, SORT_TYPE_AES, nRows, 0, "", "", 0, argOrders);
		}

		if (nRet < 0)  /// ��ѯʧ��
		{
			std::string msg = SECITPDK_GetLastError();
			write_log(m_traderSink, LL_ERROR, "[TraderHTS]�û�������ѯʧ��: {}({})", msg.c_str(), nRet);
		}
		else
		{
			// ��������浽һ���ܵ�vector����ڷ�ҳ
			std::vector<ITPDK_DRWT> argTotalOrders(argOrders);
			int tmp = nRows;

			// ������������200�����ҳ��ѯ
			while (nRet >= nRows)
			{
				argOrders.clear();

				// ѭ����ѯ��ʱ��nBrowindex+1������һ��
				if (m_bUseEX)
				{
					nRet = (long)SECITPDK_QueryOrdersEx(cusreqinfo, 0, SORT_TYPE_AES, nRows, tmp, "", "", 0, argOrders);
				}
				else
				{
					nRet = (long)SECITPDK_QueryOrders(m_strUser.c_str(), 0, SORT_TYPE_AES, nRows, tmp, "", "", 0, argOrders);
				}

				// ����ҳ��ѯ������뵽�ܵĽ����
				if (nRet >= 0)
					argTotalOrders.insert(argTotalOrders.end(), argOrders.begin(), argOrders.end());

				tmp += nRows;
			}

			WTSArray* ayOrds = WTSArray::create();

			for (auto& itr : argTotalOrders)
			{
				/*printf("�˻�: %s, ί�к�: %ld, �����̱��ر��: %ld, �걨ί�к�: %s, ����: %s, ������: %s, �۸�: %f, ����: %ld, ��ѯҳ��: %ld \n",
					itr.AccountId, (long)itr.OrderId, (long)itr.KFSBDBH, itr.SBWTH, itr.StockCode, itr.Market, itr.OrderPrice, (long)itr.OrderQty, (long)itr.BrowIndex);*/

				const char* exchg = exchgO2I(itr.Market);
				const char* code = itr.StockCode;

				WTSContractInfo* contract = m_bdMgr->getContract(code, exchg);
				if (contract)
				{
					WTSCommodityInfo* commInfo = contract->getCommInfo();
					WTSOrderInfo* ordInfo = WTSOrderInfo::create();
					ordInfo->setCode(code);
					ordInfo->setExchange(exchg);

					ordInfo->setPrice(itr.OrderPrice);
					ordInfo->setDirection(WDT_LONG);
					ordInfo->setPriceType(WPT_LIMITPRICE);  //TODO
					ordInfo->setOffsetType(itr.EntrustType == 1 ? WOT_OPEN : WOT_CLOSE);

					double total = itr.OrderQty;
					double traded = itr.MatchQty;
					double canceled = itr.WithdrawQty;
					ordInfo->setVolume(total);
					ordInfo->setVolTraded(traded);
					ordInfo->setVolLeft(total - canceled - traded);

					ordInfo->setOrderDate((uint32_t)itr.EntrustDate);  /// ί������

					uint32_t uTime = strToTime(itr.EntrustTime);   /// ί��ʱ��
					ordInfo->setOrderTime(TimeUtils::makeTime(ordInfo->getOrderDate(), uTime));

					int state = itr.OrderStatus;
					ordInfo->setOrderState(wrapOrderState(state));
					if (state == 3)
						ordInfo->setError(true);

					ordInfo->setEntrustID(std::to_string(itr.KFSBDBH).c_str());  /// ����ʹ�ÿ����̱�� 

					ordInfo->setOrderID(std::to_string(itr.OrderId).c_str());  /// ����ʹ��ί�к�

					ordInfo->setStateMsg(itr.ResultInfo);  /// ���˵��

					std::string usertag = m_iniHelper.readString(ENTRUST_SECTION, ordInfo->getEntrustID(), "");
					if (usertag.empty())
					{
						ordInfo->setUserTag(ordInfo->getEntrustID());
					}
					else
					{
						ordInfo->setUserTag(usertag.c_str());

						if (strlen(ordInfo->getOrderID()) > 0)
						{
							m_iniHelper.writeString(ORDER_SECTION, StrUtil::trim(ordInfo->getOrderID()).c_str(), usertag.c_str());
							m_iniHelper.save();
						}
					}

					if (ordInfo->isAlive())
					{
						if (m_mapLives == NULL)
							m_mapLives = TradeDataMap::create();

						m_mapLives->add(ordInfo->getOrderID(), ordInfo, true);
					}

					ayOrds->append(ordInfo, false);
				}
			}


			if (m_traderSink) m_traderSink->onRspOrders(ayOrds);
			ayOrds->release();
		}

		m_bInQuery = false;
	});

	triggerQuery();

	return 0;
}

int TraderHTS::queryTrades()
{
	if (m_wrapperState != WS_ALLREADY)
	{
		return -1;
	}

	StdUniqueLock lock(m_mtxQuery);
	m_queQuery.push([this]() {
		std::vector<ITPDK_SSCJ> argTrades;
		const size_t nRows = 200;
		argTrades.reserve(nRows);  //��ҪԤ�����㹻�ռ䣬��ѯ�����󷵻�200��

		int64 nRet = 0;
		if (m_bUseEX)
		{
			nRet = (long)SECITPDK_QueryMatchsEx(cusreqinfo, 0, SORT_TYPE_AES, nRows, 0, "", "", 0, argTrades);
		}
		else
		{
			nRet = (long)SECITPDK_QueryMatchs(m_strUser.c_str(), 0, SORT_TYPE_AES, nRows, 0, "", "", 0, argTrades);
		}

		if (nRet < 0)
		{
			std::string msg = SECITPDK_GetLastError();
			write_log(m_traderSink, LL_ERROR, "[TraderHTS]�û��ɽ���ѯʧ��: {}({})", msg.c_str(), nRet);
		}
		else
		{
			// ��������浽һ���ܵ�vector����ڷ�ҳ
			std::vector<ITPDK_SSCJ> argTotalTrades(argTrades);
			int tmp = 200;

			// ������������200�����ҳ��ѯ
			while (nRet >= 200)
			{
				argTrades.clear();

				if (m_bUseEX)
				{
					nRet = (long)SECITPDK_QueryMatchsEx(cusreqinfo, 0, SORT_TYPE_AES, nRows, tmp, "", "", 0, argTrades);
				}
				else
				{
					nRet = (long)SECITPDK_QueryMatchs(m_strUser.c_str(), 0, SORT_TYPE_AES, nRows, tmp, "", "", 0, argTrades);
				}

				// ����ҳ��ѯ������뵽�ܵĽ����
				if (nRet >= 0)
					argTotalTrades.insert(argTotalTrades.end(), argTrades.begin(), argTrades.end());

				tmp += 200;
			}

			WTSArray* ayTrds = WTSArray::create();

			int index = 0;
			for (auto& itr : argTotalTrades)
			{
				//printf("ί�к�: %ld, ����: %s ������: %s, ��ѯҳ��: %ld, �ɽ��۸�: %f, �ɽ�����: %d\n",
				//	(long)itr.OrderId, itr.StockCode, itr.Market, (long)itr.BrowIndex, itr.MatchPrice, (long)itr.MatchQty);
				
				const char* exchg = exchgO2I(itr.Market);
				const char* code = itr.StockCode;

				WTSContractInfo* contract = m_bdMgr->getContract(code, exchg);
				if (contract)
				{
					WTSCommodityInfo* commInfo = contract->getCommInfo();
					WTSTradeInfo *trdInfo = WTSTradeInfo::create(code, exchg);
					trdInfo->setPrice(itr.MatchPrice);
					trdInfo->setVolume(itr.MatchQty);
					trdInfo->setTradeID(itr.MatchSerialNo);  // ����ʹ�óɽ����

					trdInfo->setTradeDate(m_lDate);
					trdInfo->setTradeTime(strToTime(itr.MatchTime));

					// #################### ��Ҫ����###############
					trdInfo->setDirection(WDT_LONG);
					trdInfo->setOffsetType(itr.EntrustType == 1 ? WOT_OPEN : WOT_CLOSE);
					trdInfo->setRefOrder(std::to_string(itr.KFSBDBH).c_str());  // ����ʹ�ÿ����̱��
					trdInfo->setTradeType(WTT_Common);

					trdInfo->setAmount(itr.MatchAmt);

					std::string usertag = m_iniHelper.readString(ORDER_SECTION, StrUtil::trim(trdInfo->getRefOrder()).c_str());
					if (!usertag.empty())
						trdInfo->setUserTag(usertag.c_str());

					ayTrds->append(trdInfo, false);
				}
			}

			if (m_traderSink) m_traderSink->onRspTrades(ayTrds);
			ayTrds->release();
		}

		m_bInQuery = false;
	});

	triggerQuery();

	return 0;
}

bool TraderHTS::isConnected()
{
	return (m_wrapperState == WS_ALLREADY);
}

void TraderHTS::triggerQuery()
{
	m_strandIO->post([this]() {
		if (m_queQuery.empty() || m_bInQuery)
			return;

		//����ӿں���û���������ƣ���ȥ��
		//uint64_t curTime = TimeUtils::getLocalTimeNow();
		//if (curTime - m_lastQryTime < 1000)
		//{
		//	boost::this_thread::sleep(boost::posix_time::milliseconds(50));
		//	m_strandIO->post([this](){
		//		triggerQuery();
		//	});
		//	return;
		//}

		m_bInQuery = true;
		CommonExecuter& handler = m_queQuery.front();
		handler();

		{
			StdUniqueLock lock(m_mtxQuery);
			m_queQuery.pop();
		}

		m_lastQryTime = TimeUtils::getLocalTimeNow();
	});
}

HTSCallMgr::HTSCallMgr()
{

}

HTSCallMgr::~HTSCallMgr()
{

}

bool HTSCallMgr::init(WTSVariant* params)
{
	if (params)
	{
		SECITPDK_SetLogPath("./");            //��־Ŀ¼
		SECITPDK_SetProfilePath("./");           //�����ļ�Ŀ¼

		bool b_inited = false;
		b_inited = SECITPDK_Init(HEADER_VER);  //��ʼ���������нӿ�ʹ��ǰ���ã���·�����ýӿ���

		if (b_inited)
		{
			SECITPDK_SetWriteLog(true);  // ��ӡ��־
			SECITPDK_SetFixWriteLog(true);

			SECITPDK_SetWTFS(params->getCString("order_way"));               //����ί�з�ʽ
			SECITPDK_SetNode(params->getCString("node"));               //����վ����Ϣ

			char sVer[64] = { 0 };
			SECITPDK_GetVersion(sVer);
			std::cout << "SECITPDK Version: " << sVer << std::endl;
		}

		return b_inited;
	}

	return false;
}

void HTSCallMgr::setHTSCallPtr(std::string khh, TraderHTS *pTrader)
{
	if (itpdkCallMap.find(khh) == itpdkCallMap.end())
	{
		//std::cout << "Current key not existed, insert it to address map." << std::endl;
		setCallbackMsgFunc();  // ������Ϣ

		itpdkCallMap.insert(std::pair<string, TraderHTS *>(khh, pTrader));
	}
}

void HTSCallMgr::callbackMsgFunc(const char* pTime, const char* pMsg, int nType)
{
	// �Ƚ������ݽ���
	//std::cout << "HTSCallMgr callbackMsgFunc begins ..." << std::endl;

	rapidjson::Document document;
	if (document.Parse(pMsg).HasParseError())
	{
		//cout << "HTSCallManage callbackMsgFunc Parse confirm message failed." << endl;
		return;
	}

	//long nWTH = (long)document["WTH"].GetInt64();
	const char* sKHH = document["KHH"].GetString();

	//����itpdkCallMap
	for (auto iter = itpdkCallMap.begin(); iter != itpdkCallMap.end(); ++iter) {
		string sTmp = iter->first;
		if (strcmp(sTmp.c_str(), sKHH) == 0) {
			TraderHTS* pTrader = iter->second;
			if (pTrader != NULL) {
				//std::cout << "HTSCallManage get report message khh: " << iter->first << "  " << iter->second << std::endl;

				//�˴����������Ľ���������ݷ�װ��һ���ṹ����ٴ���HTSCall��ʵ���У��������ν������˷���Դ���˴�ֻ�Ǽ�ʾ��������û�ж���ر��ṹ��
				pTrader->htsGetCallback(pTime, pMsg, nType);
			}
		}
	}
}

void HTSCallMgr::callbackConnFunc(const char* pKhh, const char* pConnKey, int nEvent, int nType)
{
	string sType;
	if (NOTIFY_CONNEVENT_MGR == nType)
		sType = "��������";
	if (NOTIFY_CONNEVENT_TRADE == nType)
		sType = "��������";
	if (NOTIFY_CONNEVENT_QUERY == nType)
		sType = "��ѯ����";
	if (NOTIFY_CONNEVENT_SUBS == nType)
		sType = "��������";

	printf("khh=%s, event=%s, type=%s\n", pKhh, (nEvent == 0 ? "�ָ�" : "�Ͽ�"), sType.c_str());

	if (nEvent != 0)
	{
		// �Ͽ�����
		//����itpdkCallMap
		for (auto iter = itpdkCallMap.begin(); iter != itpdkCallMap.end(); ++iter) {
			string sTmp = iter->first;
			if (strcmp(sTmp.c_str(), pKhh) == 0) {
				TraderHTS* pTrader = iter->second;
				if (pTrader != NULL) {
					printf("[TraderHTS] ���ڳ��Իָ�����, �ͻ��� khh: %s\n", iter->first);

					//�˴����������Ľ���������ݷ�װ��һ���ṹ����ٴ���HTSCall��ʵ���У��������ν������˷���Դ���˴�ֻ�Ǽ�ʾ��������û�ж���ر��ṹ��
					pTrader->connect();
				}
			}
		}
	}
}

void HTSCallMgr::setCallbackMsgFunc()
{
	//���ûص�����
	SECITPDK_SetConnEventCallback(&HTSCallMgr::callbackConnFunc);  // �������ӻص�
	SECITPDK_SetMsgCallback(&HTSCallMgr::callbackMsgFunc);  // ������Ϣ���ͻص�
	SECITPDK_SetFuncCallback(&HTSCallMgr::callbackMsgFunc);  // �����첽���ͻص�
}
