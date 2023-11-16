/*!
 * \file ParserYD.cpp
 * \project	WonderTrader
 *
 * \author Wesley
 * \date 2020/03/30
 * 
 * \brief 
 */
#include "ParserYD.h"

#include "../Includes/WTSDataDef.hpp"
#include "../Includes/WTSContractInfo.hpp"
#include "../Includes/WTSVariant.hpp"
#include "../Includes/IBaseDataMgr.h"

#include "../Share/ModuleHelper.hpp"
#include "../Share/TimeUtils.hpp"
#include "../Share/StdUtils.hpp"

#include <boost/filesystem.hpp>

 //By Wesley @ 2022.01.05
#include "../Share/fmtlib.h"
template<typename... Args>
inline void write_log(IParserSpi* sink, WTSLogLevel ll, const char* format, const Args&... args)
{
	if (sink == NULL)
		return;

	static thread_local char buffer[512] = { 0 };
	memset(buffer, 0, 512);
	fmt::format_to(buffer, format, args...);

	sink->handleParserLog(ll, buffer);
}

extern "C"
{
	EXPORT_FLAG IParserApi* createParser()
	{
		ParserYD* parser = new ParserYD();
		return parser;
	}

	EXPORT_FLAG void deleteParser(IParserApi* &parser)
	{
		if (NULL != parser)
		{
			delete parser;
			parser = NULL;
		}
	}
};


ParserYD::ParserYD()
	: m_pUserAPI(NULL)
	, m_uTradingDate(0)
	, m_bApiInited(false)
{
}


ParserYD::~ParserYD()
{
	m_pUserAPI = NULL;
}

void ParserYD::notifyReadyForLogin(bool hasLoginFailed)
{
	if (m_sink)
	{
		write_log(m_sink, LL_INFO, "[ParserYD] Market data server connected");
		m_sink->handleEvent(WPE_Connect, 0);
	}

	DoLogin();
}


void ParserYD::notifyLogin(int errorNo, int maxOrderRef, bool isMonitor)
{
	if (errorNo == 0)
	{
		write_log(m_sink, LL_INFO, "[ParserYD] {} login successfully", m_strUserID);

		m_uTradingDate = m_pUserAPI->getTradingDay();
		if (m_sink)
		{
			m_sink->handleEvent(WPE_Login, 0);
		}

		//���API��ʼ�����ˣ���ֱ�Ӷ���
		//����һ���Ƕ�������
		if (m_bApiInited)
		{
			//������������
			DoSubscribe();
		}
	}
	else
	{
		write_log(m_sink, LL_INFO, "[ParserYD] {} login failed, error no: {}", m_strUserID, errorNo);
	}
}


void ParserYD::notifyFinishInit(void)
{
	if (m_sink) m_sink->handleParserLog(LL_INFO, "[ParserYD] All things ready");

	//���APIû�г�ʼ������˵������ص��ǵ�һ�δ���
	//����Ҫ����һЩ���ĵȲ���
	if(!m_bApiInited)
	{
		//������������
		DoSubscribe();
		m_bApiInited = true;
	}
}


void ParserYD::notifyMarketData(const YDMarketData *pDepthMarketData)
{
	if (m_pBaseDataMgr == NULL)
		return;

	const YDInstrument* instInfo = pDepthMarketData->m_pInstrument;
	const YDExchange* exchgInfo = instInfo->m_pExchange;
	uint32_t actDate = pDepthMarketData->TradingDay;
	uint32_t actTime = pDepthMarketData->TimeStamp;
	uint32_t actHour = actTime / 10000000;

	WTSContractInfo* contract = m_pBaseDataMgr->getContract(instInfo->InstrumentID, exchgInfo->ExchangeID);
	if (contract == NULL)
		return;

	WTSTickData* tick = WTSTickData::create(instInfo->InstrumentID);
	WTSTickStruct& quote = tick->getTickStruct();
	wt_strcpy(quote.exchg, contract->getExchg());
	tick->setContractInfo(contract);

	quote.action_date = actDate;
	quote.action_time = actTime;

	quote.price = pDepthMarketData->LastPrice;
	quote.total_volume = pDepthMarketData->Volume;
	quote.trading_date = m_uTradingDate;
	quote.total_turnover = pDepthMarketData->Turnover;

	quote.open_interest = pDepthMarketData->OpenInterest;

	quote.upper_limit = pDepthMarketData->UpperLimitPrice;
	quote.lower_limit = pDepthMarketData->LowerLimitPrice;

	quote.pre_close = pDepthMarketData->PreClosePrice;
	quote.pre_settle = pDepthMarketData->PreSettlementPrice;
	quote.pre_interest = pDepthMarketData->PreOpenInterest;

	//ί���۸�
	quote.ask_prices[0] = pDepthMarketData->AskPrice;
	//ί��۸�
	quote.bid_prices[0] = pDepthMarketData->BidPrice;
	//ί����
	quote.ask_qty[0] = pDepthMarketData->AskVolume;
	//ί����
	quote.bid_qty[0] = pDepthMarketData->BidVolume;

	if (m_sink)
		m_sink->handleQuote(tick, 1);

	tick->release();
}

bool ParserYD::init(WTSVariant* config)
{
	m_strCfgFile = config->getCString("config");
	m_strUserID = config->getCString("user");
	m_strPassword = config->getCString("pass");

	std::string module = config->getCString("ydmodule");
	if (module.empty())
		module = "yd";

	std::string dllpath = getBinDir() + DLLHelper::wrap_module(module.c_str(), "");
	m_hInstYD = DLLHelper::load_library(dllpath.c_str());	
	m_funcCreator = (YDCreator)DLLHelper::get_symbol(m_hInstYD, "makeYDApi");
	m_pUserAPI = m_funcCreator(m_strCfgFile.c_str());

	return true;
}

void ParserYD::release()
{
	disconnect();
}

bool ParserYD::connect()
{
	if(m_pUserAPI)
	{
		m_pUserAPI->start(this);
	}

	return true;
}

bool ParserYD::disconnect()
{
	if(m_pUserAPI)
	{
		m_pUserAPI->disconnect();
		m_pUserAPI = NULL;
	}

	return true;
}


void ParserYD::DoLogin()
{
	if(m_pUserAPI == NULL)
	{
		return;
	}

	if(!m_pUserAPI->login(m_strUserID.c_str(), m_strPassword.c_str(), NULL, NULL))
	{
		if(m_sink)
			write_log(m_sink, LL_ERROR, "[ParserYD] Sending login request failed");
	}
}

void ParserYD::DoSubscribe()
{
	CodeSet codeFilter = m_filterSubs;
	if(codeFilter.empty())
	{//����������ֻ�յ�,��ȡ��ȫ����Լ�б�
		return;
	}

	int nCount = 0;
	for(auto& code : codeFilter)
	{
		std::size_t pos = code.find('.');
		const YDInstrument* instInfo = NULL;
		if (pos != std::string::npos)
		{
			instInfo = m_pUserAPI->getInstrumentByID((char*)code.c_str() + pos + 1);
		}
		else
		{
			instInfo = m_pUserAPI->getInstrumentByID((char*)code.c_str());
		}

		if(instInfo == NULL)
			continue;

		bool bRet = m_pUserAPI->subscribe(instInfo);
		write_log(m_sink, LL_ERROR, "[ParserYD] Subscribe md of {} {}", code.c_str(), bRet?"succeed":"failed");
	}
}

void ParserYD::subscribe(const CodeSet &vecSymbols)
{
	if(m_uTradingDate == 0)
	{
		m_filterSubs = vecSymbols;
	}
	else
	{
		m_filterSubs = vecSymbols;
		DoSubscribe();
	}
}

void ParserYD::unsubscribe(const CodeSet &vecSymbols)
{
}

bool ParserYD::isConnected()
{
	return m_pUserAPI!=NULL;
}

void ParserYD::registerSpi(IParserSpi* listener)
{
	m_sink = listener;

	if(m_sink)
		m_pBaseDataMgr = m_sink->getBaseDataMgr();
}