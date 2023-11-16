#include <string>
#include <map>
//v6.3.15
#include "../API/CTP6.3.15/ThostFtdcTraderApi.h"
#include "TraderSpi.h"

#include "../Share/IniHelper.hpp"
#include "../Share/ModuleHelper.hpp"
#include "../Share/DLLHelper.hpp"
#include "../Share/StdUtils.hpp"

#include "../Share/charconv.hpp"

#include "../WTSUtils/WTSCfgLoader.h"
#include "../Includes/WTSVariant.hpp"
USING_NS_WTP;

#include <boost/filesystem.hpp>

// UserApi����
CThostFtdcTraderApi* pUserApi;

// ���ò���
std::string	FRONT_ADDR;	// ǰ�õ�ַ
std::string	BROKER_ID;	// ���͹�˾����
std::string	INVESTOR_ID;// Ͷ���ߴ���
std::string	PASSWORD;	// �û�����
std::string SAVEPATH;	//����λ��
std::string APPID;
std::string AUTHCODE;
uint32_t	CLASSMASK;	//��Ȩ
bool		ONLYINCFG;	//ֻ��������ļ��е�

std::string COMM_FILE;		//�����Ʒ���ļ���
std::string CONT_FILE;		//����ĺ�Լ�ļ���

std::string MODULE_NAME;	//�ⲿģ����

typedef std::map<std::string, std::string>	SymbolMap;
SymbolMap	MAP_NAME;
SymbolMap	MAP_SESSION;

typedef CThostFtdcTraderApi* (*CTPCreator)(const char *);
CTPCreator		g_ctpCreator = NULL;

// ������
int iRequestID = 0;

#ifdef _MSC_VER
#	define EXPORT_FLAG __declspec(dllexport)
#else
#	define EXPORT_FLAG __attribute__((__visibility__("default")))
#endif

#ifdef __cplusplus
extern "C"
{
#endif
	EXPORT_FLAG int run(const char* cfgfile, bool bAsync, bool isFile);
#ifdef __cplusplus
}
#endif

int run(const char* cfgfile, bool bAsync = false, bool isFile = true)
{
	std::string map_files;

	if(!isFile)
	{
		WTSVariant* root = WTSCfgLoader::load_from_content(cfgfile, true);
		if (root == NULL)
			return 0;

		WTSVariant* ctp = root->get("ctp");
		FRONT_ADDR = ctp->getCString("front");
		BROKER_ID = ctp->getCString("broker");
		INVESTOR_ID = ctp->getCString("user");
		PASSWORD = ctp->getCString("pass");
		APPID = ctp->getCString("appid");
		AUTHCODE = ctp->getCString("authcode");

		WTSVariant* cfg = root->get("config");
		SAVEPATH = cfg->getCString("path");
		CLASSMASK = cfg->getUInt32("mask"); //1-�ڻ�,2-��Ȩ,4-��Ʊ

		COMM_FILE = cfg->getCString("commfile");
		if (COMM_FILE.empty())
			COMM_FILE = "commodities.json";

		CONT_FILE = cfg->getCString("contfile");
		if (CONT_FILE.empty())
			CONT_FILE = "contracts.json";

		map_files = cfg->getCString("mapfiles");
		ONLYINCFG = ctp->getBoolean("onlyincfg");

		MODULE_NAME = ctp->getCString("module");
		if (MODULE_NAME.empty())
		{
#ifdef _WIN32
			MODULE_NAME = "thosttraderapi_se.dll";
#else
			MODULE_NAME = "thosttraderapi_se.so";
#endif
		}

		root->release();
	}
	else if(StrUtil::endsWith(cfgfile, ".ini"))
	{
		IniHelper ini;

		ini.load(cfgfile);

		FRONT_ADDR = ini.readString("ctp", "front", "");
		BROKER_ID = ini.readString("ctp", "broker", "");
		INVESTOR_ID = ini.readString("ctp", "user", "");
		PASSWORD = ini.readString("ctp", "pass", "");
		APPID = ini.readString("ctp", "appid", "");
		AUTHCODE = ini.readString("ctp", "authcode", "");

		SAVEPATH = ini.readString("config", "path", "");
		CLASSMASK = ini.readUInt("config", "mask", 1 | 2 | 4); //1-�ڻ�,2-��Ȩ,4-��Ʊ
		ONLYINCFG = wt_stricmp(ini.readString("config", "onlyincfg", "false").c_str(), "true") == 0;

		COMM_FILE = ini.readString("config", "commfile", "commodities.json");
		CONT_FILE = ini.readString("config", "contfile", "contracts.json");

		map_files = ini.readString("config", "mapfiles", "");

#ifdef _WIN32
		MODULE_NAME = ini.readString("ctp", "module", "thosttraderapi_se.dll");
#else
		MODULE_NAME = ini.readString("ctp", "module", "thosttraderapi_se.so");
#endif
	}
	else
	{
		WTSVariant* root = WTSCfgLoader::load_from_file(cfgfile);
		if (root == NULL)
			return 0;

		WTSVariant* ctp = root->get("ctp");
		FRONT_ADDR = ctp->getCString("front");
		BROKER_ID = ctp->getCString("broker");
		INVESTOR_ID = ctp->getCString("user");
		PASSWORD = ctp->getCString("pass");
		APPID = ctp->getCString("appid");
		AUTHCODE = ctp->getCString("authcode");

		WTSVariant* cfg = root->get("config");
		SAVEPATH = cfg->getCString("path"); 
		CLASSMASK = cfg->getUInt32("mask"); //1-�ڻ�,2-��Ȩ,4-��Ʊ

		COMM_FILE = cfg->getCString("commfile");
		if (COMM_FILE.empty())
			COMM_FILE = "commodities.json";

		CONT_FILE = cfg->getCString("contfile"); 
		if(CONT_FILE.empty())
			CONT_FILE = "contracts.json";

		map_files = cfg->getCString("mapfiles");
		ONLYINCFG = ctp->getBoolean("onlyincfg");

		MODULE_NAME = ctp->getCString("module");
		if(MODULE_NAME.empty())
		{
#ifdef _WIN32
			MODULE_NAME = "thosttraderapi_se.dll";
#else
			MODULE_NAME = "thosttraderapi_se.so";
#endif
		}
		root->release();
	}
	
	if(!boost::filesystem::exists(MODULE_NAME.c_str()))
	{
		MODULE_NAME = StrUtil::printf("%straders/%s", getBinDir(), MODULE_NAME.c_str());
	}

	if(FRONT_ADDR.empty() || BROKER_ID.empty() || INVESTOR_ID.empty() || PASSWORD.empty() || SAVEPATH.empty())
	{
		return 0;
	}

	SAVEPATH = StrUtil::standardisePath(SAVEPATH);

	
	if(!map_files.empty())
	{
		StringVector ayFiles = StrUtil::split(map_files, ",");
		for(const std::string& fName:ayFiles)
		{
			printf("Reading mapping file %s...\r\n", fName.c_str());
			IniHelper iniMap;
			if(!StdFile::exists(fName.c_str()))
				continue;

			iniMap.load(fName.c_str());
			FieldArray ayKeys, ayVals;
			int cout = iniMap.readSecKeyValArray("Name", ayKeys, ayVals);
			for (int i = 0; i < cout; i++)
			{
				std::string pName = ayVals[i];
				bool isUTF8 = EncodingHelper::isUtf8((unsigned char*)pName.c_str(), pName.size());
				if (!isUTF8)
					pName = ChartoUTF8(ayVals[i]);
				//�����ʱ��ȫ��ת��UTF8
				MAP_NAME[ayKeys[i]] = pName;
#ifdef _WIN32
				printf("Commodity name mapping: %s - %s\r\n", ayKeys[i].c_str(), isUTF8 ? UTF8toChar(ayVals[i]).c_str() : ayVals[i].c_str());
#else
				printf("Commodity name mapping: %s - %s\r\n", ayKeys[i].c_str(), isUTF8 ? ayVals[i].c_str() : ChartoUTF8(ayVals[i]).c_str());
#endif
			}

			ayKeys.clear();
			ayVals.clear();
			cout = iniMap.readSecKeyValArray("Session", ayKeys, ayVals);
			for (int i = 0; i < cout; i++)
			{
				MAP_SESSION[ayKeys[i]] = ayVals[i];
				printf("Trading session mapping: %s - %s\r\n", ayKeys[i].c_str(), ayVals[i].c_str());
			}
		}
		
	}

	// ��ʼ��UserApi
	DllHandle dllInst = DLLHelper::load_library(MODULE_NAME.c_str());
	if (dllInst == NULL)
		printf("Loading module %s failed\r\n", MODULE_NAME.c_str());
#ifdef _WIN32
#	ifdef _WIN64
	g_ctpCreator = (CTPCreator)DLLHelper::get_symbol(dllInst, "?CreateFtdcTraderApi@CThostFtdcTraderApi@@SAPEAV1@PEBD@Z");
#	else
	g_ctpCreator = (CTPCreator)DLLHelper::get_symbol(dllInst, "?CreateFtdcTraderApi@CThostFtdcTraderApi@@SAPAV1@PBD@Z");
#	endif
#else
	g_ctpCreator = (CTPCreator)DLLHelper::get_symbol(dllInst, "_ZN19CThostFtdcTraderApi19CreateFtdcTraderApiEPKc");
#endif
	if (g_ctpCreator == NULL)
		printf("Loading CreateFtdcTraderApi failed\r\n");
	pUserApi = g_ctpCreator("");			// ����UserApi
	CTraderSpi* pUserSpi = new CTraderSpi();
	pUserApi->RegisterSpi((CThostFtdcTraderSpi*)pUserSpi);			// ע���¼���
	pUserApi->SubscribePublicTopic(THOST_TERT_QUICK);					// ע�ṫ����
	pUserApi->SubscribePrivateTopic(THOST_TERT_QUICK);					// ע��˽����
	pUserApi->RegisterFront((char*)FRONT_ADDR.c_str());				// connect
	pUserApi->Init();

    //��������첽����ȴ�API����
    if(!bAsync)
	    pUserApi->Join();

	return 0;
}