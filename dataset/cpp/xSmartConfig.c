#include "include/xSmartConfig.h"
#include "mem.h"
#include "ip_addr.h"
#include "espconn.h"
#include "osapi.h"
#include "user_interface.h"
#include "smartconfig.h"
#include "airkiss.h"
#include "user_interface.h"

int flag = 0;
os_timer_t mTimerXSmartConfig;
xSmartConfig_CallBack statusCodeDriver;
struct station_config *sta_conf;
void ICACHE_FLASH_ATTR smartconfig_done(sc_status status, void *pdata) {

	switch (status) {

	case SC_STATUS_FIND_CHANNEL:
		statusCodeDriver(xSmartConfig_Status_Get_Pas);
		break;

	case SC_STATUS_GETTING_SSID_PSWD:
		statusCodeDriver(xSmartConfig_Status_Get_Pas);
		break;

	case SC_STATUS_LINK:
		os_strcpy(&sta_conf, &pdata);
		wifi_station_set_config(sta_conf);
		wifi_station_disconnect();
		wifi_station_connect();
		statusCodeDriver(xSmartConfig_Status_Connectting_GateWay);
		break;

	case SC_STATUS_LINK_OVER:
		if (pdata != NULL) {
			statusCodeDriver(xSmartConfig_Status_Succeed);
			flag = 40;
		}
		smartconfig_stop();
		break;
	}

}

//ע��ص�����
void register_xSmartConfigCallBack(xSmartConfig_CallBack tempCallBack) {
	statusCodeDriver = tempCallBack;
}

//���r���ص�����
void Check_WifiState() {

	flag++;
	if (flag > 35) {
		struct ip_info ipConfig;
		wifi_get_ip_info(STATION_IF, &ipConfig);
		u8 getState = wifi_station_get_connect_status();
		if (!(getState == STATION_GOT_IP && ipConfig.ip.addr != 0)) {
			statusCodeDriver(xSmartConfig_Status_Failed);
		}
		os_timer_disarm(&mTimerXSmartConfig); //������ʱ��ǰ��ȡ����ʱ����ʱ
	}

}

//��ʼһ������
void xSmartConfig_Start() {

	statusCodeDriver(xSmartConfig_Status_Connectting_Early);

	smartconfig_set_type(SC_TYPE_ESPTOUCH);
	wifi_set_opmode(STATION_MODE);
	smartconfig_start(smartconfig_done);

	os_timer_disarm(&mTimerXSmartConfig); //������ʱ��ǰ��ȡ����ʱ����ʱ
	os_timer_setfn(&mTimerXSmartConfig, (os_timer_func_t *) Check_WifiState,
	NULL); //���ö�ʱ���ص�����
	os_timer_arm(&mTimerXSmartConfig, 2000, 1); //������ʱ��

}
