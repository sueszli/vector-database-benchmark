
#include "driver/uart.h"  //����0��Ҫ��ͷ�ļ�
#include "osapi.h"  //����1��Ҫ��ͷ�ļ�
#include "user_interface.h" //WIFI������Ҫ��ͷ�ļ�
#include "espconn.h"//TCP������Ҫ��ͷ�ļ�
#include "mem.h" //ϵͳ������Ҫ��ͷ�ļ�


struct espconn user_udp_espconn;
os_timer_t checkTimer_wifistate;

void ICACHE_FLASH_ATTR user_udp_sent_cb(void *arg)   //����
{
	os_printf("\r\n���ͳɹ���\r\n");

}

void ICACHE_FLASH_ATTR user_udp_recv_cb(void *arg,    //����
		char *pdata, unsigned short len) {
	os_printf("�������ݣ�%s", pdata);

	//ÿ�η�������ȷ����������
	user_udp_espconn.proto.udp = (esp_udp *) os_zalloc(sizeof(esp_udp));
	user_udp_espconn.type = ESPCONN_UDP;
	user_udp_espconn.proto.udp->local_port = 2000;
	user_udp_espconn.proto.udp->remote_port = 8686;
	const char udp_remote_ip[4] = { 255, 255, 255, 255 };
	os_memcpy(user_udp_espconn.proto.udp->remote_ip, udp_remote_ip, 4);

	espconn_sent((struct espconn *) arg, "�Ѿ��յ�����", strlen("�Ѿ��յ���!"));
}

void Check_WifiState(void) {

	uint8 getState = wifi_station_get_connect_status();

	//���״̬��ȷ��֤���Ѿ�����
	if (getState == STATION_GOT_IP) {

		os_printf("WIFI���ӳɹ���");
		os_timer_disarm(&checkTimer_wifistate);

		wifi_set_broadcast_if(0x01);	 //���� ESP8266 ���� UDP�㲥��ʱ���� station �ӿڷ���
		user_udp_espconn.proto.udp = (esp_udp *) os_zalloc(sizeof(esp_udp));//����ռ�
		user_udp_espconn.type = ESPCONN_UDP;	 		  //��������ΪUDPЭ��
		user_udp_espconn.proto.udp->local_port = 2000;	 		  //���ض˿�
		user_udp_espconn.proto.udp->remote_port = 8686;	 		  //Ŀ��˿�
		const char udp_remote_ip[4] = { 255, 255, 255, 255 };	 	//Ŀ��IP��ַ���㲥��
		os_memcpy(user_udp_espconn.proto.udp->remote_ip, udp_remote_ip, 4);

		espconn_regist_recvcb(&user_udp_espconn, user_udp_recv_cb);	 		//����
		espconn_regist_sentcb(&user_udp_espconn, user_udp_sent_cb);	 		//����
		espconn_create(&user_udp_espconn);	 		  //���� UDP ����
		espconn_sent(&user_udp_espconn, "���ӷ�����", strlen("���ӷ�����"));

	}
}

void udp_client_init() //��ʼ��
{
	wifi_set_opmode(0x01); //����ΪSTATIONģʽ
	struct station_config stationConf;
	os_strcpy(stationConf.ssid, "meizu");	  //�ĳ���Ҫ���ӵ� ·�������û���
	os_strcpy(stationConf.password, "12345678"); //�ĳ���Ҫ���ӵ�·����������

	wifi_station_set_config(&stationConf);	  //����WiFi station�ӿ����ã������浽 flash
	wifi_station_connect();	  //����·����
	os_timer_disarm(&checkTimer_wifistate);	  //ȡ����ʱ����ʱ
	os_timer_setfn(&checkTimer_wifistate, (os_timer_func_t *) Check_WifiState,
	NULL);	  //���ö�ʱ���ص�����
	os_timer_arm(&checkTimer_wifistate, 500, 1);	  //������ʱ������λ������
}

