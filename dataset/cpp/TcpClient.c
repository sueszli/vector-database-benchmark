
#include "driver/uart.h"  //����0��Ҫ��ͷ�ļ�
#include "osapi.h"  //����1��Ҫ��ͷ�ļ�
#include "user_interface.h" //WIFI������Ҫ��ͷ�ļ�
#include "espconn.h"//TCP������Ҫ��ͷ�ļ�
#include "mem.h" //ϵͳ������Ҫ��ͷ�ļ�
#include "gpio.h"

os_timer_t checkTimer_wifistate;
struct espconn user_tcp_conn;

void ICACHE_FLASH_ATTR user_tcp_sent_cb(void *arg)  //����
{
	os_printf("�������ݳɹ���");
}
void ICACHE_FLASH_ATTR user_tcp_discon_cb(void *arg)  //�Ͽ�
{
	os_printf("�Ͽ����ӳɹ���");
}
void ICACHE_FLASH_ATTR user_tcp_recv_cb(void *arg,  //����
		char *pdata, unsigned short len) {

	os_printf("�յ����ݣ�%s\r\n", pdata);
	espconn_sent((struct espconn *) arg, "0", strlen("0"));

}
void ICACHE_FLASH_ATTR user_tcp_recon_cb(void *arg, sint8 err) //ע�� TCP ���ӷ����쳣�Ͽ�ʱ�Ļص������������ڻص������н�������
{
	os_printf("���Ӵ��󣬴������Ϊ%d\r\n", err);
	espconn_connect((struct espconn *) arg);
}
void ICACHE_FLASH_ATTR user_tcp_connect_cb(void *arg)  //ע�� TCP ���ӳɹ�������Ļص�����
{
	struct espconn *pespconn = arg;
	espconn_regist_recvcb(pespconn, user_tcp_recv_cb);  //����
	espconn_regist_sentcb(pespconn, user_tcp_sent_cb);  //����
	espconn_regist_disconcb(pespconn, user_tcp_discon_cb);  //�Ͽ�
	espconn_sent(pespconn, "8226", strlen("8226"));

}

void ICACHE_FLASH_ATTR my_station_init(struct ip_addr *remote_ip,
		struct ip_addr *local_ip, int remote_port) {
	user_tcp_conn.proto.tcp = (esp_tcp *) os_zalloc(sizeof(esp_tcp));  //����ռ�
	user_tcp_conn.type = ESPCONN_TCP;  //��������ΪTCPЭ��
	os_memcpy(user_tcp_conn.proto.tcp->local_ip, local_ip, 4);
	os_memcpy(user_tcp_conn.proto.tcp->remote_ip, remote_ip, 4);
	user_tcp_conn.proto.tcp->local_port = espconn_port();  //���ض˿�
	user_tcp_conn.proto.tcp->remote_port = remote_port;  //Ŀ��˿�
	//ע�����ӳɹ��ص��������������ӻص�����
	espconn_regist_connectcb(&user_tcp_conn, user_tcp_connect_cb);//ע�� TCP ���ӳɹ�������Ļص�����
	espconn_regist_reconcb(&user_tcp_conn, user_tcp_recon_cb);//ע�� TCP ���ӷ����쳣�Ͽ�ʱ�Ļص������������ڻص������н�������
	//��������
	espconn_connect(&user_tcp_conn);
}

void Check_WifiState(void) {
	uint8 getState;
	getState = wifi_station_get_connect_status();
	//��ѯ ESP8266 WiFi station �ӿ����� AP ��״̬
	if (getState == STATION_GOT_IP) {
		os_printf("WIFI���ӳɹ���\r\n");
		os_timer_disarm(&checkTimer_wifistate);
		struct ip_info info;
		const char remote_ip[4] = { 192, 168, 43, 1 };//Ŀ��IP��ַ,����Ҫ�ȴ��ֻ���ȡ����������ʧ��.
		wifi_get_ip_info(STATION_IF, &info);	//��ѯ WiFiģ��� IP ��ַ
		my_station_init((struct ip_addr *) remote_ip, &info.ip, 6000);//���ӵ�Ŀ���������6000�˿�
 }
}

void tcp_client_init()	//��ʼ��
{

	wifi_set_opmode(0x01);	//����ΪSTATIONģʽ

	struct station_config stationConf;
	os_strcpy(stationConf.ssid, "meizu");	  //�ĳ����Լ���   ·�������û���
	os_strcpy(stationConf.password, "12345678"); //�ĳ����Լ���   ·����������
	wifi_station_set_config(&stationConf);	//����WiFi station�ӿ����ã������浽 flash
	wifi_station_connect();	//����·����

	os_timer_disarm(&checkTimer_wifistate);	//ȡ����ʱ����ʱ
	os_timer_setfn(&checkTimer_wifistate, (os_timer_func_t *) Check_WifiState,
	NULL);	//���ö�ʱ���ص�����
	os_timer_arm(&checkTimer_wifistate, 500, 1);	//������ʱ������λ������
}

