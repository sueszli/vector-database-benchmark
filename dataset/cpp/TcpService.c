#include "driver/uart.h"  //����0��Ҫ��ͷ�ļ�
#include "osapi.h"  //����1��Ҫ��ͷ�ļ�
#include "user_interface.h" //WIFI������Ҫ��ͷ�ļ�
#include "espconn.h"//TCP������Ҫ��ͷ�ļ�
#include "mem.h" //ϵͳ������Ҫ��ͷ�ļ�
#include "gpio.h"

struct espconn user_tcp_espconn;

void ICACHE_FLASH_ATTR server_recv(void *arg, char *pdata, unsigned short len) {
	os_printf("�յ�PC���������ݣ�%s", pdata);
	espconn_sent((struct espconn *) arg, "�Ѿ��յ�����", strlen("�Ѿ��յ�����"));

}
void ICACHE_FLASH_ATTR server_sent(void *arg) {
	os_printf("���ͳɹ���");
}
void ICACHE_FLASH_ATTR server_discon(void *arg) {
	os_printf("�����Ѿ��Ͽ���");
}

void ICACHE_FLASH_ATTR server_listen(void *arg)  //ע�� TCP ���ӳɹ�������Ļص�����
{
	struct espconn *pespconn = arg;
	espconn_regist_recvcb(pespconn, server_recv);  //����
	espconn_regist_sentcb(pespconn, server_sent);  //����
	espconn_regist_disconcb(pespconn, server_discon);  //�Ͽ�
}
void ICACHE_FLASH_ATTR server_recon(void *arg, sint8 err) //ע�� TCP ���ӷ����쳣�Ͽ�ʱ�Ļص������������ڻص������н�������
{
	os_printf("���Ӵ��󣬴������Ϊ��%d\r\n", err); //%d,�������ʮ��������
}

void Inter213_InitTCP(uint32_t Local_port) {
	user_tcp_espconn.proto.tcp = (esp_tcp *) os_zalloc(sizeof(esp_tcp)); //����ռ�
	user_tcp_espconn.type = ESPCONN_TCP; //��������ΪTCPЭ��
	user_tcp_espconn.proto.tcp->local_port = Local_port; //���ض˿�

	espconn_regist_connectcb(&user_tcp_espconn, server_listen); //ע�� TCP ���ӳɹ�������Ļص�����
	espconn_regist_reconcb(&user_tcp_espconn, server_recon); //ע�� TCP ���ӷ����쳣�Ͽ�ʱ�Ļص������������ڻص������н�������
	espconn_accept(&user_tcp_espconn); //���� TCP server����������
	espconn_regist_time(&user_tcp_espconn, 180, 0); //���ó�ʱ�Ͽ�ʱ�� ��λ���룬���ֵ��7200 ��

}

void WIFI_Init() {
	struct softap_config apConfig;
	wifi_set_opmode(0x02);    //����ΪAPģʽ�������浽 flash
	apConfig.ssid_len = 10;						//����ssid����
	os_strcpy(apConfig.ssid, "xuhongLove");	    //����ssid����
	os_strcpy(apConfig.password, "12345678");	//��������
	apConfig.authmode = 3;                      //���ü���ģʽ
	apConfig.beacon_interval = 100;            //�ű���ʱ��100 ~ 60000 ms
	apConfig.channel = 1;                      //ͨ����1 ~ 13
	apConfig.max_connection = 4;               //���������
	apConfig.ssid_hidden = 0;                  //����SSID

	wifi_softap_set_config(&apConfig);		//���� WiFi soft-AP �ӿ����ã������浽 flash
}

void tcp_service_init()		//��ʼ��
{
	WIFI_Init();
	Inter213_InitTCP(8266);		//���ض˿�
}

