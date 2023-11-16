#include "driver/uart.h"  //����0��Ҫ��ͷ�ļ�
#include "osapi.h"  //����1��Ҫ��ͷ�ļ�
#include "user_interface.h" //WIFI������Ҫ��ͷ�ļ�
#include "espconn.h"//TCP������Ҫ��ͷ�ļ�
#include "mem.h" //ϵͳ������Ҫ��ͷ�ļ�
#include "gpio.h"

struct espconn user_udp_espconn;

static void Inter213_Receive(void *arg, char *pdata, unsigned short len) {  //����
	os_printf("�յ����ݣ�%s\r\n", pdata); // %s,�������һ���ַ���
	espconn_sent((struct espconn *) arg, "�Ѿ��յ�", strlen("�Ѿ��յ�"));

}
static void Inter213_Send_Cb(void *arg) {  //����
	os_printf("\r\n�ѷ���\r\n");

}

void Inter213_InitUDP(int32_t Remote_port, uint32_t Local_port) {
	user_udp_espconn.proto.udp = (esp_udp *) os_zalloc(sizeof(esp_udp));  //����ռ�
	user_udp_espconn.type = ESPCONN_UDP;  //��������ΪUDPЭ��
	user_udp_espconn.proto.udp->local_port = Local_port;  //���ض˿�
	user_udp_espconn.proto.udp->remote_port = Remote_port;  //Ŀ��˿�

	espconn_regist_recvcb(&user_udp_espconn, Inter213_Receive);  //����
	espconn_regist_sentcb(&user_udp_espconn, Inter213_Send_Cb);  //����
	espconn_create(&user_udp_espconn);  //����UDP����

}

void WIFI_Init() {
	struct softap_config apConfig;

	wifi_set_opmode(0x02);  //����ΪAPģʽ�������浽 flash

	apConfig.ssid_len = 10;				        //����ssid����
	os_strcpy(apConfig.ssid, "meizu");	//����ssid����
	os_strcpy(apConfig.password, "12345678");	//��������
	apConfig.authmode = 3;                      //���ü���ģʽ
	apConfig.beacon_interval = 100;            //�ű���ʱ��100 ~ 60000 ms
	apConfig.channel = 1;                      //ͨ����1 ~ 13
	apConfig.max_connection = 4;               //���������
	apConfig.ssid_hidden = 0;                  //����SSID

	wifi_softap_set_config(&apConfig);		//���� WiFi soft-AP �ӿ����ã������浽 flash
}
void udp_services_init()		//��ʼ��
{
	os_printf("\r\n udp_services_init ... \r\n");
	WIFI_Init();
	Inter213_InitUDP(8266, 8266);		//Ŀ��˿ڣ����ض˿�

}

