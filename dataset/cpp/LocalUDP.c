#include "LocalUDP.h"
#include "ets_sys.h"
#include "os_type.h"
#include "osapi.h"
#include "mem.h"
#include "user_interface.h"
#include "espconn.h"
#include "gpio.h"



LOCAL struct espconn user_udp_espconn;
LOCAL os_timer_t checkTimer_wifistate;

void ICACHE_FLASH_ATTR user_udp_sent_cb(void *arg)   //����
{
    os_printf("\r\n���ͳɹ���\r\n");


}


LOCAL void ICACHE_FLASH_ATTR user_udp_recv_cb(void *arg, char *pdata, unsigned short length) {
	os_printf("�������ݣ�%s", pdata);
}


LOCAL void sendDataUDP() {

	os_printf("\r\n send data ...\r\n");
	//ÿ�η�������ȷ���˿ڲ�������
	user_udp_espconn.proto.udp = (esp_udp *) os_zalloc(sizeof(esp_udp));
	user_udp_espconn.type = ESPCONN_UDP;
	user_udp_espconn.proto.udp->local_port = 2000;
	user_udp_espconn.proto.udp->remote_port = 8686;
	const char udp_remote_ip[4] = { 255, 255, 255, 255 };
	os_memcpy(user_udp_espconn.proto.udp->remote_ip, udp_remote_ip, 4);
	espconn_sent(&user_udp_espconn, "this is message!", strlen("this is message!"));
}

//udpԶ������ģ��wifir�ӿ�
void ICACHE_FLASH_ATTR udpwificfgg_init(void) {

	user_udp_espconn.proto.udp = (esp_udp *) os_zalloc(sizeof(esp_udp)); //����ռ�
	user_udp_espconn.type = ESPCONN_UDP;              //��������ΪUDPЭ��
	user_udp_espconn.proto.udp->local_port = 2000;            //���ض˿�
	user_udp_espconn.proto.udp->remote_port = 8686;           //Ŀ��˿�
	const char udp_remote_ip[4] = { 255, 255, 255, 255 };       //Ŀ��IP��ַ���㲥��
	os_memcpy(user_udp_espconn.proto.udp->remote_ip, udp_remote_ip, 4);

	espconn_regist_recvcb(&user_udp_espconn, user_udp_recv_cb);	 		//����
	espconn_regist_sentcb(&user_udp_espconn, user_udp_sent_cb);	 		//����
	espconn_create(&user_udp_espconn); //���� UDP ����

	wifi_set_broadcast_if(1);

	os_timer_disarm(&checkTimer_wifistate);   //ȡ����ʱ����ʱ
	os_timer_setfn(&checkTimer_wifistate, (os_timer_func_t *) sendDataUDP,
	NULL);    //���ö�ʱ���ص�����
	os_timer_arm(&checkTimer_wifistate, 1000, 1);      //������ʱ������λ������

}

