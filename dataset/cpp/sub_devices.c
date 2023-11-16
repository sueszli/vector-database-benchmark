#include "osapi.h"
#include "espnow.h"
#include "ets_sys.h"
#include "osapi.h"
#include "ip_addr.h"
#include "espconn.h"
#include "mem.h"
#include "spi_flash.h"
#include "user_interface.h"
#include "c_types.h"
#include "smartconfig.h"

os_timer_t gateway_esp_now_timer;
//����оƬMAC��ַ
u8 macStationAdress[1024];

//���ص�ַ ���˵�ַ���滻����Ҫ���Ե��豸��mac��ַ
u8 gateWayDeviceMac[6] = { 0xA0, 0x20, 0xA6, 0x08, 0x2E, 0xDC };

static void ICACHE_FLASH_ATTR gateway_esp_now_recv_cb(u8 *macaddr, u8 *data,
		u8 len) {

	int i;
	static u16 ack_count = 0;
	u8 ack_buf[16];
	u8 recv_buf[17];
	os_printf("recieve from gataway[");
	for (i = 0; i < 6; i++) {
		os_printf("%02X ", macaddr[i]);
	}
	os_printf(" len: %d]:", len);

	os_bzero(recv_buf, 17);
	os_memcpy(recv_buf, data, len < 17 ? len : 16);
	os_printf(recv_buf);
	os_printf("\r\n");

	if (os_strncmp(data, "ACK", 3) == 0) {
		return;
	}

	//���������豸��Ҫд����mac��ַ��
	user_esp_now_send(gateWayDeviceMac, macStationAdress,
	os_strlen(macStationAdress));

}

void ICACHE_FLASH_ATTR gateway_esp_now_send_cb(u8 *mac_addr, u8 status) {

	if (1 == status) {
		os_printf(
				"send message to gateWayDevice fail ! the fail send macAdress:");
		int i;
		for (i = 0; i < 6; i++) {
			os_printf("%02X-", mac_addr[i]);
		}
		os_printf("\r\n");
	} else if (0 == status) {
		os_printf(
				"send message to gateWayDevice successful ! the send macAdress:");
		int i;
		for (i = 0; i < 6; i++) {
			os_printf("%02X-", mac_addr[i]);
		}
		os_printf("\r\n");
	}
}

void subDevice_Device_init() {

	wifi_set_opmode(STATION_MODE);	//����ΪSTATIONģʽ
	struct station_config stationConf;
	os_strcpy(stationConf.ssid, "iPhone");	  //�ĳ����Լ���   ·�������û���
	os_strcpy(stationConf.password, "xh870189248"); //�ĳ����Լ���   ·����������
	wifi_station_set_config(&stationConf);	//����WiFi station�ӿ����ã������浽 flash
	wifi_station_connect();	//����·����
	os_printf("As a subDevice ...\r\n");

	u8 macAddr[6] = { 0 };
	wifi_get_macaddr(STATION_IF, macAddr);
	os_sprintf(macStationAdress,
			"Hello! I am message from subDevice: %02x%02x%02x%02x%02x%02x",
			macAddr[0], macAddr[1], macAddr[2], macAddr[3], macAddr[4],
			macAddr[5]);

	if (esp_now_init() == 0) {

		os_printf("esp_now subDevice init ok! \n");
		// ע�� ESP-NOW �հ��Ļص�����
		esp_now_register_recv_cb(gateway_esp_now_recv_cb);
		// ע�ᷢ���ص�����
		esp_now_register_send_cb(gateway_esp_now_send_cb);

		esp_now_set_self_role(ESP_NOW_ROLE_COMBO);
		//������أ��ŵ�Ϊ1�������ܣ�
		esp_now_add_peer(gateWayDeviceMac, ESP_NOW_ROLE_COMBO, 1, NULL, 16);
	}

}
