/*
 * xSntp.c
 *
 *  Created on: 2019��4��2��
 *      Author: XuHongYss
 */

#include "xSntp.h"
#include "c_types.h"
#include "osapi.h"
//ʹ�ýṹ�������Ϊ�����ķ���ֵ
sntp_data ICACHE_FLASH_ATTR sntp_get_time_change(char *s) {
	s[24]='\0';//��Ҫ��ݺ��������
	sntp_data data;
	unsigned char status[7] = { 3, 3, 2, 2, 2, 2, 4 };
	unsigned char c[5], i = 0, j = 0;
	while (*s != '\0') {
		if (*s == ' ' || *s == ':') {
			s++;
			i++;
		} else {
			os_memset(c, '\0', 5); //����ֵ
			for (j = 0; j < status[i]; j++) {
				c[j] = *s;
				s++;
			}
			switch (i) {
			case 0:
				if (strcmp(c, "Mon") == 0)
					data.week = 1; //����һ
				else if (strcmp(c, "Tue") == 0)
					data.week = 2; //���ڶ�
				else if (strcmp(c, "Wed") == 0)
					data.week = 3; //������
				else if (strcmp(c, "Thu") == 0)
					data.week = 4; //������
				else if (strcmp(c, "Fri") == 0)
					data.week = 5; //������
				else if (strcmp(c, "Sat") == 0)
					data.week = 6; //������
				else if (strcmp(c, "Sun") == 0)
					data.week = 7; //������
				break;
			case 1:
				if (strcmp(c, "Jan") == 0)
					data.month = 1; //һ��
				else if (strcmp(c, "Feb") == 0)
					data.month = 2; //����
				else if (strcmp(c, "Mar") == 0)
					data.month = 3; //����
				else if (strcmp(c, "Apr") == 0)
					data.month = 4; //����
				else if (strcmp(c, "May") == 0)
					data.month = 5; //����
				else if (strcmp(c, "Jun") == 0)
					data.month = 6; //����
				else if (strcmp(c, "Jul") == 0)
					data.month = 7; //����
				else if (strcmp(c, "Aug") == 0)
					data.month = 8; //����
				else if (strcmp(c, "Sep") == 0)
					data.month = 9; //����
				else if (strcmp(c, "Oct") == 0)
					data.month = 10; //ʮ��
				else if (strcmp(c, "Nov") == 0)
					data.month = 11; //ʮһ��
				else if (strcmp(c, "Dec") == 0)
					data.month = 12; //ʮ����

				break;
			case 2: //��
				data.day = (c[0] & 0x0f) << 4;
				data.day = data.day | (c[1] & 0x0f);
				break;
			case 3: //ʱ
				data.hour = (c[0] & 0x0f) << 4;
				data.hour = data.hour | (c[1] & 0x0f);
				break;
			case 4: //��
				data.minute = (c[0] & 0x0f) << 4;
				data.minute = data.minute | (c[1] & 0x0f);
				break;
			case 5: //��
				data.second = (c[0] & 0x0f) << 4;
				data.second = data.second | (c[1] & 0x0f);
				break;
			case 6: //��
				data.year = (c[2] & 0x0f) << 4;
				data.year = data.year | (c[3] & 0x0f);
				break;
			default:
				break;
			}
		}

	}
	return data;
}
