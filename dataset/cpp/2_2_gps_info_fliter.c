#include<stdio.h>
// (./fliter | ./gps) < gps.csv > gps.json �������ض���
int main(){
	float latitude;//γ��
	float longitude;//���� 
	char info[80]; //�ٶ���Ϣ�ַ�����	
	while (scanf("%f,%f,%79[^\n]",&latitude,&longitude,info)==3)
		if((latitude>34)&&(latitude<76))
			if((longitude>-76)&&(longitude<-64))
				printf("%f,%f,%s\n",latitude,longitude,info);// �Ѳ��� ����Ҫ������� �������׼����� 
	return 0;
} 
