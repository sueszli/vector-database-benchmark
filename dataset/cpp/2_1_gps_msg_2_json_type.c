#include<stdio.h>
// ./gps < gps.csv > gps.json �������ض���
// 2> reeor.txt  �ض��������� 
// gps.csv
// 42.3432222222221,-71.21344444444,Speed = 21
// gps.json
// data=[ 
//{latitude: 42.3432222222221, longitude: -71.21344444444, info: 'Speed = 21'},
// ]
int main(){
	float latitude;//γ��
	float longitude;//���� 
	char info[80]; //�ٶ���Ϣ�ַ�����
	int started=0;// ,���б�ʶ��ʼ
	
	puts("data,=[");
	while (scanf("%f,%f,%79[^\n]",&latitude,&longitude,info)==3){
	// %79[^\n]�൱��˵ ����һ�����µ��ַ������� scanf() ���سɹ���ȡ���������� 
	if(started)	
	    printf(",\n");
	else
	    started=1;
	
	//  printf('��ϲ��") �������׼���  ��ͬ�� fprintf(stdout, "��ϲ��");    ���� scanf();   �� fscanf(stdin,...); 
	// ǰһ�� f ���� �� flow  ��һ�� f  ���� format ��ʽ�� 
	if((latitude>90)||(latitude<-90)){
	 //	printf("error"); //���������׼�����  �����ȷ��Ϣ������һ�� 
	    fprintf(stderr,"error"); 
		return 2;
	}
	if((longitude>180)||(longitude<-180)){
	 //	printf("error");
	    fprintf(stderr,"error"); 
		return 2;
	}    
	printf("{latitude: %f, longitude: %f, info: '%s'}",latitude,longitude,info);
	} 
	puts("\n}");
	
	return 0;
} 
