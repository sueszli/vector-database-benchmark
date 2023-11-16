#include <stdio.h>
#include <stdlib.h>

enum response_type{DUMP, SECOND_CHANCE, MARRIAGE};//ö�ٷ��ű��� �洢ʱ�� 0 1 2 3 ... �洢 
typedef struct{
char *name;
enum response_type type; 
} response;

void dump(response r){//ֻ�����ýṹ����� �����Ҫ�޸� ��Ҫ�޸�Ϊ �ṹ��ָ��  response *r
	printf("Deer %s, \n",r.name);
	puts("�ܲ��ң���û�л�����\r\n");
}
void second_chance(response r){ 
	printf("Deer %s, \n",r.name);
	puts("����Ϣ��������һ�λ���\r\n");
}
void marriage(response r){ 
	printf("Deer %s, \n",r.name);
	puts("ף�أ���ͨ����!\r\n");
}

//����ָ������
//�������� ����ָ������ ��������       
void (*replies[]) (response) = {dump, second_chance, marriage};

int main(){
	response r[]={
		{"Mike", DUMP}, {"Luna",SECOND_CHANCE},
		{"Hatter",MARRIAGE}, {"Wind",SECOND_CHANCE}
	}; 
	int i;
	for(i=0; i<4; i++){
		(replies[r[i].type])(r[i]);//r[i].type = 0/1/2
	}
	
	return 0;
}

