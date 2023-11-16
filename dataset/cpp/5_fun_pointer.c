#include <stdio.h>
#include <string.h>
#include <stdlib.h>
// ��Ҫʹ�ú���ָ��ĺ��� ���β����� ����Ϊ const void* ����
int compare_scores(const void* s_a,  const void* s_b){
	int a=*(int*)s_a;//void* ת�� int* ��ȡ��ַ�е�ֵ
	int b=*(int*)s_b;
	return a-b;//����  b-aΪ��������
}
typedef struct{
	int width;
	int height;
} rectangle;//�������� �ṹ�� ����
int compare_areas(const void* a, const void* b){
	rectangle* ra = (rectangle*)a;//void* ת�� �ṹ������ָ��
	rectangle* rb = (rectangle*)b;//void* ת�� �ṹ������ָ��
return 	(ra->width*ra->height - rb->width*rb->height); 
}
int compare_name(const void* a, const void* b){
	char** sa=(char**)a;
	//�ַ���Ϊ�ַ�ָ�룬�����Ϊ�ַ���ָ�� ����Ϊ char** ָ���ָ��
	char** sb=(char**)b;
	return strcmp(*sa, *sb);
	
}

int main(){
	
	int scores[] = {534, 122, 12, 45, 345, 970, 10};
	int i;
	qsort(scores, 7, sizeof(int), compare_scores);
	puts("�����ķ���:\r\n");
	for(i=0;i<7;i++){
		printf("���� = %i\n", scores[i]);
	}
	
	char *name[]={"Karen", "Mark", "Brett", "Molly", "Tony"};//�ַ���ָ������
	qsort(name, 5, sizeof(char*), compare_name);
	puts("����������:\r\n");
		for(i=0;i<5;i++){
		printf("%s\n", name[i]);
	}
	return 0;
}

