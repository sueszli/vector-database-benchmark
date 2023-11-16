#include<stdio.h>

//ö��  �������� (�ַ�����ʶ��)  ö����δ�� ����
typedef enum {
	COUNT, POUNDS, PINTS//ʹ�ö��� �ָ�  ������ ������ �ݻ���
} unit_of_measure;//ö�ٱ���  Ϊ���е�һ��ֵ


//���� ��Ч���ô洢�ռ�  ����һ��ռ� 
// ���治ͬ������ 
typedef union {
	short count;// ��Ӧ ö�����͵� COUNT
	float weight;//ʹ�÷ֺ� �ָ� ��Ӧ ö�����͵� COUNT POUNDS
	float volume; // ��Ӧ ö�����͵� PINTS
} quantity;//���ϱ���


// �ṹ��
typedef struct {
	const char *name;//�ַ�������ֵ     ˮ������
	const char *country;//ʹ�÷ֺ� �ָ� ����
	quantity amount;//���� 
	unit_of_measure units;//ö��
} fruit_order;// �ṹ���������  ����������ָ�� *fruit_order;


// �ṹ��  λ�ֶ�bitfield �洢���� ��Ч���ÿռ�
typedef struct {//����ʹ��  unsigned int ��ʶ 
	unsigned int low_pass_vcf:1;//���ֶ���һλ����  0~1 
	unsigned int two_pass_vcf:2;//���ֶ���2λ����	0~3
	unsigned int month_nomber:4;//4Ϊ���� 0~15 �·�1~12 	
} bitfield_temp; 


void display(fruit_order order){//  fruit_order *order    &apples   order->units
	printf("ˮ������\r\n");
	if(order.units==PINTS)//������� ��ȡ�ṹ�е��ֶ�  -> ���� ���½ṹ�е��ֶ� 
		printf("%2.2f pints of %s\n", order.amount.volume,order.name);
	else if(order.units==POUNDS)
		printf("%2.2f lbs of %s\n", order.amount.weight,order.name);	
    else
		printf("%i %s\n", order.amount.count,order.name);
}

int main(){
fruit_order apples = {"apple","England",.amount.count=144,COUNT};//ƻ��	
fruit_order strawberries = {"strawberries","China",.amount.weight=17.6,POUNDS};//��ݮ
fruit_order orangej = {"orange juice","U.S.A",.amount.volume=11.5,PINTS};//����֭
display(apples);
display(strawberries);
display(orangej);

return 0;
}
