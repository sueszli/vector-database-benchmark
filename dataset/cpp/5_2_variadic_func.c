#include <stdarg.h>

// va_list va_arg va_end  Ϊ����ĺ�  ����ǰ Ԥ����ʱ���滻Ϊ����Ĵ��� 
void print_ints(int args, ...){
	va_list val;//�����б� 
	va_start(val, args);//��args ��ʼ 
	int i;
	for(i=0;i<args;i++){
	   printf("������%i\r\n",va_arg(val,int));//�б� ��������
	}
	va_end(val); //���� �����б� 
}

//������Ŀ 
typedef enum drink{
	MUDSLIDE, FUZZY_NAVEL, MONKEY_GLAND, ZOMBIE
}drink; //ö������ 
double price(drink d){
	switch(d){
		case MUDSLIDE:
		  return 6.79; 
		case FUZZY_NAVEL:
		  return 5.31;
		case MONKEY_GLAND:
		  return 4.82;
	    case ZOMBIE:
		  return 5.89;		  
	}
	return 0;	
} 

double total(int args, ...){
	va_list val;//�����б� 
	va_start(val, args);//��args ��ʼ 
	int i;
	double sum=0;
	for(i=0;i<args;i++){		
	   sum += price(va_arg(val, drink)); //��Ҫָ����������  
	}
	va_end(val); //���� �����б� 
	return  sum;
}

int main(){
    print_ints(4, 111, 222, 1212, 1111); //���� ����... 
	printf("�����ѽ�%.2f\r\n",total(1, MUDSLIDE)); 
	printf("�����ѽ�%.2f\r\n",total(2, MUDSLIDE, FUZZY_NAVEL));
	printf("�����ѽ�%.2f\r\n",total(3, MUDSLIDE, FUZZY_NAVEL, MONKEY_GLAND));
	printf("�����ѽ�%.2f\r\n",total(4, MUDSLIDE, FUZZY_NAVEL, MONKEY_GLAND, ZOMBIE));	
	return 0;
}

