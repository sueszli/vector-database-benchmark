#include<stdio.h>

void printf_char(char msg[]){ //��ͬ��  void printf_char(char *msg){
 printf("Message read: %s\r\n",msg);
 puts(msg);//puts���Զ����� 
 printf("The size of msg is: %d\r\n",sizeof(msg));//���ַ�ָ������Ĵ�С 
 printf("Message read: %p\r\n",msg+9);//�׵�ַ ƫ�� 9����Ԫ 
 printf("Message read: %s\r\n",msg+9);//�׵�ַ ƫ�� 9����Ԫ ��ַ�ڴ洢������  �ַ� 
} 

int main(){
char que[]="qwertryu qwert uy";
printf("the dress of que:%p\r\n",que);
printf_char(que);
printf("SIZE of char* :%d\r\n",sizeof(char*));
char *p_que = que;//�����׵�ַָ�� 
printf_char(p_que);

char xin[29],ming[29];
printf("Enter Your first name and last name"); 
scanf("%28s %28s",xin ,ming);//�����׵�ַ     ��󳤶� + \0 Ϊ�ܳ��� 

int age;
printf("�����������:\r\n");
scanf("%d",&age);// age ������ַ 

char food[10];
printf("�������������:\r\n");
fgets(food,sizeof(food),stdin);//�����������󳤶� sizeof(food) 


return 0;
} 
