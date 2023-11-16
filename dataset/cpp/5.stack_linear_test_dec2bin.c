#include<stdio.h>  
#include<malloc.h>  

#define ERROR 0  
#define OK 1  
#define STACK_INT_SIZE 10  /*�洢�ռ��ʼ������*/  
#define STACKINCREMENT 5  /*�洢�ռ��������*/  
typedef  int ElemType; /*����Ԫ�ص����� int �ȼ� �������� ���� */  

typedef struct{  
    ElemType *base;  
    ElemType *top;  
    int stacksize;     /*��ǰ�ѷ���Ĵ洢�ռ�*/  
}SqStack;  
  
   
int push(SqStack *S,ElemType e); /*��ջ*/  
int Pop(SqStack *S,ElemType *e);  /*��ջ*/  
void PrintStack(SqStack *S);   /*��ջ�����ջ��Ԫ��*/  
  
  
int Push(SqStack *S,ElemType e){  
    if(S->top-S->base==STACK_INT_SIZE)  
        return 0;  
    *(++S->top)=e;  
    return 1;  
}/*Push*/  
  
int Pop(SqStack *S,ElemType *e){  
    if(S->top==S->base)  
        return 0;  
    *e=*S->top;  
    S->top--;  
    return 1;  
}/*Pop*/  
  
  
void Conversion(SqStack *S,ElemType e)  
{  
    while(e/2)  
    {  
        Push(S,e%2);  
        e/=2;  
    }  
    Push(S,e);  
  
}  
  
void PrintStack(SqStack *S){  
    ElemType e;  
    while(Pop(S,&e))  
        printf("%d",e);  
    printf("\n");  
}/*Pop_and_Print*/  
  
int main(){  
    int num;  
    SqStack *S;  
    S=(SqStack *)malloc(sizeof(SqStack));  
    S->base=(ElemType *)malloc(STACK_INT_SIZE *sizeof(ElemType));  
    if(!S->base) return ERROR;  
    S->top=S->base;  
    S->stacksize=STACK_INT_SIZE;  
  
    printf("������Ҫת�������֣�\n");  
    scanf("%d",&num);  
    Conversion(S,num);  
    PrintStack(S);  
    return 0;  
} 
