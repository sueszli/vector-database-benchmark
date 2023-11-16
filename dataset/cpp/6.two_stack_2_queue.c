#include <stdio.h>  
#include <stdlib.h>  
#define STACK_INIT_SIZE 100  //�洢�ռ��ʼ������  
#define STACKINCREMENT 10    //�洢�ռ��������  
typedef struct{  
int *base;  
int *top;  
int stacksize;} stack;  
int initStack(stack *s)//����һ����ջ  
{  
    s->base = (int *)malloc(sizeof(int)*STACK_INIT_SIZE);  
    if(s->base == NULL)  
    {  
        printf("�洢�ռ����ʧ�ܣ�");  
        exit(-1);  
    }  
    s->top = s->base;  
    s->stacksize = STACK_INIT_SIZE;  
    return 1;  
}  
//ѹջ�ķ�����iΪ��Ҫѹ��ջ����Ԫ��  
int push(stack *s,int i)  
{  
    //�ж�ջ�����Ƿ��Ѿ����ˣ�������������·���洢�ռ�  
    if(s->top - s->base >= s->stacksize)  
    {  
        s->base = (int *)realloc(s->base,(s->stacksize + STACKINCREMENT)*sizeof(int));  
    }  
    if(s->base == NULL)  
    {  
        printf("�ڴ����ʧ�ܣ�");  
        exit(-1);  
    }  
    *s->top++ = i; //��Ԫ��ѹջ  
    return 1;  
}  
//��ջ�ķ���  
int pop(stack *s)  
{  
    if(s->base == s->top)  
        return -1;  
        //������Ĭ��ѹ��ջ�е�Ԫ�ض��������������Է���-1��ʾջ��  
    //���ջ���գ���ɾ��ջ��Ԫ�ز�������ֵ  
    return *--s->top;  
}  
//��������ջģ��һ������  
int main()  
{  
    stack st1,st2;  
    initStack(&st1);  
    initStack(&st2);  
    //���贫��Ķ���Ϊ 1��2��3��4��5��6  
    //������е�ʱ��˳��Ϊ 1��2��3��4��5��6  
    //����ջ1��ѹ���⼸����  
    int i;  
    printf("������е�����Ϊ:\n");  
    for(i = 1; i < 7 ;i++)  
    {  
        push(&st1,i);  
        printf("%d ",i);  
  
    }  
    printf("\n");  
    //��ջ��ʱ��ÿ���Ȱ�1ջ�г�ջ��Ԫ��֮�������Ԫ�ص���2ջ��  
    //Ȼ���ٵ���1ջջ��Ԫ��  
    //����ٰ�2ջ�е�Ԫ��ѹ��1ջ��  
    //�ظ����϶�����ֱ��1ջΪ��  
    printf("�����е�˳��Ϊ:\n");  
    while(st1.base!= st1.top)  
    {  
        while((st1.top - st1.base) > 1)  
            push(&st2,pop(&st1));  
        printf("%d ",pop(&st1));  
        while((st2.top - st2.base) > 0)  
            push(&st1,pop(&st2));  
    }  
    return 0;  
}
