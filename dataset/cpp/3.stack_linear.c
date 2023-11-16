/*
���Խṹ-ջ�������ṹ 
ջ���Ƚ���������������Ƚ��ȳ���

ջ��ֻ���ڱ�β���в����ɾ�����������Ա�
ͨ�����ǳƱ�β��Ϊջ��
����ͷ��Ϊջ�ף�
����һ���Ƚ���������Ա�
��ֻ���ڱ�βջ���˲���Ԫ�أ���Ϊ��ջ��
Ҳֻ���ڱ�β��ջ��ɾ��Ԫ�أ���Ϊ��ջ��

ջ��ȻҲ�����Ա���ô��Ҳ��
˳��洢�ṹ����ʽ�洢�ṹ���ֱ�ʾ������
�����ֱ�ʾ����ʵ�����ƣ�
*/

#include <stdio.h>
#include <stdlib.h>

#define TRUE 1
#define FALSE 0
#define OK 1
#define ERROR 0
#define OVERFLOW -2
#define INIT_SIZE 20//��ʼջ�����ڴ��С 
#define INCREMENT_SIZE 5//��������  

typedef int SElemType;
typedef int Status;

/*
 * �洢�ṹ
 */
typedef struct
{
    SElemType *base;    //ջβָ��   ��ͷ 
    SElemType *top;     //ջ��ָ��  ��β 
    int size;           //ջ�Ĵ�С
}SqStack;

/*
 * ��ʼ��ջ
 */
Status InitStack(SqStack *S)
{
    S->base = (SElemType*) malloc(INIT_SIZE * sizeof(SElemType));//��ͷ 
    if (!S->base)//������� 
    {
        exit(OVERFLOW);
    }
    S->top = S->base;//ջ��Ҳ��ջβ  ��ͷҲ�Ǳ�β 
    S->size = INIT_SIZE;
    return OK;
}

/*
 * ����ջ
 */
Status DestroyStack(SqStack *S)
{
    free(S->base);//ջβ ��ͷ 
    S->base = NULL;
    S->top = NULL;
    S->size = 0;
    return OK;
}

/*
 * ���ջ
 */
Status ClearStack(SqStack *S)
{
    S->top = S->base;//ջ��ָ��ջβ 
    return OK;
}

/*
 * �ж�ջ�Ƿ�Ϊ��
 */
Status IsEmpty(SqStack S)
{
    if (S.top == S.base)//ջ�� = ջβ 
    {
        return TRUE;
    }
    else
        return FALSE;
}

/*
 * ��ȡջ�ĳ���
 */
int GetLength(SqStack S)
{
    return S.top - S.base;//ջ��-ջβ 
}


/*
 * ��ȡջ��Ԫ��
 */
Status GetTop(SqStack S, SElemType *e)
{
    if (S.top > S.base)
    {
        *e = *(--S.top);
        return OK;
    }
    else
    {
        return ERROR;
    }
}

/*
 * ѹջ
 */
Status Push(SqStack *S, SElemType e)
{
    if ((S->top - S->base) / sizeof(SElemType) >= S->size)//�ڴ治���� 
    {
        S->base = (SElemType*) realloc(S->base, (S->size + INCREMENT_SIZE) * sizeof(SElemType));
        if (!S->base)//������� 
        {
            exit(OVERFLOW);
        }
        S->top = S->base + S->size;//����ջ����ַ   �������� 
        S->size += INCREMENT_SIZE;//���´�С 
    }
    *S->top = e;//*��ַ  ȡֵ  ����ֵ 
    S->top++;//��ַ+1 
    return OK;
}

/*
 * ��ջ
 */
Status Pop(SqStack *S, SElemType *e)
{
    if (S->top == S->base)//�� 
    {
        return ERROR;
    }
    S->top--;//��ַ-1 
    *e = *S->top;//ȡջ����ַ���ֵ 
    return OK;
}

/*
 * ����Ԫ��
 */
void visit(SElemType e)
{
    printf("%d ", e);
}

/*
 * ����ջ
 */
Status TraverseStack(SqStack S, void (*visit)(SElemType))
{
    while (S.top > S.base)
    {
        visit(*S.base);
        S.base++;//ջβ��ַ++ 
    }
    return OK;
}

int main()
{
    SqStack S;
    if (InitStack(&S))
    {
        SElemType e;
        int i;

        printf("init_success\n");

        if (IsEmpty(S))
        {
            printf("Stack is empty\n");
        }

        for (i = 0; i < 10; i++)
        {
            Push(&S, i);
        }

        GetTop(S, &e);
        printf("The first element is %d\n", e);

        printf("length is %d\n", GetLength(S));

        Pop(&S, &e);
        printf("Pop element is %d\n", e);

        TraverseStack(S, *visit);

        if (DestroyStack(&S))
        {
            printf("\ndestroy_success\n");
        }
    }
}
