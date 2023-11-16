/*
��ʽ ���� 
ջ���Ƚ���������������Ƚ��ȳ���
���иպú�ջ�෴������һ���Ƚ��ȳ������Ա�
ֻ����һ�˲���Ԫ�أ�����һ��ɾ��Ԫ�أ�
�������Ԫ�أ���ӣ���һ�˳�Ϊ��β��
����ɾ��Ԫ�أ����ӣ���һ�˳�Ϊ��ͷ��

����Ҳһ����˳�����ʽ�洢�ṹ���ֱ�ʾ������
ǰ���ջ����ʵ����˳��洢�ṹ��
*/ 
#include <stdio.h>
#include <stdlib.h>

#define TRUE 1
#define FALSE 0
#define OK 1
#define ERROR 0
#define OVERFLOW -2

typedef int QElemType;
typedef int Status;

/*
 * �洢�ṹ
 */
typedef struct QNode
{
    QElemType data;//Ԫ�� 
    struct QNode *next;//���ָ��   ��ʽ���� 
}QNode, *QueuePtr;

typedef struct
{
    QueuePtr front;   //��ͷָ��
    QueuePtr rear;    //��βָ��
}LinkQueue;

/*
 * ��ʼ������
 */
Status InitQueue(LinkQueue *Q)
{
    Q->front = Q->rear = (QueuePtr) malloc(sizeof(QNode));//��β ��ͷ 
    if (!Q->front)//������� 
    {
        exit(OVERFLOW);
    }
    Q->front->next = NULL;//Ϊ��ָ�� 
    return OK;
}

/*
 * ���ٶ���
 */
Status DestroyQueue(LinkQueue *Q)
{
    while (Q->front)//
    {
        Q->rear = Q->front->next;//��һ�� 
        free(Q->front);
        Q->front = Q->rear;
    }
    return OK;
}

/*
 * ��ն���
 */
Status ClearQueue(LinkQueue *Q)
{
    DestroyQueue(Q);
    InitQueue(Q);
    return OK;
}

/*
 * �ж϶����Ƿ�Ϊ��
 */
Status IsEmpty(LinkQueue Q)
{
    if (Q.front->next == NULL)
    {
        return TRUE;
    }
    else
    {
        return FALSE;
    }
}

/*
 * ��ȡ���еĳ���
 */
int GetLength(LinkQueue Q)
{
    int i = 0;
    QueuePtr p = Q.front;
    while (Q.rear != p)
    {
        i++;
        p = p->next;
    }
    return i;
}

/*
 * ��ȡ��ͷԪ��
 */
Status GetHead(LinkQueue Q, QElemType *e)
{
    QueuePtr p;
    if (Q.front == Q.rear)
    {
        return ERROR;
    }
    p = Q.front->next;
    *e = p->data;
    return OK;
}

/*
 * ���
 */
Status EnQueue(LinkQueue *Q, QElemType e)
{
    QueuePtr p = (QueuePtr) malloc(sizeof(QNode));
    if (!p)
    {
        exit(OVERFLOW);
    }
    p->data = e;
    p->next = NULL;
    Q->rear->next = p;
    Q->rear = p;
    return OK;
}

/*
 * ����
 */
Status DeQueue(LinkQueue *Q, QElemType *e)
{
    QueuePtr p;
    if (Q->front == Q->rear)
    {
        return ERROR;
    }
    p = Q->front->next;
    *e = p->data;
    Q->front->next = p->next;
    if (Q->rear == p)
    {
        Q->rear = Q->front;
    }
    free(p);
    return OK;
}

/*
 * ����Ԫ��
 */
void visit(QElemType e)
{
    printf("%d ", e);
}

/*
 * ��������
 */
Status TraverseQueue(LinkQueue Q, void (*visit)(QElemType))
{
    QueuePtr p = Q.front->next;
    while (p)
    {
        visit(p->data);
        p = p->next;
    }
    return OK;
}

int main()
{
    LinkQueue Q;
    if (InitQueue(&Q))
    {
        QElemType e;
        int i;

        printf("init_success\n");

        if (IsEmpty(Q))
        {
            printf("queue is empty\n");
        }

        for (i = 0; i < 10; i++)
        {
            EnQueue(&Q, i);
        }

        GetHead(Q, &e);
        printf("The first element is %d\n", e);

        printf("length is %d\n", GetLength(Q));

        DeQueue(&Q, &e);
        printf("delete element is %d\n", e);

        TraverseQueue(Q, *visit);

        if (DestroyQueue(&Q))
        {
            printf("\ndestroy_success\n");
        }
    }
}
