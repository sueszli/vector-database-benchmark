//���Ա����ʽ��ʾ
//������
// ��ͷ��㿪ʼ��ָ��������ָ����һ�����ݵĴ洢��ַ��β�ڵ��ָ����ΪNULL
// ���� ѭ������ �� ˫������
/*
���Ա��˳��洢�ṹ���߼�λ�ú�����λ�ö����ڣ�
����ʽ�洢�ṹ���߼�λ�����ڣ�������λ�ò�һ�����ڣ�
���˳��洢�ṹ�������������ȡ��
���ڲ����ɾ������ʱ����Ҫ�ƶ�Ԫ�أ�
�����������Ӻ�ɾ��Ԫ�ص�Ч�ʡ�

˳��洢�ṹ��ȡԪ�ص�Ч�ʱȽϸߣ���ʽ�洢�ṹ��Ӻ�ɾ��Ԫ�ص�Ч�ʱȽϸߡ�
*/ 
#include <stdio.h>  //��׼�� ������� 
#include <stdlib.h> //��׼��  

#define TRUE 1
#define FALSE 0
#define OK 1
#define ERROR 0
#define OVERFLOW -2

//  �ȼ� �������� ���� 
typedef int Status;  //״̬ 
typedef int ElemType;//�� Ԫ������ 

/*
 * �洢�ṹ
 */
typedef struct LNode
{
    ElemType data;//Ԫ��  �����Ԫ�ؿ��Բ�ֻһ��  ����Ҳ����Ϊ �ַ��������
	              // ���� char cName[20];/*ѧ������*/  int iNumber;/*ѧ��ѧ��*/
    struct LNode *next;//���ָ�� 
}LNode, *LinkList;//���� �ṹ�����  �ṹ��ָ�� 

/*
 * ��ʼ�����Ա�
 */
void InitList(LinkList *L)
{
    *L = (LinkList) malloc(sizeof(LNode));// һ��Ԫ�� + һ��ָ�����  ���ڴ��С 
    if (!L)
    {
        exit(OVERFLOW);
    }
    (*L)->next = NULL;//��ʼ��ʱ  ���ָ��ΪNULL 
}

/*
 * �������Ա�
 */
void DestroyList(LinkList *L)
{
    LinkList temp;
    while (*L)//δ����β ��β��ΪNULL 
    {
        temp = (*L)->next;//�����̵Ľڵ� 
        free(*L);//ɾ����ǰ�ڵ��ڴ� 
        *L = temp;//��һ���ڵ� 
    }
}

/*
 * ������Ա�
 */
void ClearList(LinkList L)
{
    LinkList p = L->next;//��һ���ڵ� 
    L->next = NULL;//���ָ��ΪNULL 
    DestroyList(&p);//����ڴ� 
}

/*
 * �ж��Ƿ�Ϊ��
 */
Status isEmpty(LinkList L)
{
    if (L->next)
    {
        return FALSE;//��������� ��Ϊ�� 
    }
    else
    {
        return TRUE;//���������  Ϊ�� 
    }
}

/*
 * ��ȡ����
 */
int GetLength(LinkList L)
{
    int i = 0;
    LinkList p = L->next;//��̽ڵ� 
    while (p)
    {
        i++;//��̽ڵ� ���� 
        p = p->next;//���α�����̽ڵ� 
    }
    return i;
}

/*
 * ����λ�û�ȡԪ��
 */
Status GetElem(LinkList L, int i, ElemType *e)
{
    int j = 1;
    LinkList p = L->next;//��̽ڵ� 
    while (p && j < i)//p δ��β�ڵ� ��j<i 
    {
        j++;//λ�ü��� 
        p = p->next;//��һ���󼶽ڵ� 
    }
    if (!p || j > i)//Խ��  p����β�� 
    {
        return ERROR;
    }
    *e = p->data;
    return OK;
}

/*
 * �Ƚ�����Ԫ���Ƿ����
 */
Status compare(ElemType e1, ElemType e2)
{
    if (e1 == e2)
    {
        return 0;
    }
    else if (e1 < e2)
    {
        return -1;//ǰ���С�ں���� 
    }
    else
    {
        return 1;//ǰ��Ĵ��ں���� 
    }
}

/*
 * ����ָ��Ԫ�ص�λ��
 */
int FindElem(LinkList L, ElemType e, Status (*compare)(ElemType, ElemType))
{
    int i = 0;
    LinkList p = L->next;//��̽ڵ� 
    while (p)
    {
        i++;//λ�ü��� +1 
        if (!compare(p->data, e))//�Ƚϵ�ǰ�ڵ�Ԫ��ֵ�� Ŀ��Ԫ��ֵ 
        {
            return i;//��ȷ��� λ�ü���ֵ 
        }
        p = p->next;//ָ����  
    }
    return 0;
}

/*
 * ��ȡǰ��Ԫ��
 */
Status PreElem(LinkList L, ElemType cur_e, ElemType *pre_e)
{
    LinkList q, p = L->next;//��̽ڵ� 
    while (p->next)//��� ��Ϊ�� 
    {
        q = p->next;//p�ڵ�ĺ��Ϊq 
        if (q->data == cur_e)//q�ڵ��Ԫ����Ŀ�� Ԫ�����
        {
            *pre_e = p->data;//q��ǰ�� p��Ԫ��Ϊ Ŀ��Ԫ�ص�ǰ�� 
            return OK;
        }
        p = q;//pָ����һ����� 
    }
    return ERROR;
}

/*
 * ��ȡ���Ԫ��
 */
Status NextElem(LinkList L, ElemType cur_e, ElemType *next_e)
{
    LinkList p = L->next;//��� 
    while (p->next)//�ڵ��̲�Ϊ�� 
    {
        if (p->data == cur_e)//�ڵ�Ԫ����Ŀ��Ԫ����� 
        {
            *next_e = p->next->data;//��ǰ�ڵ�ĺ�̽ڵ��Ԫ��Ϊ Ŀ��Ԫ�صĺ��Ԫ�� 
            return OK;
        }
        p = p->next;//ָ����һ����� 
    }
    return ERROR;
}

/*
 * ����Ԫ��
 */
Status InsertElem(LinkList L, int i, ElemType e)
{
    int j = 0;
    LinkList s, p = L;// 
    while (p && j < i - 1)
    {
        j++;//λ�ü��� +1 
        p = p->next;//ָ���� ���� 
    }
    if (!p || j > i - 1)//λ��Խ�� 
    {
        return ERROR;
    }
    s = (LinkList) malloc(sizeof(LNode));//�¿���һ���ڴ�ռ� 
    s->data = e;//Ԫ��ֵ 
    s->next = p->next;//���ָ��p�ĺ�� 
    p->next = s;//p�ĺ��ָ���µ�Ԫ�ؽڵ� 
    return OK;
}

/*
 * ɾ��Ԫ�ز�����ֵ
 */
Status DeleteElem(LinkList L, int i, ElemType *e)
{
    int j = 0;
    LinkList q, p = L;
    while (p->next && j < i - 1)
    {
        j++;
        p = p->next;//�ҵ� Ŀ��λ�ýڵ� 
    }
    if (!p->next || j > i - 1)
    {
        return ERROR;
    }
    q = p->next;//����ڵ�ĺ�� ΪĿ��λ�ýڵ� 
    p->next = q->next;//�ڵ�ĺ�� ָ���̵ĺ�� 
    *e = q->data;//�ڵ��� ��ֵ Ŀ��λ�ýڵ��Ԫ��ֵ 
    free(q);//����ռ� 
    return OK;
}

/*
 * ����Ԫ��
 */
void visit(ElemType e)
{
    printf("%d ", e);
}

/*
 * �������Ա�
 */
void TraverseList(LinkList L, void (*visit)(ElemType))
{
    LinkList p = L->next;//��� 
    while (p)
    {
        visit(p->data);//���� 
        p = p->next;//ָ���� 
    }
}

int main()
{
    LinkList L;
    InitList(&L);
    ElemType e;
    int i;
    if (L)
    {
        printf("init success\n");
    }

    if (isEmpty(L))
    {
        printf("list is empty\n");    
    }

    for (i = 0; i < 10; i++)
    {
        InsertElem(L, i + 1, i);
    }

    if (GetElem(L, 1, &e)) {
        printf("The first element is %d\n", e);
    }

    printf("length is %d\n", GetLength(L));

    printf("The 5 at %d\n", FindElem(L, 5, *compare));

    PreElem(L, 6, &e);
    printf("The 6's previous element is %d\n", e);

    NextElem(L, 6, &e);
    printf("The 6's next element is %d\n", e);

    DeleteElem(L, 1, &e);
    printf("delete first element is %d\n", e);

    printf("list:");
    TraverseList(L,visit);

    DestroyList(&L);
    if (!L) {
        printf("\ndestroy success\n");    
    }
}
