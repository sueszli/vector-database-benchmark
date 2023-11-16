// �Ա��˳��洢�ṹ���߼�λ�ú�����λ�ö�����
/*
���ҽ���һ����ʼ���û��ǰ������һ����̽�㣬
���ҽ���һ���ն˽��û�к�̵���һ��ǰ����㣬
�����Ľ�㶼���ҽ���һ��ǰ����һ����̽�㡣
һ��أ�һ�����Ա���Ա�ʾ��һ���������У�k1��k2������kn������k1�ǿ�ʼ��㣬kn���ն˽�㡣
˳��洢�ṹ��ȡԪ�ص�Ч�ʱȽϸߣ���ʽ�洢�ṹ��Ӻ�ɾ��Ԫ�ص�Ч�ʱȽϸߡ�
*/

#include <stdio.h>  //��׼�� ������� 
#include <stdlib.h> //��׼��  

// �궨�� 
#define TRUE 1
#define FALSE 0
#define OK 1
#define ERROR 0
#define INIT_SIZE 10        //��ʼ������ �ڵ����� 
#define INCREMENT_SIZE 5    //��������  �ڴ治���� �����ӽڵ��������� 

//  �ȼ� �������� ���� 
typedef int Status;  //״̬ 
typedef int Elemtype;//�� Ԫ������ 

/*
 * �洢�ṹ  �ṹ��  SqList *list;  list ->  length
 */
typedef struct
{
    Elemtype *elem;    //�洢�ռ��ַ  ָ�� ����ַ 
    int length;        //��ǰ����(ʵ�ʳ���)
    int size;          //��ǰ����ı���С(�����С) 
}SqList;

/*
 * ��ʼ��һ���յ����Ա�
 */
Status InitList(SqList *L)
{
    L->elem = (Elemtype *) malloc(INIT_SIZE * sizeof(Elemtype));//��ʼ��С*Ԫ�����ʹ洢����  ����һ���ڴ�ռ� malloc(size) 
    if (!L->elem)//����ַ 
    {
        return ERROR;
    }
    L->length = 0;// ʵ�ʳ��� 
    L->size = INIT_SIZE;//��ʼ�����С 
    return OK;//����״̬ 
}

/*
 * �������Ա�
 */
Status DestroyList(SqList *L)
{
    free(L->elem);//ɾ���ڴ�ռ� 
    L->length = 0;
    L->size = 0;
    return OK;
}

/*
 * ������Ա�
 */
Status ClearList(SqList *L)
{
    L->length = 0;// ʵ�ʳ��� 
    return OK;
}

/*
 * �ж����Ա��Ƿ�Ϊ��
 */
Status isEmpty(const SqList L)
{
    if (0 == L.length)
    {
        return TRUE;//1 Ϊ�� 
    }
    else
    {
        return FALSE;// 0 Ϊ�ǿ� 
    }
}

/*
 * ��ȡ����
 */
Status getLength(const SqList L)
{
    return L.length;// ʵ�ʳ��� 
}

/*
 * ����λ�û�ȡԪ��  �� L  λ�� i  ���ص�Ԫ��ֵ  GetElem(L, 1, &e)
 */
Status GetElem(const SqList L, int i, Elemtype *e)
{
    if (i < 1 || i > L.length)
    {
        return ERROR;
    }
    *e = L.elem[i-1];
    return OK;
}

/*
 * �Ƚ�����Ԫ���Ƿ����
 */
Status compare(Elemtype e1, Elemtype e2)
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
 * ����Ԫ��           �� L       Ԫ��ֵ          ����״̬ 
 */
Status FindElem(const SqList L, Elemtype e, Status (*compare)(Elemtype, Elemtype))
{
    int i;
    for (i = 0; i < L.length; i++)
    {
        if (!(*compare)(L.elem[i], e))// (*compare)(L.elem[i], e) ��� Ϊ0 
        {
            return i + 1;// ����Ԫ��λ�� 
        }
    }
    if (i >= L.length)
    {
        return ERROR;
    }
}

/*
 * ����ǰ��Ԫ��       �� L      ��ǰԪ��ֵ      ǰ���Ԫ��ֵ    PreElem(L, 6, &e);  &eȡ��ַ 
 */
Status PreElem(const SqList L, Elemtype cur_e, Elemtype *pre_e)
{
    int i;
    for (i = 0; i < L.length; i++)
    {
        if (cur_e == L.elem[i])//���ҵ�ǰԪ��λ�� 
        {
            if (i != 0)//ȷ��  i - 1��Խ�� 
            {
                *pre_e = L.elem[i - 1];//����ǰһ��Ԫ��ֵ  
            }
            else
            {
                return ERROR;
            }
        }
    }
    if (i >= L.length)
    {
        return ERROR;
    }
}

/*
 * ���Һ��Ԫ��       �� L      ��ǰԪ��ֵ      �����Ԫ��ֵ    NextElem(L, 6, &e);  &eȡ��ַ 
 */
Status NextElem(const SqList L, Elemtype cur_e, Elemtype *next_e)
{
    int i;
    for (i = 0; i < L.length; i++)
    {
        if (cur_e == L.elem[i])//���ҵ�ǰԪ��λ�� 
        {
            if (i < L.length - 1)//ȷ��  i + 1��Խ�� 
            {
                *next_e = L.elem[i + 1];
                return OK;
            }
            else
            {
                return ERROR;
            }
        }
    }
    if (i >= L.length)
    {
        return ERROR;
    }
}

/*
 * ����Ԫ��        �� L     λ��   �����ֵ    InsertElem(&L, i + 1, i);
 */
Status InsertElem(SqList *L, int i, Elemtype e)
{
    Elemtype *new;//ָ�� 
    if (i < 1 || i > L->length + 1)//λ��Խ�� 
    {
        return ERROR;
    }
    if (L->length >= L->size)//ʵ�ʴ�С ���ڵ��� ���ٵĿռ��С 
    {
        new = (Elemtype*) realloc(L->elem, (L->size + INCREMENT_SIZE) * sizeof(Elemtype));//��ԭ���ڴ�ռ��� �� ����һ�δ洢�ռ� 
        if (!new)//���ٳ���  ���ܵ��� ϵͳ�ռ� 
        {
            return ERROR;
        }
        L->elem = new;//�µĴ洢�ռ�  Ԫ�� 
        L->size += INCREMENT_SIZE;//�µ� �����С 
    }
    Elemtype *p = &L->elem[i - 1];//Ŀ��λ��Ԫ�ص�ַ ָ�� 
    Elemtype *q = &L->elem[L->length - 1];//���Ԫ�ص�ַָ��  
    for (; q >= p; q--)
    {
        *(q + 1) = *q;//�����Ԫ�ؿ�ʼ һ����� �ƶ�һ��λ�� 
    }
    *p = e;//�������Ԫ�ش��� 
    ++L->length;//ʵ�ʳ���+1 
    return OK;
}

/*
 * ɾ��Ԫ�ز�������ֵ   ��L   ��Ҫɾ����λ��  ���ر�ɾ����Ԫ��   DeleteElem(&L, 1, &e)  &e ȡ��ַ ָ�������ַ��ָ�� 
 */
Status DeleteElem(SqList *L, int i, Elemtype *e)
{
    if (i < 1 || i > L->length)//λ��Խ�� 
    {
        return ERROR;
    }
    Elemtype *p = &L->elem[i - 1];// Ŀ��λ��Ԫ�ص�ַ ָ�� pָ��ĵ�ַ 
    *e = *p;//Ŀ��Ԫ��ֵ 
    for (; p < &L->elem[L->length]; p++)//Ԫ�ص�ַ ��Χ 
    {
        *(p) = *(p + 1);//��ǰ �ƶ�һ��λ�� 
    }
    --L->length;//ʵ�ʳ���-1 
    return OK;
}

/*
 * ����Ԫ��
 */
void visit(Elemtype e)
{
    printf("%d ", e);
}

/*
 * �������Ա�
 */
Status TraverseList(const SqList L, void (*visit)(Elemtype))
{
    int i;
    for(i = 0; i < L.length; i++)
    {
        visit(L.elem[i]);
    }
    return OK;
}

/*
 * ����������
 */
int main()
{
    SqList L;// �ṹ����� 
    if (InitList(&L))//��ʼ���� 
    {
        Elemtype e;//Ԫ�ر��� 
        printf("init_success\n");
        int i;
        for (i = 0; i < 10; i++)
        {
            InsertElem(&L, i + 1, i);//��1λ�����Ԫ��  10�� 
        }
        printf("length is %d\n", getLength(L));//��ӡ �� ���� 
        if (GetElem(L, 1, &e)) {
            printf("The first element is %d\n", e);
        }
        else
        {
            printf("element is not exist\n");        
        }
        if (isEmpty(L))//��Ϊ�գ� 
        {
            printf("list is empty\n");
        }
        else
        {
            printf("list is not empty\n");
        }
        printf("The 5 at %d\n", FindElem(L, 5, *compare));//����Ԫ�� 
        PreElem(L, 6, &e);//ǰ��Ԫ�� 
        printf("The 6's previous element is %d\n", e);
        NextElem(L, 6, &e);//����Ԫ�� 
        printf("The 6's next element is %d\n", e);
        DeleteElem(&L, 1, &e);//ɾ��Ԫ�� 
        printf("delete first element is %d\n", e);
        printf("list:");
        TraverseList(L,visit);//���� �� 
        if (DestroyList(&L))//ɾ�� �� 
        {
            printf("\ndestroy_success\n");
        }
    }
}
