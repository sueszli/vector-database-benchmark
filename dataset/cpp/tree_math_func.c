#include <stdio.h>
#include <stdlib.h>

#define TRUE 1
#define FALSE 0
#define OVERFLOW -2
#define OK 1
#define ERROR 0
/*
���õ����ṹ--�������������ص���һ������ֱ���ӽڵ����ֻ����������
����������֮�֡��ڶ����������ֳ����ĳ�Ϊ��ȫ�������Ľṹ��
�����ص��ǳ����һ����ÿһ��Ľ����Ϊ2^i-1��
���һ��Ľ������������2^i-1����ô���һ��Ľ���������������е�

*/
/** 
 * �����������׺���ʽΪ��  
 * ��a+b)*((c+d)*e+f*h*g)  
 *
 * 1���Զ�������Ĵ洢�ṹ����һ�á�
 * 2���ȸ�����Ϊ������׺���ʽ��ǰ׺���ʽ  
 * 3���������Ϊ������׺���ʽ�ĺ�׺���ʽ   
 *
          *
       +            +  
    a   b      *        * 
            +     e    f   * 
          c  d            h   g 
  */ 
   
typedef int Status;
typedef int TElemType;

/*
 * �洢�ṹ
 */
typedef struct BiTNode
{
    union{  
        int opnd; // �� 
        char optr; //������ 
    }val;    //����
    
    struct BiTNode *lchild, *rchild;//����  �Һ��� �ṹ��ָ��
}BiTNode, *BiTree;//�ṹ�����  �ṹ��ָ��

/*
 * ����������,����0��ʾ��������
 */
Status CreateBiTree(BiTree *T)
{
    TElemType e;
    scanf("%d", &e);//����һ��������
    if (e == 0)
    {
        *T = NULL;//0Ϊ����
    }
    else
    {
        *T = (BiTree) malloc(sizeof(BiTNode));//��ʼ��һ���ṹ����ڴ�ռ�
        if (!T)//����ʧ��
        {
            exit(OVERFLOW);
        }
        (*T)->data = e;//��ʼ����  ������
        CreateBiTree(&(*T)->lchild);    //����������  ����  0�Ļ� ������Ϊ��
        CreateBiTree(&(*T)->rchild);    //����������        1�Ļ��ֻ���� ���������� ������  0 ������Ϊ��  1 ������
		//�ֻᴴ�������� ������ 0 ������Ϊ�� 0 ������Ϊ�� ���� 
    }
    return OK;
}

/*
 * ����Ԫ��
 */
void visit(TElemType e)
{
    printf("%d ", e);
}

/*
 * ���������������ָ�ȷ��ʸ���Ȼ����ʺ��ӵı�����ʽ
 */
Status PreOrderTraverse(BiTree T, void (*visit)(TElemType))
{
    if (T)
    {
        visit(T->data);//�ȷ��ʸ� Ԫ��
        PreOrderTraverse(T->lchild, visit);//���������� Ԫ��
        PreOrderTraverse(T->rchild, visit);//����������Ԫ��
    }
}

/*
 * ���������������ָ�ȷ������ң����ӣ�Ȼ����ʸ����������ң��󣩺��ӵı�����ʽ
 */
Status InOrderTraverse(BiTree T, void (*visit)(TElemType))
{
    if (T)
    {
        InOrderTraverse(T->lchild, visit);//���������� Ԫ��
        visit(T->data);//���ʸ� Ԫ��
        InOrderTraverse(T->rchild, visit);//����������Ԫ��
    }
}

/*
 * ���������������ָ�ȷ��ʺ��ӣ�Ȼ����ʸ��ı�����ʽ
 */
Status PostOrderTraverse(BiTree T, void (*visit)(TElemType))
{
    if (T)
    {
        PostOrderTraverse(T->lchild, visit);//�ȷ��������� Ԫ��
        PostOrderTraverse(T->rchild, visit);//����������Ԫ��
        visit(T->data);//���ʸ� �� Ԫ��
    }
}

int main()
{
    BiTree T;
    printf("������������0Ϊ������\n");
    CreateBiTree(&T);
    printf("���������");
    PreOrderTraverse(T, *visit);
    printf("\n���������");
    InOrderTraverse(T, *visit);
    printf("\n���������");
    PostOrderTraverse(T, *visit);
    printf("\n");

    return 0;
}

