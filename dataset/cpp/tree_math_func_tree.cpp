/** 
 * �����������׺���ʽΪ��  
 * ��a+b)*((c+d)*e+f*h*g)  
 *
 * 1���Զ�������Ĵ洢�ṹ����һ�á�
 * 2���ȸ�����Ϊ������׺���ʽ��ǰ׺���ʽ  
 * 3���������Ϊ������׺���ʽ�ĺ�׺���ʽ   
 *
 * 2005/04/28
 */
#include<stack>  
#include<iostream>  
#include<stdio.h> 
#include<stdlib.h> 
#include<ctype.h>  
#include<string.h> 
 
using   namespace   std;  

//////////////////////////////////////////////////////////////////////////  
//   �������Ͷ�����  
//  
typedef   struct nodeTag{ /*   ���ʽ�������������   */  
    union{  
        int opnd;  
        char optr;  
    }val;  
    struct   nodeTag *left;  
    struct   nodeTag *right;  
}treeNode;  

typedef   struct pTag{ /*   ���ȱ�������   */  
    char op;  
    int f;  
    int g;  
}Prior;  

//////////////////////////////////////////////////////////////////////////  
//   ȫ�ֱ���������  
//  
Prior pList[]   =   { /*   ���ȱ�   */  
    '+',   2,   1,  
    '-',   2,   1,  
    '*',   4,   3,  
    '/',   4,   3,  
    '^',   4,   5,  
    '(',   0,   5,  
    ')',   6,   0,  
    '$',   0,   0  
};  
stack<char> OptrStack; /*   ������ջ   */  
stack<treeNode*> ExprStack; /*   ���ʽջ   */  
const   int   NUM =   256;  
const   int   OPTR =   257;  
int tokenval; /*   ��һ����ֵ   */  

/**************************************************************************  
*   descr     :�Ƚ�ջ�����������һ������������ȹ�ϵ  
*   param     :opf   ջ�������  
*   param     :opg   ��һ���������  
*   return   :��ϵ'>',   '=',   '<'  
**************************************************************************/  
char   Precede(char   opf,   char   opg)  
{  
    int   op1=-1,op2=-1;  
    for   (int   i=0;   i   <   8;   i++)  
    {  
        if   (pList[i].op   ==   opf)  
            op1 =   pList[i].f;  
        if   (pList[i].op   ==   opg)  
            op2   =   pList[i].g;  
    }  
    if   (op1   ==   -1   ||   op2   ==   -1)  
    {  
        cout<<"operator   error!"<<endl;  
        exit(1);  
    }  
    if   (op1   >   op2)  
        return   '>';  
    else   if   (op1   ==   op2)  
        return   '=';  
    else  
        return   '<';  
}  

/**************************************************************************  
*   descr     :  
*   return   :  
**************************************************************************/  
int   lexan()  
{  
    int   t;  
    while(1)  
    {  
        t   =   getchar();  
        if   ( (t ==' ')||(t=='\t')||(t=='\n')); //ȥ���հ��ַ�  
        else   if   (isdigit(t))  
        {  
            ungetc(t,   stdin);  
            cin>>tokenval;  
            return   NUM;  
        }  
        else  
        {  
            return   t;  
        }    
    }  
}  
/**************************************************************************  
*   descr     :   ���������������(Ҷ���)  
*   param     :   num   ������  
*   return   :   ������Ҷ���ָ��   treeNode*  
**************************************************************************/  
treeNode*   mkleaf(int   num)  
{  
    treeNode   *tmpTreeNode   =   new   treeNode;  
    if   (tmpTreeNode   ==   NULL)  
    {  
        cout<<"Memory   allot   failed!"<<endl;  
        exit(1);  
    }  
    tmpTreeNode->left =   NULL;  
    tmpTreeNode->right =   NULL;  
    tmpTreeNode->val.opnd   =   num;  
    return   tmpTreeNode;  
}  

/**************************************************************************  
*   descr     :   ������������������(�ڽ��)  
*   param     :   op�����  
*   param     :   left������ָ��  
*   param     :   right������ָ��  
*   return   :   �������ڽ��ָ��   treeNode*  
**************************************************************************/  
treeNode*   mknode(char   op,   treeNode*   left,treeNode*   right)  
{  
    treeNode   *tmpTreeNode   =   new   treeNode;  
    if   (tmpTreeNode   ==   NULL)  
    {  
        cout<<"Memory   allot   failed!"<<endl;  
        exit(1);  
    }  
    if   (left   ==   NULL   ||   right   ==   NULL)  
    {  
        cout<<"Lossing   operand!"<<endl;  
        exit(1);  
    }  
    tmpTreeNode->left =   left;  
    tmpTreeNode->right =   right;  
    tmpTreeNode->val.optr   =   op;  
    return   tmpTreeNode;  
}  

/**************************************************************************  
*   descr     :   �������ʽ������(�ο���ε������ΰ��ġ����ݽṹ��P/53)  
*   return   :   �����������ָ��  
**************************************************************************/  
treeNode*   CreateBinaryTree()  
{  
    int     lookahead;  
    char   op;  
    treeNode   *opnd1,   *opnd2;  
    OptrStack.push('$');  
    lookahead   =   lexan();  
    while   (   lookahead   !=   '$'   ||   OptrStack.top()   !=   '$')  
    {  
        if   (lookahead   ==   NUM   )  
        {  
            ExprStack.push(   mkleaf(tokenval));  
            lookahead   =   lexan();  
        }  
        else  
        {  
            switch   (Precede(OptrStack.top(),   lookahead))  
            {  
            case   '<':  
                OptrStack.push(lookahead);  
                lookahead   =   lexan();  
                break;  
            case   '=':  
                OptrStack.pop();  
                lookahead   =   lexan();  
                break;  
            case   '>':  
                opnd2 =   ExprStack.top();ExprStack.pop();  
                opnd1 =   ExprStack.top();ExprStack.pop();  
                op =   OptrStack.top();OptrStack.pop();  
                ExprStack.push(   mknode(op,   opnd1,   opnd2));  
                break;  
            }  
        }  
    }  
    return   ExprStack.top();  
}  

/**************************************************************************  
*   descr     :   ���ǰ׺���ʽ  
*   param     :  
*   return   :  
**************************************************************************/  
int   PreOrderTraverse(treeNode*   T)  
{  
    if   (   T   ==   NULL)  
        return   1;  
    if(T->left   !=   NULL)  
    {  
        cout<<T->val.optr<<"   ";  
        if   (PreOrderTraverse(T->left))  
            if   (PreOrderTraverse(T->right))  
                return   1;  
        return   0;  
    }  
    else  
    {  
        cout<<T->val.opnd<<"   ";  
        return   1;  
    }  
}  

/**************************************************************************  
*   descr     :   �����׺���ʽ  
*   param     :  
*   return   :  
**************************************************************************/  
int   FollowOrderTraverse(treeNode*   T)  
{  
    if   (   T   ==   NULL)  
        return   1;  
    if   (   T->left   !=NULL)  
    {  
        if   (FollowOrderTraverse(T->left))  
            if   (FollowOrderTraverse(T->right))  
            {  
                cout<<T->val.optr<<"   ";  
                return   1;  
            }  
            return   0;  

    }  
    else  
    {  
        cout<<T->val.opnd<<"   ";  
        return   1;  
    }  
}  

//////////////////////////////////////////////////////////////////////////  
//   ������  
//  
int   main()  
{  
    treeNode   *ExprTree;  
    ExprTree   =   CreateBinaryTree();  
    PreOrderTraverse(ExprTree);  
    cout<<endl;  
    FollowOrderTraverse(ExprTree);  
    cout<<endl;  
    return 0; 
}   
