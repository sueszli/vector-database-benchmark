#include <stdio.h>
#include <stdlib.h>

int n;

/*
 * ð������
  �����鿪ʼ���αȽ���������Ԫ��
  С��������      ��������
  ���������      �������� 
 */
void BubbleSort(int *array)
{
    int i, j, temp;//�м���� 
    for (i = 0; i < n - 1; i++)//n-1�αȽ� ѭ�� 
    {
        for (j = 0; j < n - 1 - i; j++)//ÿ�αȽ�ѭ�� �ıȽϴ��� n-1-i 
        {
            if (array[j] > array[j + 1])//ǰ���Ԫ�ش� �Ƶ� ����  �������� 
            {
                temp = array[j];//������ 
                array[j] = array[j + 1];//С���Ƶ�ǰ�� 
                array[j + 1] = temp;//����Ƶ����� 
            }
        }
    }
}
void BubbleSort_digress(int *array)
{
    int i, j, temp;//�м���� 
    for (i = 0; i < n - 1; i++)//n-1�αȽ� ѭ�� 
    {
        for (j = 0; j < n - 1 - i; j++)//ÿ�αȽ�ѭ�� �ıȽϴ��� n-1-i 
        {
            if (array[j] < array[j + 1])//ǰ���Ԫ��С 
            {
                temp = array[j+1];//������ 
                array[j+1] = array[j];//С���Ƶ�ǰ�� 
                array[j] = temp;//����Ƶ����� 
            }
        }
    }
}

int main()
{
    int i;
    int *array,*array_src;
    printf("����������Ĵ�С��");
    scanf("%d", &n);
    array = (int*) malloc(sizeof(int) * n);//��ַ 
    array_src = (int*) malloc(sizeof(int) * n);
    printf("���������ݣ��ÿո�ָ�����");
    for (i = 0; i < n; i++)
    {
        scanf("%d", &array[i]);
        array_src[i]=array[i]; 
    }
    
    BubbleSort(array);
    printf("���������Ϊ��");
    for (i = 0; i < n; i++)
    {
        printf("%d ", array[i]);
    }
    BubbleSort_digress(array_src);
    printf("���������Ϊ��");
    for (i = 0; i < n; i++)
    {
        printf("%d ", array_src[i]);
    }
    printf("\n");
}
