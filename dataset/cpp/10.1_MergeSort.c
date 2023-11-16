#include <stdio.h>
#include <stdlib.h>

int n;
// �������� �ٺϲ�����
/*
�鲢�����ǽ����ڹ鲢�����ϵ�һ����Ч�������㷨��
������Ϊ���Ƚ�a[i]��a[j]�Ĵ�С����a[i]��a[j]��
�򽫵�һ��������е�Ԫ��a[i]���Ƶ�r[k]�У�����i��k�ֱ����1��
���򽫵ڶ���������е�Ԫ��a[j]���Ƶ�r[k]�У�����j��k�ֱ����1��
���ѭ����ȥ��ֱ������һ�������ȡ�꣬
Ȼ���ٽ���һ���������ʣ���Ԫ�ظ��Ƶ�r�д��±�k���±�t�ĵ�Ԫ��

*/ 
/*
 * �ϲ�
 */
void Merge(int *source, int *target, int i, int m, int n)
{
    int j, k;
    for (j = m + 1, k = i; i <= m && j <= n; k++)
    {
        if (source[i] <= source[j])
        {
            target[k] = source[i++];
        }
        else
        {
            target[k] = source[j++];
        }
    }
    while (i <= m)
    {
        target[k++] = source[i++];
    }
    while (j <= n)
    {
        target[k++] = source[j++];
    }
}

/* 
 * �鲢����
 */
 void MergeSort(int *source, int *target, int s, int t)
 {
     int m, *temp;
     if (s == t)
     {
         target[s] = source[s];
     }
     else
     {
         temp = (int*) malloc(sizeof(int) * (t - s + 1));
         m = (s + t) / 2;
         MergeSort(source, temp, s, m);
         MergeSort(source, temp, m + 1, t);
         Merge(temp, target, s, m, t);
     }
 }

 int main()
 {
     int i;
    int *array;
    printf("����������Ĵ�С��");
    scanf("%d", &n);
    array = (int*) malloc(sizeof(int) * n);
    printf("���������ݣ��ÿո�ָ�����");
    for (i = 0; i < n; i++)
    {
        scanf("%d", &array[i]);
    }
    MergeSort(array, array, 0, n - 1);
    printf("�����Ϊ��");
    for (i = 0; i < n; i++)
    {
        printf("%d ", array[i]);
    }
    printf("\n");
 }
