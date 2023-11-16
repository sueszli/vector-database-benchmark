#include <stdio.h>
#include <stdlib.h>

int n;
/*
������ȫ�������������Ѻ���С�ѣ����������Ǹ�����ֵ���ӽ���
��Ӧ����С�Ѿ��Ǹ�����ֵ���ӽڵ�С��

������������������ѣ�����С�ѣ��Ѷ���¼�Ĺؼ�����󣨻���С����һ������
ʹ���ڵ�ǰ��������ѡȡ��󣨻���С���ؼ��ֱ�ü򵥡�������Ϊ�������Ļ���˼����ǣ�

�Ƚ���ʼ�ļ�R[1..n]����һ�����ѣ��˶�Ϊ��ʼ����������
�ٽ��ؼ������ļ�¼R[1]�����Ѷ����������������һ����¼R[n]������
�ɴ˵õ��µ�������R[1..n-1]��������R[n]��������R[1..n-1].keys��R[n].key��
���ڽ������µĸ�R[1]����Υ�������ʣ���Ӧ����ǰ������R[1..n-1]����Ϊ�ѡ�
Ȼ���ٴν�R[1..n-1]�йؼ������ļ�¼R[1]�͸���������һ����¼R[n-1]������
�ɴ˵õ��µ�������R[1..n-2]��������R[n-1..n]�����������ϵR[1..n-2].keys��R[n1..n].keys��
ͬ��Ҫ��R[1..n-2]����Ϊ�ѣ� �ظ��˲���ֱ��ȫ������


*/ 
/*
 * ���ɶ�
 */
void HeapAdjust(int *array, int s, int m)
{
    int i;
    array[0] = array[s];
    for (i = s * 2; i <= m; i *= 2)
    {
        if (i < m && array[i] < array[i + 1])
        {
            i++;
        }
        if (!(array[0] < array[i]))
        {
            break;
        }
        array[s] = array[i];
        s = i;
    }
    array[s] = array[0];
}

/*
 * ������
 */
void HeapSort(int *array)
{
    int i;
    for (i = n / 2; i > 0; i--)
    {
        HeapAdjust(array, i, n);
    }
    for (i = n; i > 1; i--)
    {
        array[0] = array[1];
        array[1] = array[i];
        array[i] = array[0];
        HeapAdjust(array, 1, i - 1);
    }
}

int main()
{
    int i;
    int *array;
    printf("����������Ĵ�С��");
    scanf("%d", &n);
    array = (int*) malloc(sizeof(int) * (n + 1));
    printf("���������ݣ��ÿո�ָ�����");
    for (i = 1; i <= n; i++)
    {
        scanf("%d", &array[i]);
    }
    HeapSort(array);
    printf("�����Ϊ��");
    for (i = 1; i <= n; i++)
    {
        printf("%d ", array[i]);
    }
    printf("\n");
}
