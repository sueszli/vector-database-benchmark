#include <stdio.h>
#include <stdlib.h>
// �Ƚϸ�����λ�ϵ�Ԫ��
/*
���������Ǹ�ǰ��ļ��������㷨��ȫ��һ���������㷨��
ǰ��������㷨��Ҫͨ���ؼ���֮��ıȽϺ��ƶ���ʵ�֣�
������������Ҫ���йؼ���֮��ıȽϣ����ǽ�����ؼ��ֵ�˼����ʵ�ֵġ�
�������֣�ÿһλ�ϵ����־���һ���ؼ��֣�ÿһλ�����ַ�Χ���ǹؼ��ַ�Χ����
����Ҫ����Ϊ�������д��Ƚ���ֵ����������ͳһΪͬ������λ���ȣ�
��λ�϶̵���ǰ�油�㡣Ȼ�󣬴����λ��ʼ�����ν���һ������
���������λ����һֱ�����λ��������Ժ�,���оͱ��һ���������У�
����ͼ��ʾ�����ƴӵ�λ����λ�Ƚϣ����ǴӴιؼ��ֵ����ؼ��ֱȽϣ�
���ֳ�Ϊ���λ���ȣ�LSD������֮��Ϊ���λ���ȣ�MSD����

*/ 

int n;         //Ԫ�ظ���      4  6 100 ����Ԫ�� 
int bit_num;   //�������λ��   100�������λ    4 6 λ������ǰ�油�� 

/*
 * ��ȡ��Ӧλ���ϵ��������ҵ���
 */
int GetNumInPos(int num, int pos)
{
    int i, temp = 1;
    for (i = 0; i < pos - 1; i++)
    {
        temp *= 10;
    }
    return (num / temp) % 10;
}

/*
 * ��������LSD��
 */
void RadixSort(int *array)
{
    int radix = 10;
    int *count, *bucket, i, j, k;
    count = (int*) malloc(sizeof(int) * radix);
    bucket = (int*) malloc(sizeof(int) * n);
    for (k = 1; k <= bit_num; k++)
    {
        for (i = 0; i < radix; i++)
        {
            count[i] = 0;
        }
        //ͳ�Ƹ���Ͱ����ʢ���ݸ���
        for (i = 0; i < n; i++)
        {
            count[GetNumInPos(array[i], k)]++;
        }
        //count[i]��ʾ��i��Ͱ���ұ߽�����
        for (i = 1; i < radix; i++)
        {
            count[i] = count[i] + count[i - 1];
        }
        //����
        for (i = n - 1; i >= 0; i--)
        {
            j = GetNumInPos(array[i], k);
            bucket[count[j] - 1] = array[i];
            count[j]--;
        }
        //�ռ�
        for (i = 0, j = 0; i < n; i++, j++)
        {
            array[i] = bucket[j];
        }
    }
}

int main()
{
    int i;
    int *array;
    printf("������������ֵ�λ����");
    scanf("%d", &bit_num);
    printf("����������Ĵ�С��");
    scanf("%d", &n);
    array = (int*) malloc(sizeof(int) * n);
    printf("���������ݣ��ÿո�ָ�����");
    for (i = 0; i < n; i++)
    {
        scanf("%d", &array[i]);
    }
    RadixSort(array);
    printf("�����Ϊ��");
    for (i = 0; i < n; i++)
    {
        printf("%d ", array[i]);
    }
    printf("\n");
}
