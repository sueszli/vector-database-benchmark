#include<stdio.h>
#include<stdlib.h>   // stdlib ͷ�ļ��� standard library ��׼��ͷ�ļ� ʹ�� exit(0) 
int main()
{
   FILE * fp;//�ļ����͵�ָ�� 
   char ch,filename[10];//�ַ� ���ַ����� 
   printf("Please enter the file name:");

   scanf("%s",filename);//�ļ��� 
   if((fp=fopen(filename,"w"))==NULL)    // ��writeд�뷽ʽ ������ļ���ʹ fp ָ����ļ�
   {
      printf("Unable to open this file\n");     // ����򿪳�����������򲻿�������Ϣ
      exit(0);     // ��ֹ����   #include<stdlib.h> 
   }
   ch=getchar();     // �������������ļ���ʱ �������Ļس���
   
   printf("Please enter a string  in the disk��Ends with a #����");
   ch=getchar();     // ���մӼ�������ĵ�һ���ַ�
   while(ch!='#')     // ������ # ʱ����ѭ��
   {
      fputc(ch,fp);//д��һ���ַ� 
      putchar(ch);
      ch=getchar();
   }

   fclose(fp);//�ر��ļ� 
  // putchar(10);
   return 0;
}
