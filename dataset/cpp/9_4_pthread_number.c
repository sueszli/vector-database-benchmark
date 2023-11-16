#include <stdio.h>
#include <stdlib.h>
#include <string.h>// strerror 
#include <unistd.h>// ���� 
#include <errno.h>//errno 
#include <pthread.h>//�߳� 

 void* do_stuff(void* param)//���� һ�� void* ���Ͳ��� 
 {
   long thread_no = (long)param;//ת����long 
   printf("Thread number %ld\n", thread_no);//��ӡ��� 
   return (void*)(thread_no + 1);//���� ����+1 void* ���� 
 }
  
 int main()
 {
   pthread_t threads[20];
   long t; 
   for (t = 0; t < 3; t++) {
     pthread_create(&threads[t], NULL, do_stuff, (void*)t);//��long �ͱ���t ת���� void*ָ������ 
   }
   void* result;
   for (t = 0; t < 3; t++) {
     pthread_join(threads[t], &result);
     printf("Thread %ld returned %ld\n", t, (long)result);//ת��long��  ���� ���̽�� 
   }
   return 0;
 }
