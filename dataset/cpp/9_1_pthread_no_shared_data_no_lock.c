#include <stdio.h>
#include <stdlib.h>
#include <string.h>// strerror 
#include <unistd.h>// ���� 
#include <errno.h>//errno 
#include <pthread.h>//�߳� 
 void error(char* msg)
 {
   fprintf(stderr, "%s: %s\n", msg, strerror(errno));
   exit(1);
 }
 
 // �̺߳��� ����Ϊ void*   ���� ��ָ������ ����Ϊ�������� 
 void* does_not()
 {
   int i = 0;
   for (i = 0; i < 5; i++) {
     sleep(1);
     puts("Does not!");
   }
   return NULL;
 }
 void* does_too()
 {
   int i = 0;
   for (i = 0; i < 5; i++) {
     sleep(1);
     puts("Does too!");
   }
   return NULL;
 }
 
 int main(){
   pthread_t t0;//�����߳���Ϣ 
   pthread_t t1;
   // �����߳� 
   if (pthread_create(&t0, NULL, does_not, NULL) == -1)
     error("Can't create thread t0");
   if (pthread_create(&t1, NULL, does_too, NULL) == -1)
     error("Can't create thread t1"); 
   // �ȴ��߳̽��� 
   void* result;
   if (pthread_join(t0, &result) == -1)
     error("Can't join thread t0");
   if (pthread_join(t1, &result) == -1)
     error("Can't join thread t1");
   return 0;
 } 
 
 
 
