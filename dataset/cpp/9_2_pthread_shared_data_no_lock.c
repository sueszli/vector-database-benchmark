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
 int beers = 2000000;//���� 
 // void* ���� 
 void* drink_lots()
 {
   int i;
   for (i = 0; i < 100000; i++) {
     beers = beers - 1;//ÿ�� -1 
   }
   return NULL;
 }
 int main()
 {
   pthread_t threads[20];//������Ϣ 
   int t;
   printf("%i bottles of beer on the wall\n%i bottles of beer\n", beers, beers);
   for (t = 0; t < 20; t++) {
     pthread_create(&threads[t], NULL, drink_lots, NULL);//�������� 
   }
   void* result;
   for (t = 0; t < 20; t++) {
     pthread_joint(threads[t], &result);//�ȴ����̽��� 
   }
   printf("There are now %i bottles of beer on the wall\n", beers);
   return 0;
 }
