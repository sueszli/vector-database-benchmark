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
 void* drink_lots_lock1()
 {
   int i;
   pthread_mutex_t a_lock = PTHREAD_MUTEX_INITIALIZER;//��ʼ���� 
   pthread_mutex_lock(&a_lock);//������   �� pthread_mutex_unlock(&a_lock);//����  ֮�� ͬʱֻ��ͨ��һ���߳� 
   for (i = 0; i < 100000; i++) {
     beers = beers - 1;//ÿ�� -1 
   }
   pthread_mutex_unlock(&a_lock);//���� 
   return NULL;
 }
 // �����lock  ÿ���߳� һ�� �ȵ� 100000 ƿơ��
 
  
  // void* ���� 
 void* drink_lots_lock2()
 {
   int i;
   for (i = 0; i < 100000; i++) {
   	 pthread_mutex_t a_lock = PTHREAD_MUTEX_INITIALIZER;//��ʼ���� 
     pthread_mutex_lock(&a_lock);//������   �� pthread_mutex_unlock(&a_lock);//����  ֮�� ͬʱֻ��ͨ��һ���߳� 
     beers = beers - 1;//ÿ�� -1 
     pthread_mutex_unlock(&a_lock);//���� 
   }
   return NULL;
 }
  // �����lock  ÿ���߳� һ�� �ȵ� 1 ƿơ��
 
 int main()
 {
   pthread_t threads[20];//������Ϣ 
   int t;
   printf("%i bottles of beer on the wall\n%i bottles of beer\n", beers, beers);
   for (t = 0; t < 20; t++) {
     pthread_create(&threads[t], NULL, drink_lots_lock1, NULL);//�������� 
     //pthread_create(&threads[t], NULL, drink_lots_lock2, NULL);//�������� 
   }
   void* result;
   for (t = 0; t < 20; t++) {
     pthread_joint(threads[t], &result);//�ȴ����̽��� 
   }
   printf("There are now %i bottles of beer on the wall\n", beers);
   return 0;
 }
