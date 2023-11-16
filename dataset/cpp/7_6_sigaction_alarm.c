#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>//  strerror()
#include <errno.h>//  errno
#include <signal.h>
int score = 0;
void end_game(int sig)
 {
   printf("\n���յ÷�: %i\n", score);
   exit(0);
 }
 
 void times_up(int sig)
 {
   puts("\nʱ�䵽��!");
   raise(SIGINT);
 }
int catch_signal(int sig, void (*handler)(int)){//�źű�� ������ָ��
	struct sigaction action;  // �ṹ��
	action.sa_handler = handler;// �źŴ���������
	sigemptyset(&action.sa_mask); // ��������� sigactionҪ������ź�
	action.sa_flags = 0;  // ���ӱ�־λ
	return sigaction(sig, &action, NULL);
}
 
 void error(char* msg)
 {
   fprintf(stderr, "%s: %s\n", msg, strerror(errno));
   exit(1);
 }
 int main()
 {
   catch_signal(SIGALRM, times_up);//��ʱ�������� SIGALRM �ź� �ٵ��� times_up���� ����  SIGTERM��ֹ���� 
   catch_signal(SIGINT,  end_game);//Ctrl-C �ź� 
   srandom (time (0));//ÿ�ζ��ܵõ���ͬ������� 
   while(1) {
     int a = random() % 11;
     int b = random() % 11;
     char txt[4];
     alarm(5);//5s �ڴ���  5s�ڻص�����ط� ���ӻ����� 
     printf("\nWhat is %i * %i = ? ", a, b);
     fgets(txt, 4, stdin);
     int answer = atoi(txt);//ת������ 
     if (answer == a * b)
       score++;
     else
       printf("\nWrong! Score: %i\n", score);
   }
   return 0;
 }
