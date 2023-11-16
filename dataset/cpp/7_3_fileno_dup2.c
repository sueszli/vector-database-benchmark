#include <stdio.h>
#include <stdlib.h> //exit()
#include <unistd.h>// execXXX()���� 
#include <errno.h> // errno ���� 
#include <string.h>// strerror() ���� 
#include <sys/wait.h>// waitptd() �ȴ����� ����  window Ҳû�� 
// ��������
void error(char *msg){
	fprintf(stderr, "%s %s\n", msg, strerror(errno));
	exit(1);//��������ֹ���� �����˳�״̬�� 1   exit() ��Ψһû�з���ֵ ���Ҳ���ʧ�ܵĺ��� 
} 
 int main(int argc, char* argv[])
 {
   char* phrase = argv[1];//��Ҫ������ �ı� 
     char* vars[] = {"RSS_FEED=http://www.cnn.com/rss/celebs.xml", NULL};
     FILE  *f = fopen("stories.txt","w");// д��ʽ���ļ�
	 if(!f){//�򲻿��ļ� f=0 
	  error("���ļ�ʧ��"); 	
	 }
	 
     pid_t pid = fork();// �᷵��һ������ֵ �ӽ��̷���0  �����̷���һ������ ���Ƴ����� -1 
     if(pid == -1) {
		error("���ܸ��ƽ���");
		} 
		
     if(!pid){// pid=0Ϊ �ӽ��� ���µĽ��̽��� 
         if(dup2(fileno(f),1) == -1){//fileno(f)��ȡ�ļ�  ������ ���������ж�Ӧ�� ������   0~255  0 1 2�̶� Ϊ��־���� ��־��� ��־���� 
         	error("�����ض����־���������");
         }
         // �����ӳ���  ��Ҫһ����ʱ��  
	     if (execle("/usr/bin/python", "/usr/bin/python", "./rssgossip.py", NULL, vars) == -1) {
			error("�������нű�"); 
	     }
     }
     // �ȴ��ӽ��̽���
	int pid_status;
	if(waitpid(pid, &pid_status, 0)==-1){//�ӽ���ID �ӽ����˳���Ϣ int ָ��  ѡ��0 �ȴ����� 
		error("�ȴ��ӽ��̳���");
	}
	if(WEXITSTATUS(pid_status))//ʹ�ú�鿴 �˳�״̬���� 0  pid_status ���� ��ͬ��λ��ʾ��ͬ����Ϣ 
	  puts("�˳�״̬����0"); // ��Ȼ���� Ϊ0  ��ɱ��Ϊ��0 �� 
	
   return 0;
 }
