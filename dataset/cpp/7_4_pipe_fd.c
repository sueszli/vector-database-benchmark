#include <stdio.h>
#include <stdlib.h> //exit()
#include <unistd.h>// execXXX()���� 
#include <errno.h> // errno ���� 
#include <string.h>// strerror() ���� 
//#include <sys/wait.h>// waitptd() �ȴ����� ����  window Ҳû�� 
// ��������
void error(char *msg){
	fprintf(stderr, "%s %s\n", msg, strerror(errno));
	exit(1);//��������ֹ���� �����˳�״̬�� 1   exit() ��Ψһû�з���ֵ ���Ҳ���ʧ�ܵĺ��� 
} 

// ��url����
void open_url(char * url){
	char launch[255];
	sprintf(launch, "cmd /c start %s", url);//window �´���ҳ
	system(launch);
	sprintf(launch, "x-www-browser '%s' &", url);//linux�´���ҳ
	system(launch);
	sprintf(launch, "open '%s'", url);//mac�´���ҳ
	system(launch);	
	
} 
 int main(int argc, char* argv[])
 {
   char* phrase = argv[1];//��Ҫ������ �ı� 
   char* vars[] = {"RSS_FEED=http://www.cnn.com/rss/celebs.xml", NULL};
    
    int fd[2];//�ܵ� ����� д���fd[1]  �����fd[0] ������ 
	if(pipe(fd) == -1){
		error("�����ܵ�ʧ��");
	}// pipe�����Ĺܵ������ļ� ����ʹ���� ���ӽ�����
	// ����ʹ�� mkfifo() ϵͳ���ô���  ʵ�� �����ܵ� FIFO  
	
	
     pid_t pid = fork();// �᷵��һ������ֵ �ӽ��̷���0  �����̷���һ������ ���Ƴ����� -1 
     if(pid == -1) {
		error("���ܸ��ƽ���");
		} 
		
     if(!pid){// pid=0Ϊ �ӽ��� ���µĽ��̽��� 
         close(fd[0]);// �ӽ��� �����ȡ �ܵ������ 
		 dup2(fd[1],1);//�ӽ��� �ı�־��� ���� �ܵ���д��� 
         // �����ӳ���  ��Ҫһ����ʱ��  
	     if (execle("/usr/bin/python", "/usr/bin/python", "./rssgossip.py", "-u", phrase, NULL, vars) == -1) {
			error("�������нű�"); //�ӽ��������� RSS_FEED������  phrase ���� url 
	     }
     }
     //������ ������
	 close(fd[1]);//�ر� �ܵ�дfd[1] 
	 dup2(fd[0],0);//������ ��׼����stdin ������0 �ض��� �ܵ����� fd[0] 
     char line[255];
	 while(fgets(line, 255, stdin)) {
	 	if(line[0] == '\t')
	 	open_url(line + 1);
	 }
     
   return 0;
 }
