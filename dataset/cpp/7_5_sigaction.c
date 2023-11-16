#include <stdio.h>
#include <stdlib.h> //exit()
#include <signal.h>

void diediedie(int sig){      // �Զ����źŴ���������
  puts("�ټ���...");
  exit(1);
}
int catch_signal(int sig, void (*handler)(int)){//�źű�� ������ָ��
struct sigaction action;  // �ṹ��
action.sa_handler = handler;// �źŴ���������
sigemptyset(&action.sa_mask); // ��������� sigactionҪ������ź�
action.sa_flags = 0;  // ���ӱ�־λ
return sigaction(sig, &action, NULL);
}

int main(){
	/*
	SIGINT   �ж��ź�  Ctrl-C��
    SIGQUIT  ֹͣ���� 
    SIGFPE   ������� 
    SIGTRAP  ѯ�ʳ���ִ�е������� 
    SIGSEGV  ���ʷǷ��洢����ַ 
    SIGWINCH �ն˴��� ��С�ı� 
    SIGTERM  ��ֹ���� 
    SIGPIPE  ��һ��û���˶��Ĺܵ�д���� 
    SIGKILL  ���ܺ��Ե� ��ֹ�ź�   ����˵�� ������Ӧ�� Ҳ����ʹ��  SIGKILL �ս��� 
    SIGSTOP  ���ܺ��Ե� ��ͣ�ź�  
    SIGALRM  �����ź�  alrm 
    ������ kill�����ź�  �������� 
	���磺
	ps -a �г�������Ϣ ���н��̵Ľ��̺�
	kill  78222        //Ĭ�Ϸ���  SIGTERM �ź�
	kill -INT   78222  // ���� SIGINT�ź� 
	kill -SEGV  78222  // ���� SIGSEGV�ź� 
	kill -KILL  78222  // ���� SIGKILL�ź�   �ͳ���������
	
    �� raise()  �����ź� ������ 
	raise(SIGTERM);//������Լ����ź�  �ź�����  �յ��ͼ��źź� �� �����߼��ź�   ����ЧӦ 
	 
	 alarm(120); ��ʱ120��� ����  SIGALRM �ź� �ڼ��������������   Ҳsleep()��������ͬʱʹ�� 
	 // setitimer()  �����趨 ����֮һ�� ��ʱ ���� 
	 catch_signal(SIGALRM, do_other)//��ʱ���������� 
	
	 catch_signal(SIGALRM, SIG_DFL)//��ԭĬ�Ϸ�ʽ �����ź� 
	 catch_signal(SIGINT, SIG_IGN) // ����ĳ���ź�    SIGKILL SIGSTOP���ܱ����� 
	*/
	if(catch_signal(SIGITN, diediedie) == -1){// Ctrl-C֮�󴥷��ĺ���
	 fprintf(stderr, "����ָ���ж��źŴ�����"); 
	 exit(2); 
	}
	char name[30];
	printf("Please enter your name: \r\n");
	fgets(name, 30, stdin);
	printf("Hello %s \r\n",name);
	return 0;	
	
}
