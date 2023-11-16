#include <stdio.h>
#include <stdlib.h>
#include <string.h>//  strerror()
#include <errno.h>//  errno
#include <sys/socket.h> 
#include <arpa/inet.h> 
#include <signal.h>//�����ź�  �������̼�ͨ��
#include <unistd.h>// execXXX()����   ���̽������� 
// gcc socket_server.c -o server
// ./server ���������
// telnet 127.0.0.1 30000  �ͻ��� �������� 

// ��ӡ���� ����������   
  void error(char* msg)
 {
   fprintf(stderr, "%s: %s\n", msg, strerror(errno));
   exit(1);//���ս���� 
 }
 
 int listener_d;//�������׽��� 
 // ����������  ���� Ctrl-C 
 void handle_shutdown(int sig){
 	if(listener_d) 
 		close(listener_d);
 	fprintf(stderr, "�ټ���...\n");
    exit(0);	
 } 
 
//���� ��Ϣ  ��ִ�д��������� 
int catch_signal(int sig, void (*handler)(int)){//�źű�� ������ָ��
	struct sigaction action;  // �ṹ��
	action.sa_handler = handler;// �źŴ���������
	sigemptyset(&action.sa_mask); // ��������� sigactionҪ������ź�
	action.sa_flags = 0;  // ���ӱ�־λ
	return sigaction(sig, &action, NULL);
}
 
 
 // �� �������� �׽��� 
 int open_listener_socket()
 {
   int s = socket(PF_INET, SOCK_STREAM, 0);
   if (s == -1)
     error("Can't open socket");
   return s;
 }
 
 // �󶨶˿� 
 void bind_to_port(int socket, int port)
 {
   struct sockaddr_in name;
   name.sin_family = PF_INET;
   name.sin_port = (in_port_t)htons(30000);//�˿�  
   name.sin_addr.s_addr = htonl(INADDR_ANY);
   int reuse = 1;//�˿����� 
   if (setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, (char *)&reuse, sizeof(int)) == -1)
     error("Can't set the reuse option on the socket");
   int c = bind (socket, (struct sockaddr *) &name, sizeof (name));
   if (c == -1)
     error("Can't bind to socket");
 }
 //�����˿� 
 void listen_to_port(int socket, int queue_size)
 {
 	if(listen(socket, queue_size) == -1) 
		error("�޷�����");
	puts("Waiting for connection");
 } 
 // ��������
  
 //������Ϣ ���ͻ��� 
 int say(int socket, char* s)
 {
   int result = send(socket, s, strlen(s), 0);
   if (result == -1)
     fprintf(stderr, "%s: %s\n", "Error talking to the client", strerror(errno));
   return result;
 }
 // �Ӵӿͻ�������Ϣ 
    /* Handle the error */
 int read_in(int socket, char* buf, int len){
 // �ͻ��˹�����������\r\n��β 
   char* s = buf;
   int slen = len;
   int c = recv(socket, s, slen, 0);//���� һ�������� ����ʵ�ʽ��յ����ݳ���  
   while ((c > 0) && (s[c-1] != '\n')) {// ֱ��û�пɶ����ַ� ���߶����� \n 
     s++; slen -= c;
     c = recv(socket, s, slen, 0);
   }
   if (c < 0)//  recv ���ʹ��� ����-1 
     return c;
   else if (c == 0)//ʲô��û�ж��� 
     buf[0] = '\0';//���ؿ��ַ� 
   else
     s[c-1]='\0';//��'\0' ����'\r' 
   return slen - len;
 }
 
 //������ 
  int main(int argc, char* argv[])
 {
   if (catch_signal(SIGINT, handle_shutdown) == SIG_ERR)
     error("���������жϴ���������");
   //1 �����׽���  ���׽���  listener_d  �������� 
   listener_d = open_listener_socket();
   //2 �󶨶˿� 
   bind_to_port(listener_d, 30000);
   //3 �����˿� 
   listen_to_port(listener_d, 10); 
   //4  ���տͻ������� 
   struct sockaddr_storage client_addr;//�ͻ��˵�ַ 
   unsigned int address_size = sizeof client_addr;
   char buf[255];//������Ϣ������ 
   for(;;) {
   	// ���׽���  connect_d  ���� �ͻ��˹��� 
     int connect_d = accept(listener_d, (struct sockaddr *)&client_addr, &address_size);
     if (connect_d == -1)
       error("Can't open secondary socket");
     //5 ���� ��Ϣ 
     if(!fock()){// �ӽ��� fock() ����0
	 	 close(listener_d); //�ӽ��̹ر� �����׽��� ���� �����׽��� 
	     if (say(connect_d, "Internet Knock-Knock Protocol Server\r\nVersion 1.0\r\nKnock! Knock!\r\n> ")!= -1) {
	       //6 ������Ϣ 
	       read_in(connect_d, buf, sizeof(buf));
	       if (strncasecmp("Who's there?", buf, 12))//�Ƚ��ַ���  û���ҵ� ����1 
	          // ������Ϣ 
	         say(connect_d, "You should say 'Who's there?'!");
	       else {//��ȷ�ش� 
	         if (say(connect_d, "Oscar\r\n> ") != -1) {
	           read_in(connect_d, buf, sizeof(buf));
	           if (strncasecmp("Oscar who?", buf, 10))
	             say(connect_d, "You should say 'Oscar who?'!\r\n");
	           else
	             say(connect_d, "Oscar silly question, you get a silly answer\r\n");
	         }
           }
         }
         close(connect_d);//�ӽ��� ��ͻ���ͨ�Ž�����  �ر� �����׽���
		 exit(0); //�ӽ����˳� ���� 
     }    
     close(connect_d);//������ �ر� �����׽���  ���� �����׽��� 
   }
   return 0;
 }
 
 
