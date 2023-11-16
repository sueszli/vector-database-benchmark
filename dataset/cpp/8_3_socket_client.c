 #include <stdio.h>
 #include <string.h>
 #include <errno.h>
 #include <stdlib.h>
 #include <sys/socket.h>//socket
 #include <arpa/inet.h>//ip 
 #include <unistd.h>
 #include <netdb.h>// getaddrinfo ������ȡ  ip 
 
 void error(char* msg)
 {
   fprintf(stderr, "%s: %s\n", msg, strerror(errno));
   exit(1);
 }
 /*
//����IP ��ʽ�� socket  dns ������������û�ж�Ӧ������ʱ ��֪�� ip��ַ 
int s = socket(PF_INET, SOCKET_STREAM, 0);// Э�� ?socket������ ? 0ΪЭ���
if( s == -1) error("�޷����׽���");
struct sockaddr_in name;
name.sin_family = PF_INET;
name.sin_addr.s_addr = inet_addr("208.201.239.100");//Զ�� ������ ip��ַ ? getaddrinfo()��ȡ������ip ?#include<netdb.h>
name.sin_port = htons(80);// ������� ��������� ������ 80 �˿�
int c = connect(s, (struct sockaddr *) &name, sizeof(name));
if(c == -1) error("�޷����ӷ�����");
 */ 
 
 // ��������ʽ�� socket   �Ⱦ����� �������� ������������ע�� 
 int open_socket(char* host, char* port)
 {
   struct addrinfo *res;//������� �ṹ��ָ��
   struct addrinfo hints;//�ṹ�����
   memset(&hints, 0, sizeof(hints));//��ʼ��Ϊ0
   hints.ai_family = PF_UNSPEC; //Э������ AF_UNSPEC  0 Э���޹�  AF_INET 2 IPv4Э�� ? AF_INET6 23 IPv6Э��
   hints.ai_socktype = SOCK_STREAM; //�������� SOCK_STREAM 1 ��  SOCK_DGRAM  2  ���ݱ�
   if (getaddrinfo(host, port, &hints, &res) == -1)// �˿�80 ���� "www.oreilly.com" �Ľ�����Ϣ
     error("Can't resolve the address");
   int d_sock = socket(res->ai_family, res->ai_socktype,res->ai_protocol);//����socket 
   if (d_sock == -1)
     error("Can't open socket");
   int c = connect(d_sock, res->ai_addr, res->ai_addrlen);//����
   freeaddrinfo(res);//���Ӻ�ɾ�� ��ַ��Ϣ ?����Ϣ�洢�� ���� ��Ҫ�ֶ����
   if (c == -1)
     error("Can't connect to socket");
   return d_sock;
 }
 
  //������Ϣ �������� 
 int say(int socket, char* s)
 {
   int result = send(socket, s, strlen(s), 0);
   if (result == -1)
     fprintf(stderr, "%s: %s\n", "Error talking to the client", strerror(errno));
   return result;
 }
 
 
 int main(int argc, char* argv[])
 {
   int d_sock;
   d_sock = open_socket("en.wikipedia.org", "80");
   char buf[255];
   // �ͻ��������Ϸ�������
   // 1��Ҫ�ȷ���  GET ���� 
   sprintf(buf, "GET /wiki/%s http/1.1\r\n", argv[1]);//�����Ӧ����ҳ 
   say(d_sock, buf);
   // 2 ������  + ���� \r\n 
   say(d_sock, "Host: en.wikipedia.org\r\n\r\n");
   // ���ջ����� 
   char rec[256];
   int bytesRcvd = recv(d_sock, rec, 255, 0);
   while (bytesRcvd) {//���ܽ��յ���Ϣ 
     if (bytesRcvd == -1)
       error("Can't read from server");
     rec[bytesRcvd] = '\0';// '\0' �滻  '\r' 
     printf("%s", rec);//��ӡ�յ�����Ϣ 
     bytesRcvd = recv(d_sock, rec, 255, 0);//�ٴν��� ���(255)������ 
   }
   close(d_sock);//�ر�socket���� 
   return 0;
 }
 
