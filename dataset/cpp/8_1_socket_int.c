#include <stdio.h>
#include <stdlib.h>
#include <string.h>//  strerror()
#include <errno.h>//  errno
#include <sys/socket.h> 
#include <arpa/inet.h> 

 void error(char* msg)
 {
   fprintf(stderr, "%s: %s\n", msg, strerror(errno));
   exit(1);
 }
 
int main(int argc, char* argv[])
 {
   char* advice[] = {
     "Take smaller bites\r\n",
      "Go for the tight jeans. No they do NOT make you look fat.\r\n",
      "One word: inappropriate\r\n",
      "Just for today, be honest. Tell your boss what you *really* think\r\n",
      "You might want to rethink that haircut\r\n"
   }; 
	// ����  �������� �׽��� 
	int listener_d = socket(PF_INET, SOCK_STREAM, 0);// Э��  socket������   0ΪЭ��� 
	if(listener_d == -1) 
		error("�޷����׽���");
	
	// �󶨺� 30���ڲ������ٴΰ�
	int reuse = 1;//���� ʹ�ö˿� 
    if (setsockopt(listener_d, SOL_SOCKET, SO_REUSEADDR, (char *)&reuse, sizeof(int)) == -1)
    	error("Can't set the reuse option on the socket");
	//�� �˿� 
	struct sockaddr_in name; 
	name.sin_family = PF_INET; 
	name.sin_port = (int_port_t)htons(30000);//�˿� ��Χ  0~65535  ͨ�� ѡ�� 1024���� 
	name.sin_addr.s_addr = htonl(INADDR_ANY); 
	int c = bind(listener_d, (struct sockaddr *) &name, sizeof(name)); 
	if(c == -1) 
		error("�޷��󶨶˿�");
		
	//���� �˿� 
	if(listen(listener_d, 10) == -1) 
	   // �����������Ϊ10  �Ŷ� �ж�� 10   ����Ŀͻ��˻ᱻ֪ͨ ������æ	
		error("�޷�����");
	puts("Waiting for connection");
	
	struct sockaddr_storage client_addr;//����ͻ��˵���ϸ��Ϣ 
	unsigned int address_size = sizeof(client_addr); 
	for (;;) {		
		// �������� 
		int connect_d = accept(listener_d, (struct sockaddr *) &client_addr, &address_size); 
		if(connect_d == -1) 
			error("�޷��򿪸��׽���);	
		
		// ��ʼͨ��  ��������	
		char *msg = advice[rand() % 5]; 
		if(send(connect_d, msg, strlen(msg), 0) == -1) //���һ������ �Ǹ߼�ѡ�� 0 �Ϳ�����
			error("send error");
		close(connect_d);
	} 
	return 0;
}
