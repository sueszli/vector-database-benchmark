#include<stdio.h>
#include<unistd.h>//ʹ�������� �������� geopt()���� 

// ./ord -d now -t pineapple apple as
int main(int argc, char *argv[]){//��ʹ�������в��� 

    char *delivery = "";
	int tick = 0;
	int count = 0;
	char ch;
	
	while((ch=getopt(argc, argv, "d:t"))!=EOF){//����������Ҫ ѡ�� -d  ��:���� -dѡ����Ҫ���� ���� -tѡ�� 
		switch(ch){
			case 'd'://-d ѡ�� 
				delivery = optarg;
				break;
			case 't'://-tѡ�� 
				tick = 1;
				break;
			default://����ѡ�� ���� 
				fprintf(stderr,"δ֪ѡ��\n");
				return 1;
			
		} 
	}
	argc -= optind;//���� optind�б����� ��λ 
	argv += optind;//ָ����λ 
	
	if(tick) puts("Tick crust.");
	if(delivery[0] ) printf("To be delivered %s.\n",delivery);
	puts("���������в���");
	for(count=0;count<argc;count++)//getopt()������֮�� ��һ�������ͱ���� argv[0] 
		puts(argv[count]); 
	
	return 0;
} 
