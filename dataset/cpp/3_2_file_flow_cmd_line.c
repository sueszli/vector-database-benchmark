#include<stdio.h>
#include<stdlib.h>
#include<string.h>
 
// gcc file_flow.c -o categorize && ./categorize me me.csv ELV elv.csv the_rest.csv
int main(int argc, char *argv[]){//��ʹ�������в��� 
    if(argc!=6){
    	fprintf(stderr,"��Ҫ5������ 5 arguments \n");
    	return 1;
    } 
    char line[80];//�ַ�����
    
	FILE *in;//���� �ļ������� ָ�� in
	if( !(in = fopen("spooky.csv","r"))){
		fprintf(stderr,"�ļ������� file not existance \n");
		return 1;
	}
	
	FILE *out_file1 = fopen(argv[2],"w");          // д��ʽ�� 
    FILE *out_file2 = fopen(argv[4],"w");// д��ʽ�� 
	FILE *out_file3 = fopen(argv[5],"w");         // д��ʽ�� 
	while(fscanf(in,"%79[^\n]", line)==1){//�ɹ���ȡһ�� ���� ���� line�� 
		if(strstr(line, argv[1]))
			fprintf(out_file1, "%s\n",line); 
		if(strstr(line, argv[3]))
			fprintf(out_file2, "%s\n",line);			
		else
			fprintf(out_file3, "%s\n",line);
					
	}
	fclose(out_file1);//flow closed  �ر��ļ������� 
	fclose(out_file2);
	fclose(out_file3);
	
	return 0;
} 
