#include<stdio.h>
#include<stdlib.h>
#include<string.h>
 
// gcc file_flow.c -o categorize && ./categorize
int main(){
    char line[80];//�ַ�����
	FILE *in = fopen("spooky.csv","r");//ֻ����ʽ���ļ� ���� �ļ������� ָ�� in
	FILE *out_file1 = fopen("ufos.csv","w");          // д��ʽ�� 
    FILE *out_file2 = fopen("disappearances.csv","w");// д��ʽ�� 
	FILE *out_file3 = fopen("other.csv","w");         // д��ʽ�� 
	while(fscanf(in,"%79[^\n]", line)==1){//�ɹ���ȡһ�� ���� ���� line�� 
		if(strstr(line, "UFO"))
			fprintf(out_file1, "%s\n",line); 
		if(strstr(line, "Disappearances"))
			fprintf(out_file2, "%s\n",line);			
		else
			fprintf(out_file3, "%s\n",line);
					
	}
	fclose(out_file1);//flow closed  �ر��ļ������� 
	fclose(out_file2);
	fclose(out_file3);
	
	return 0;
} 
