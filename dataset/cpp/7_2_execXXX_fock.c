#include <stdio.h>
#include <stdlib.h>
 #include <unistd.h>// execXXX()���� 
 #include <errno.h> // errno ���� 
 #include <string.h>// strerror() ���� 
// ���� �洢�������еĳ��� 
// taskmgr window��  
//  ps -ef  linux�� 
// exec + l / v + p/e/pe ���� ��һ�������滻һ������ PID���� ������
// l list �����б� �� v vector ��������/���� ��p ·�� path ; e �������� environment 
//  C������ʹ�� getenv("������") ��ȡ����������ֵ
// char *my_env[] = {"������=ֵ",NULL}; //�������� ���һ������Ϊ NULL
// execle("diner_info", "diner_info", "4", NULL, my_env);//ʹ�� diner_info�滻��ǰ���� ǰ����������Ҫ��ͬ
// ִ�е�״̬ �����ڱ��� errno ��  puts(strderror(errno));��ѯ������Ϣ
/* ִ�г����� -1 ͬʱ �޸�  ���� errno
   EPERM=1    | ���������       Operation not permitted   |
 | ENOENT=2   | û�и��ļ���Ŀ¼ No such file or directory |
 | ESRCH=3    | û�иý���       No such process           |
 | EMULLET=81 | ���ͺ��ѿ�(����) Bad haircut  
*/ 
// #include <unistd.h>  // execXXX()���� 
// #include <errno.h>   // errno ���� 
// #include <string.h>  // strerron() ��ʾ������Ϣ
// ����鿴 ������Ϣ
/*
#include <unistd.h>  // execXXX()���� 
#include <errno.h>   // errno ���� 
#include <string.h>

int main(){// exec()���������������̳ɹ� ���������о������� 
  if(execl("/sbin/ifconfig","/sbin/ifconfig", NULL)==-1){
  	if(execl("ipconfig","ipconfig", NULL)==-1)
  	  fprintf(stderr,"�������� ipconfig: %s", strerron(erron));
  	  return 1;
  }
  return 0;
} 
*/ 

// �������� 
// coffee.c
// gcc coffee.c -o coffee
/*
 #include <stdio.h>
 #include <stdlib.h>
 int main(int argc, char* argv[])
 {
   char* w = getenv("EXTRA");
   if (!w)
     w = getenv("FOOD");
   if (!w)
     w = argv[argc - 1];
   char* c = getenv("EXTRA");
   if (!c)
     c = argv[argc - 1];
   printf("%s with %s\n", c, w);
   return 0;
 }
*/ 
//���  donuts with coffee 
/*
 #include <stdio.h>
 #include <stdlib.h>
  int main(int argc, char* argv[]){
      char* my_env[] = {"FOOD=coffee", NULL};
      if(execle("./coffee", "./coffee", "donuts", NULL, my_env) == -1){
        fprintf(stderr,"Can't run process 0: %s\n", strerror(errno));
        return 1;
       } 
      fprintf(stderr,"Can't create order: %s\n", strerror(errno));
      return 1;
    }
    return 0;
  }
*/

 int main(int argc, char* argv[])
 {
   // ��ͬ��������Դ ��վ 
   char* feeds[] = {"http://www.cnn.com/rss/celebs.xml",
                    "http://www.rollingstone.com/rock.xml",
                    "http://eonline.com/gossip.xml"};
   int times = 3;
   char* phrase = argv[1];//��Ҫ������ �ı� 
   int i;
   for (i = 0; i < times; i++) {
     char var[255];//������������ 
     sprintf(var, "RSS_FEED=%s", feeds[i]);
     char* vars[] = {var, NULL};
     // ֻҪexecle()ִ�гɹ�һ�鱾���� ��ֹͣ�ˣ�������
     // ��Ҫ�ȸ���һ�� �ӳ��� �ӽ��� �����̼���ִ�� ������ 
     // ע�� window���� fork()����  ����ʹ�� ��Ҫ��װ  Cygwin 
     pid_t pid = fork();// �᷵��һ������ֵ �ӽ��̷���0  �����̷���һ������ ���Ƴ����� -1 
     if(pid == -1) {
		fprintf(stderr, "���ܸ��ƽ��̣�%s", strerror(errno));
		return 1; 
		} 
     if(!pid){// pid=0Ϊ �ӽ��� ���µĽ��̽��� 
	     if (execle("/usr/bin/python", "/usr/bin/python", "./rssgossip.py", NULL, vars) == -1) {
	       fprintf(stderr, "Can't run script: %s\n", strerror(errno));
	       return 1;
	     }
        }
   }
   return 0;
 }
