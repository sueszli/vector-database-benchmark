/*------------------------------------
project �� ��ŵ����ʾ����� 
language�� C����
author��   404name-CTGU_LLZ
-------------------------------------*/ 
#include<stdio.h>
#include<windows.h>
/*������һЩ��ͼ��ʾ�� 
* @sleepTime: �ȴ�ʱ�� 
* @autoPlay���Ƿ��Զ����� 
* @wall: ǽ 
* @pillar������ 
* @hanoiLeft����ŵ��Բ�������ʾ 
* @hanoiRight����ŵ��Բ���ұ���ʾ 
* @hanoiAir����ŵ��Բ���м��οղ�����ʾ 
*/  
const int sleepTime = 5; 
const int autoPlay = 1; 
const char wall = '#';  
const char pillar = '|'; 
const char hanoiLeft = '['; 
const char hanoiRight = ']'; 
const char hanoiAir = '-'; 
const int up = 0,down = 1,left = 2,right = 3; 
/*�������ƶ��� 
* @next_go ��һ���ƶ����� //��0��1��2��3��
*/ 
int next_go[4][2] = {{0,-1},{0,1},{-1,0},{1,0}}; 

/*��ϵͳ������ 
* @abc_pillar[3][1000] ��ǰ���ӷ��õ�Բ����
					   [i][0]��ŵ�i�����ӵ�ǰ�߶� 
*                      [i][j]��ʾ��i�����ӵ�j���ŵ�Բ�̴�С 
* @abc_x[3] abc���ӵ����� 
* 		    �������ֱ��ͨ�� ABC - 'A' ֱ�ӻ�ȡ�±�ȡֵ 
*/  
int N;
int abc_pillar[3][1000] = {{0},{0},{0}};
int abc_x[3] = {0,0,0}; 
int mapHeight,mapWidth; 
int deep;

// ָ�������ת��x��y�����괦
// ˮƽx����ֱy 
void gotoxy(int x,int y)  
{
    HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);
    COORD pos;
    pos.X = x;
    pos.Y = y;
    SetConsoleCursorPosition(handle,pos);
} 
//�������ع�꺯��
void HideCursor()
{
	CONSOLE_CURSOR_INFO cursor;    
	cursor.bVisible = FALSE;    
	cursor.dwSize = sizeof(cursor);    
	HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);    
	SetConsoleCursorInfo(handle, &cursor);
}

// ���x��y�ط��Ļ���
// ���һ�����һ�� ����Ϊn �ĺ�ŵ�� 
void drawHanoi(int &x,int &y,int n,int next){
	// ���ԭ���� 
	char replace[2*n+1] = ""; 
	for(int i = 0; i < 2*n + 1; i++){
		replace[i] = ' ';
		if(i == n && y != 1){
			replace[i] = pillar;
		}
		if(i == 2*n){
			replace[i+1] = '\0';
		}
	}
	gotoxy(x - n,y);
	printf("%s",replace); 
	// �����µ� 
	if(next != -1){
		x += next_go[next][0];
		y += next_go[next][1];	
	}
	for(int i = 0; i < 2*n + 1; i++){
		if(i == n && y != 1){
			replace[i] = pillar;
		}else if(i == 0){
			replace[i] = hanoiLeft;
		}else if(i == 2*n){
			replace[i] = hanoiRight;
		}else{
			replace[i] = hanoiAir;
		}
		if(i == 2*n){
			replace[i+1] = '\0';
		}
	}
	gotoxy(x - n,y);
	printf("%s",replace); 
	Sleep(sleepTime);
}

void init(){
	// ��ʼ��������Ϣ 
	abc_pillar[0][0] = N;
	for(int i = 1; i <= N; i++ ){
		abc_pillar[0][i] = N + 1 - i;
	}
	//�߶� = ����ǽ(2) + deep(4) + �Ͽ���(1) 
	//��� = ����ǽ(2) + 3������[3*(deep*2+1)] + �����м����(2)
	mapHeight = 2 + N + 1;
	mapWidth = 2 + 3 * (N * 2 + 1) + 2;
	
	//����0(a)ˮƽ���� = ���ǽ(1) + deep(4) 
	//����1(b)ˮƽ���� = ����1 + 2*deep + 2 
	//����2(c)ˮƽ���� = ����2 + 2*deep + 2 
	abc_x[0] = 1 + N;
	abc_x[1] = abc_x[0] + 2* N + 2;
	abc_x[2] = abc_x[1] + 2* N + 2;
	
	// ���Ƶ�ͼ
	for(int i = 0; i < mapHeight; i++){
		for(int j = 0; j < mapWidth; j++){
			//ǽ����� 
			if(i == 0 || i == mapHeight - 1 || j == 0 || j == mapWidth - 1){
				gotoxy(j,i);
				printf("%c",wall);
			}
			//��������
			else if( i > 1 && i < mapHeight - 1){
				if(j == abc_x[0] || j == abc_x[1] || j == abc_x[2]){
					//��ʼ������Բ��
					int abc_x_index = 0;
					if(j == abc_x[0]) abc_x_index = 0;
					else if(j == abc_x[1]) abc_x_index = 1;
					else if(j == abc_x[2]) abc_x_index = 2;
					drawHanoi(j,i,abc_pillar[abc_x_index][N - i + 2],-1);
				}
			}
		}
	} 
}
// a ==> c 
// startx starty == > toX toY
void move(char from,char to){
	gotoxy(0,mapHeight+1); 
	printf("%c--->%c\n",from,to); 
	// ��ȡfrom�������ж��ٸ߶�λ���Ŀ�ʼ 
	int fromHeight = abc_pillar[from-'A'][0];
	int n = abc_pillar[from-'A'][fromHeight];
	int x = abc_x[from-'A'];
	int y = 2 + N - fromHeight;
	abc_pillar[from-'A'][0]--;
	// ��ȡto �������ж�� ��λ���Ľ��� 
	abc_pillar[to-'A'][0]++;
	int toHeight = abc_pillar[to-'A'][0];
	abc_pillar[to-'A'][toHeight] = n;
	int toX = abc_x[to-'A'];
	int toY = 2 + N - toHeight;
	// ȡ���� ==> ���������� 
	while(y > 1){
		drawHanoi(x,y,n,up);
	}
	// �ƶ��� ==> �ƶ���ָ���� 
	if(x < toX){
		while(x < toX){
			drawHanoi(x,y,n,right);
		}
	}else if(x > toX){
		while(x > toX){
			drawHanoi(x,y,n,left);
		}
	}
	// ������ ==> �½������� 
	while(y < toY){
		drawHanoi(x,y,n,down);
	}
	//�ֶ����� 
	if(autoPlay == 0){
		gotoxy(0,mapHeight+2); 
		printf("�س�������һ��"); 
		getchar(); 
	} 
}

/**
* from ���Ŀ�ʼ 
* temp ��ʱ���� 
* to   �ƶ����� 
*/
void hanoi(int n,char from ,char temp,char to){
	if(n==1){           // from��toֻ��һ����Ҫ�ƶ���ֱ���ƶ� 
		move(from, to); //�ݹ��ֹ����
	}
	else{
		hanoi(n-1,from ,to,temp); //�ƶ�from �����n-1 ==> temp �ݴ� 
		move(from,to);            //�ƶ�from ʣ�µ�1   ==> to   Ŀ�ĵ� 
		hanoi(n-1,temp,from,to);  //�ƶ�temp �ݴ��n-1 ==> to   Ŀ�ĵ� 
	}
}

int main()
{
	HideCursor();
	printf("��Ҫ���Ĳ���");
	scanf("%d",&N);
	init();
	getchar(); 
	HideCursor();
	hanoi(N,'A','B','C');
	getchar();
	getchar();
	return 0;
} 
