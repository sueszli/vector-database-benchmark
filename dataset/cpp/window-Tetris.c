#include <windows.h>
#include <stdio.h>
#include <conio.h>
#include <time.h>
#define windowWidth 120
#define windowHeight windowWidth //�������ڳ��Ϳ�
#define mapLines (GetSystemMetrics(SM_CYSCREEN) / windowWidth)
#define mapCols (GetSystemMetrics(SM_CXSCREEN) / windowHeight) //��ͼ����������
#define mapMax (mapLines * mapCols)//��ͼ�ܹ��ĸ�����
#if (windowWidth < 120)
#define WindowStyle WS_POPUP
#else
#define WindowStyle WS_OVERLAPPEDWINDOW
#endif
const char* myclass = "Trtris";
int T, X, Y, c = 0, i, j, k, oldTime = 0, *map, node[28][4][2] = {
-1,0,0,0,1,0,-1,-1,  0,1,0,0,0,-1,1,-1,  1,1,-1,0,0,0,1,0,  -1,1,0,1,0,0,0,-1,  -1,0,0,0,1,0,1,-1,  0,1,1,1,0,0,0,-1,  -1,1,-1,0,0,0,1,0,
0,1,0,0,0,-1,-1,-1,  -1,1,0,1,0,0,1,0,  1,1,0,0,1,0,0,-1,  -1,1,0,1,0,0,1,0,  1,1,0,0,1,0,0,-1,  0,1,1,1,-1,0,0,0,  0,1,0,0,1,0,1,-1,
0,1,1,1,-1,0,0,0,  0,1,0,0,1,0,1,-1,  -1,0,0,0,1,0,0,-1,  0,1,0,0,1,0,0,-1,  0,1,-1,0,0,0,1,0,  0,1,-1,0,0,0,0,-1, 0,2,0,1,0,0,0,-1,
-1,0,0,0,1,0,2,0,  0,2,0,1,0,0,0,-1,  -1,0,0,0,1,0,2,0,  0,1,1,1,0,0,1,0,  0,1,1,1,0,0,1,0,  0,1,1,1,0,0,1,0,  0,1,1,1,0,0,1,0
};
HANDLE hOut;
COORD origin = { 0, 0 };
HWND *screen;
MSG msg;
//�ƶ�
int move(int* v, int l)
{
	for (*v += l, i = 0; i < 4; i++)
	{
		if (X + node[T][i][0] > mapCols - 1)break;
		if (X + node[T][i][0] < 0)break;
		if (Y + node[T][i][1] > mapLines - 1)break;
		if (map[(node[T][i][1] + Y) * mapCols + node[T][i][0] + X])break;
	}
	if (i == 4)return 1;
	*v -= l;
	return 0;
}
//����
void down()
{
	if (move(&Y, 1))return;
	if (Y < 2)_getch(), exit(0);
	for (i = 0; i < 4; i++)
		map[(node[T][i][1] + Y) * mapCols + node[T][i][0] + X] = 1;
	X = mapCols / 2, Y = 1, T = rand() % 7 * 4;
	for (i = mapMax - mapCols; i >= mapCols; i -= mapCols)
	{
		for (j = k = 0; j < mapCols; j++)
			k += map[i + j];
		if (k - mapCols)continue;
		for (j = i + mapCols - 1; j >= mapCols; j--)
			map[j] = map[j - mapCols];
		i += mapCols;
	}
}
//��Ϣѭ��
LRESULT CALLBACK WndProc(HWND window, unsigned int msg, WPARAM wp, LPARAM lp)
{
	return DefWindowProc(window, msg, wp, lp);
}
//ע�ᴰ��
ATOM MyRegisterClass()
{
	WNDCLASSEX wc;
	wc.cbSize = sizeof(WNDCLASSEX);
	wc.style = CS_DBLCLKS;
	wc.lpfnWndProc = WndProc;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = GetModuleHandle(0);
	wc.hIcon = LoadIcon(0, IDI_APPLICATION);
	wc.hCursor = LoadCursor(0, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wc.lpszMenuName = 0;
	wc.lpszClassName = myclass;
	wc.hIconSm = LoadIcon(0, IDI_APPLICATION);
	return RegisterClassEx(&wc);
}
//��������
HWND addWindow(int x, int y, int width, int height)
{
	HWND window = CreateWindowEx(0, myclass, "title", WindowStyle,
		x, y, width, height, 0, 0, GetModuleHandle(0), 0);//��������
	if (!window)return 0;//�����Ƿ�ɹ�����
	//ShowWindow(window, SW_SHOWDEFAULT);//��ʾ����
	return window;
}
//���ÿ���̨��С
void setConsoleSize(int lines, int cols)
{
	COORD size = { lines,cols };
	SMALL_RECT rect = { 0,0,size.X - 1,size.Y - 1 };
	SetConsoleWindowInfo(hOut, 1, &rect);
	SetConsoleScreenBufferSize(hOut, size);
	SetConsoleWindowInfo(hOut, 1, &rect);
}
//���ع��
void hideCursor()
{
	CONSOLE_CURSOR_INFO CursorInfo;
	GetConsoleCursorInfo(hOut, &CursorInfo);
	CursorInfo.bVisible = 0;
	SetConsoleCursorInfo(hOut, &CursorInfo);
}

int main()
{
	srand((unsigned)time(0)); //��ʼ�����������
	if (!MyRegisterClass())return GetLastError();//ע�ᴰ��
	hOut = GetStdHandle(STD_OUTPUT_HANDLE);
	setConsoleSize(mapCols * 2, mapLines + 1);
	hideCursor();
	map = (int*)malloc(mapMax * sizeof(int));
	screen = (HWND*)malloc(mapMax * sizeof(HWND)); //����
	for (i = 0; i < mapMax; i++) //��ʼ��
	{
		map[i] = 0;
		screen[i] = addWindow(i % mapCols * windowWidth, i / mapCols * windowHeight, windowWidth, windowHeight);
	}
	X = mapCols / 2, Y = 1, T = rand() % 7 * 4;
	while (1)
	{
		for (i = 0; i < 4; i++)
			map[(node[T][i][1] + Y) * mapCols + node[T][i][0] + X] = 1;
		SetConsoleCursorPosition(hOut,origin);
		for (i = 0; i < mapMax; i++)
		{
			printf(map[i] ? "��" : "  ");
			if (map[i])ShowWindow(screen[i], SW_SHOWDEFAULT);
			else ShowWindow(screen[i], SW_HIDE);
		}
		for (i = 0; i < 4; i++)
			map[(node[T][i][1] + Y) * mapCols + node[T][i][0] + X] = 0;
		if (GetAsyncKeyState('W') & 0x8000)move(&T, (T % 4) < 3 ? 1 : -3);
		else if (GetAsyncKeyState('A') & 0x8000)move(&X, -1);
		else if (GetAsyncKeyState('D') & 0x8000)move(&X, 1);
		else if (GetAsyncKeyState('S') & 0x8000)down();
		else if (GetAsyncKeyState(' ') & 0x8000)break;
		if (clock() - 1000 > oldTime)oldTime = clock(), down();
		Sleep(100);
	}
	Sleep(1000);
	free(map);
	free(screen);
}
