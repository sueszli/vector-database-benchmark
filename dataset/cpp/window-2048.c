#include <windows.h>
#include <conio.h>
#include <time.h>
#define Std2048 0 //�����ͼΪ��׼��4*4��С��������Ӧ��Ļ��С
#if Std2048
#define windowWidth (GetSystemMetrics(SM_CXSCREEN) / mapCols - 10)
#define windowHeight (GetSystemMetrics(SM_CYSCREEN) / mapLines - 10) //�������ڳ��Ϳ�
#define mapLines 4 
#define mapCols 4  //��ͼ����������
#else
#define windowWidth 180
#define windowHeight windowWidth //�������ڳ��Ϳ�
#define mapLines (GetSystemMetrics(SM_CYSCREEN) / windowWidth)
#define mapCols (GetSystemMetrics(SM_CXSCREEN) / windowHeight) //��ͼ����������
#endif
#define mapMax (mapLines * mapCols)//��ͼ�ܹ��ĸ�����
#define WindowStyle WS_OVERLAPPEDWINDOW //WS_POPUP
const char* myclass = "2048";
int *map, score = 0, game = 1, c, i, j, k;
HANDLE hOut;
COORD origin = { 0, 0 };
HWND *screen;
MSG msg;
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
//�ƶ�
void move(int i, int v) {
	if (i + v < 0 || mapMax - 1 < i + v || !map[i])return;
	if ((v == 1 || v == -1) && i / mapCols - (i + v) / mapCols)return;
	if (map[i + v] == map[i])map[i + v] *= -2, score += map[i], map[i] = k = 0;
	if (!map[i + v])map[i + v] = map[i], map[i] = k = 0, move(i + v, v);
}
//˳��
void order(int end, int i, int v) {
	for (int begin = end == mapMax ? 0 : mapMax - 1; begin - end; begin += i)move(begin, v);
}

int main()
{
	srand((unsigned)time(0)); //��ʼ�����������
	if (!MyRegisterClass())return GetLastError();//ע�ᴰ��
	hOut = GetStdHandle(STD_OUTPUT_HANDLE);
	setConsoleSize(mapCols * 6 + 1, mapLines * 2 + 1);
	hideCursor();
	map = (int*)malloc(mapMax * sizeof(int));
	screen = (HWND*)malloc(mapMax * sizeof(HWND)); //����
	LOGFONT logfont; //�ı��������
	ZeroMemory(&logfont, sizeof(LOGFONT));
	logfont.lfCharSet = GB2312_CHARSET;
	logfont.lfHeight = 80;
	HFONT hFont = CreateFontIndirect(&logfont);
	for (i = 0; i < mapMax; i++) //��ʼ��
	{
		map[i] = 0;
		screen[i] = addWindow(i % mapCols * windowWidth, i / mapCols * windowHeight, windowWidth, windowHeight);
	}
	for (c = j = 0; game; j = k = 1, c = _getch())
	{
		if (c == 'a' || c == 'w')order(mapMax, 1, c == 'a' ? -1 : -mapCols);
		else if (c == 'd' || c == 's')order(-1, -1, c == 'd' ? 1 : mapCols);
		else if (c == ' ')break;
		for (i = 0; i < mapMax; i++)
		{
			if (map[i] < 0)map[i] = -map[i];
			if (!map[i])j = 0;
		}
		if (!j && !k)
		{
			do i = rand() % mapMax;
			while (map[i]);
			map[i] = rand() % 5 ? 2 : 4, 0;
		}
		for (i = 0; j && (i < mapMax || (game = 0)); i++)
			if (i < mapMax - mapCols && map[i] == map[i + mapCols] ||
				(i + 1) % mapCols && map[i] == map[i + 1])break;
		SetConsoleCursorPosition(hOut, origin);
		if (game || _cprintf("Game over!"))_cprintf("score:%d", score);
		for (i = 0; i < mapMax; i++)
		{
			if (i % mapCols == 0)
			{
				_cputs("\n");
				for (j = 0; j < mapCols; j++)
					_cputs("----- ");
				_cputs("\n");
			}
			else _cputs("|");
			if (map[i])_cprintf("%5d", map[i]);
			else _cputs("     ");
			if (map[i])
			{
				ShowWindow(screen[i], SW_SHOWDEFAULT);
				static PAINTSTRUCT ps;
				static RECT rect;
				static HDC hdc;
				GetClientRect(screen[i], &rect);
				char str[6] = "00000";
				int len = 1, temp = map[i];
				while ((temp /= 10) > 0)len++;
				temp = map[i];
				for (int i = 0; i < len; temp /= 10, i++)
					str[len - i - 1] = temp % 10 + 48;
				str[len] = 0;
				hdc = GetDC(screen[i]);
				SelectObject(hdc, hFont);
				Rectangle(hdc, -1, -1, windowWidth, windowHeight);
				DrawText(hdc, str, -1, &rect, DT_SINGLELINE | DT_CENTER | DT_VCENTER);
				ReleaseDC(screen[i], hdc);
				UpdateWindow(screen[i]);
			}
			else ShowWindow(screen[i], SW_HIDE);
		}
	}
	Sleep(1000);
	DeleteObject(hFont);
	free(map);
	free(screen);
}
