/*
 * Copyright (c) 1999, 2005, 2010 Greg Haerr <greg@censoft.com>
 *
 * Microwindows Terminal Emulator for Linux
 *
 * Yes, this is just a demo, and doesn't repaint contents on refresh.
 */

#define _XOPEN_SOURCE 600	/* for ptsname()*/
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#define MWINCLUDECOLORS
#include "windows.h"
#include "wintools.h"		/* Draw3dInset*/

#define UNIX98		1		/* use new-style /dev/ptmx, /dev/pts/0 ptys*/

#define COLS		80
#define ROWS		24
#define XMARGIN		2
#define YMARGIN		2
#define FGCOLOR		GREEN
#define BKCOLOR		BLACK
#define FONTNAME	SYSTEM_FIXED_FONT
/*#define FONTNAME	OEM_FIXED_FONT*/
#define APPCLASS	"mterm"

#if ELKS
#define SHELL	"/bin/sash"
#elif DOS_DJGPP
#define SHELL	"bash"
#define killpg		kill
#define SIGCHLD		17 /* from Linux, not defined in DJGPP */
#elif defined(__CYGWIN__)
#include <pty.h>
#define SHELL  "/usr/bin/bash"
#else
#define SHELL	"/bin/sh"
#endif

/* forward decls*/
LRESULT CALLBACK WndProc(HWND hwnd,UINT uMsg,WPARAM wp,LPARAM lp);
void EmOutChar(HWND hwnd, unsigned char ch);
int  CreatePtyShell(void);
int  ReadPtyShell(int fd, char *buf, int count);
int  WritePtyShell(int fd, char *buf, int count);
void ClosePtyShell(int fd);

/* local data*/
static int ttyfd = -1;
static int xpos = XMARGIN;
static int ypos = YMARGIN;
static int nCharWidth, nCharHeight;
static int nScreenWidth, nScreenHeight;

static int
RegisterAppClass(void)
{
	WNDCLASS	wc;

	wc.style = CS_DBLCLKS | CS_VREDRAW | CS_HREDRAW;
	wc.lpfnWndProc = (WNDPROC)WndProc;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = 0;
	wc.hIcon = 0; /*LoadIcon(GetHInstance(), MAKEINTRESOURCE( 1));*/
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = CreateSolidBrush(BKCOLOR);
	wc.lpszMenuName = NULL;
	wc.lpszClassName =  APPCLASS;
	RegisterClass( &wc);
	return 1;
}

static HWND
CreateAppWindow(void)
{
	HWND	hwnd;
	HDC	hdc;
	int 	w, h;
	RECT	rc;

	GetWindowRect(GetDesktopWindow(), &rc);
	w = rc.right - 40;
	h = rc.bottom;

	/* determine TE size from font*/
	hdc = GetDC(NULL);
	SelectObject(hdc, GetStockObject(FONTNAME));
	SetRect(&rc, 0, 0, 0, 0);
	nCharHeight = DrawText(hdc, "m", 1, &rc, DT_CALCRECT);
	nCharWidth = rc.right;
	nScreenWidth = min(w, nCharWidth*COLS);
	nScreenHeight = min(h, nCharHeight*ROWS);
	ReleaseDC(NULL, hdc);

	hwnd = CreateWindowEx(0L, APPCLASS,
		"Microwindows Terminal",
		WS_OVERLAPPEDWINDOW | WS_VISIBLE,
		CW_USEDEFAULT, CW_USEDEFAULT,
		nScreenWidth+4, nScreenHeight+24,
		NULL, (HMENU)1, NULL, NULL);

	return hwnd;
}

LRESULT CALLBACK
WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
	unsigned char	ch;
	HDC		hdc;
	RECT		rc;
	PAINTSTRUCT	ps;
   
	switch(msg) {
	case WM_CREATE:
		ttyfd = CreatePtyShell();
		/*if(ttyfd == -1)
			return -1;*/
		MwRegisterFdInput(hwnd, ttyfd);
		xpos = XMARGIN;
		ypos = YMARGIN;
		break;

	case WM_DESTROY:
		MwUnregisterFdInput(hwnd, ttyfd);
		ClosePtyShell(ttyfd);
		break;

	case WM_CHAR:
		ch = (char)wp;
		/* echo half duplex if CreatePtyShell() failed*/
		if(ttyfd == -1) {
			EmOutChar(hwnd, ch);
			if(ch == '\r')
				EmOutChar(hwnd, '\n');
		} else
			WritePtyShell(ttyfd, (char *)&ch, 1);
		break;

	case WM_FDINPUT:
		if(ReadPtyShell(ttyfd, (char *)&ch, 1) == 1)
			EmOutChar(hwnd, ch);
		break;

	case WM_PAINT:
		hdc = BeginPaint(hwnd, &ps);
		GetClientRect(hwnd, &rc);
		Draw3dInset(hdc, 0, 0, rc.right-rc.left, rc.bottom-rc.top);
		EndPaint(hwnd, &ps);
		break;

	default:
		return DefWindowProc(hwnd, msg, wp, lp);
	}
	return 0;
}

int WINAPI
WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd)
{
	MSG 	msg;
	//extern MWIMAGEHDR image_car8;
	extern MWIMAGEHDR image_dragon;

	RegisterAppClass();
#if !MULTIAPP
	//MwSetDesktopWallpaper(&image_car8);
	MwSetDesktopWallpaper(&image_dragon);
#endif
	CreateAppWindow();

#if !MULTIAPP
	while(GetMessage(&msg, NULL, 0, 0)) {
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
#endif
	return 0;
}

void
EmOutChar(HWND hwnd, unsigned char ch)
{
	HDC	hdc;
	RECT	rc;

	switch(ch) {
	case '\r':
		xpos = XMARGIN;
		return;
	case '\n':
		ypos += nCharHeight;
		GetClientRect(hwnd, &rc);
		if(ypos > (ROWS-1)*nCharHeight + YMARGIN) {
			ypos -= nCharHeight;

			/* scroll window using bitblt ;-)*/
			hdc = GetDC(hwnd);
			BitBlt(hdc, XMARGIN, YMARGIN, rc.right-XMARGIN*2,
				rc.bottom-nCharHeight-YMARGIN*2,
				hdc, XMARGIN, nCharHeight+YMARGIN, SRCCOPY);
			rc.top = ypos;
			rc.left += XMARGIN;
			rc.right -= XMARGIN;
			rc.bottom -= YMARGIN;
			FillRect(hdc, &rc,
				(HBRUSH)(ULONG_PTR)GetClassLongPtr(hwnd, GCL_HBRBACKGROUND));
			ReleaseDC(hwnd, hdc);
		}
		return;
	case '\007':			/* bel*/
		(void)write(STDERR_FILENO, "\007", 1);
		return;
	case '\t':
		xpos += nCharWidth;
		while((xpos/nCharWidth) & 7)
			EmOutChar(hwnd, ' ');
		return;
	case '\b':
		if(xpos <= XMARGIN)
			return;
		xpos -= nCharWidth;
		EmOutChar(hwnd, ' ');
		xpos -= nCharWidth;
		return;
	}

	/* draw some text*/
	hdc = GetDC(hwnd);
	SelectObject(hdc, GetStockObject(FONTNAME));
	SetBkColor(hdc, BKCOLOR);
	SetTextColor(hdc, FGCOLOR);
	SetRect(&rc, xpos, ypos, xpos+nCharWidth, ypos+nCharHeight);
	ExtTextOut(hdc, xpos, ypos, ETO_OPAQUE, &rc, (char *)&ch, 1, NULL);
	ReleaseDC(hwnd, hdc);
	xpos += nCharWidth;
	if(xpos > (COLS-1)*nCharWidth) {
		xpos = XMARGIN;
		EmOutChar(hwnd, '\n');
	}
}

static int pid;

static void
ptysignaled(int signo)
{
	switch(signo) {
	case SIGINT:	/* interrupt*/
#if !ELKS
		/* this doesn't work, can anyone fix it?*/
		killpg(pid, SIGINT);
#endif
		return;
	case SIGCHLD:	/* child status change - child exit*/
		DestroyWindow(GetActiveWindow());
		CreateAppWindow();
		return;
	}
	MwError("Uncaught signal %d\n", signo);
}

/*
 * Create a shell running through a pseudo tty, return the shell fd.
 */
#if defined(__CYGWIN__)
int
CreatePtyShell(void)
{
	int	master, slave;
	char *	argv[2];
	char	pty_name[32];

	signal(SIGCHLD, ptysignaled);
	signal(SIGINT, ptysignaled);
	if ((pid = forkpty(&master, pty_name, NULL, NULL)) == -1) {
		MwError("No processes\n");
		return -1;
	}
MwError("CYGWIN: Opened pty_name %s\n", pty_name);
	if (!pid) {
		close(STDIN_FILENO);
		close(STDOUT_FILENO);
		close(STDERR_FILENO);
		close(master);

		setsid();
		grantpt(master);
		unlockpt(master);
		if ((slave = open(pty_name, O_RDWR)) < 0) {
			MwError("Child: Can't open pty %s\n", pty_name);
			exit(1);
		}
		dup2(slave, STDIN_FILENO);
		dup2(slave, STDOUT_FILENO);
		dup2(slave, STDERR_FILENO);
		/*if(!(argv[0] = getenv("SHELL")))*/
		argv[0] = SHELL;
		argv[1] = NULL;
		execv(argv[0], argv);
		exit(1);
	}
	return master;
}
#elif (UNIX && UNIX98)
int
CreatePtyShell(void)
{
	int	tfd;
	char *	argv[2];

	char *pty_name;

	/* opens /dev/ptmx, can't use getpt() as needs nonblocking*/
	if ((tfd = open("/dev/ptmx", O_RDWR | O_NOCTTY | O_NONBLOCK)) < 0) {
err:
		MwError("Can't create pty /dev/ptmx\n");
		return -1;
	}
	signal(SIGCHLD, SIG_DFL);	/* required before grantpt()*/
        if (grantpt(tfd) || unlockpt(tfd) || !(pty_name = ptsname(tfd)))
		goto err;

	signal(SIGCHLD, ptysignaled);
	signal(SIGINT, ptysignaled);
	if ((pid = fork()) == -1) {
		MwError("No processes\n");
		return -1;
	}
	if (!pid) {
		close(STDIN_FILENO);
		close(STDOUT_FILENO);
		close(STDERR_FILENO);
		close(tfd);
		
		setsid();
		if ((tfd = open(pty_name, O_RDWR)) < 0) {
			MwError("Child: Can't open pty %s\n", pty_name);
			exit(1);
		}
		dup2(tfd, STDIN_FILENO);
		dup2(tfd, STDOUT_FILENO);
		dup2(tfd, STDERR_FILENO);
		/*if(!(argv[0] = getenv("SHELL")))*/
			argv[0] = SHELL;
		argv[1] = NULL;
		execv(argv[0], argv);
		exit(1);
	}
	return tfd;
}
#elif UNIX /* non-UNIX98 pty*/
int
CreatePtyShell(void)
{
	int	n = 0;
	int	tfd;
	char *	argv[2];
	char	pty_name[32];

again:
	sprintf(pty_name, "/dev/ptyp%d", n);
	if ((tfd = open(pty_name, O_RDWR | O_NONBLOCK)) < 0) {
		if ((errno == EBUSY || errno == EIO) && n < 10) {
			++n;
			goto again;
		}
		MwError("Can't create pty %s\n", pty_name);
		return -1;
	}
	signal(SIGCHLD, ptysignaled);
	signal(SIGINT, ptysignaled);
	if ((pid = fork()) == -1) {
		MwError("No processes\n");
		return -1;
	}
	if (!pid) {
		close(STDIN_FILENO);
		close(STDOUT_FILENO);
		close(STDERR_FILENO);
		close(tfd);
		
		setsid();
		pty_name[5] = 't';
		if ((tfd = open(pty_name, O_RDWR)) < 0) {
			MwError("Child: Can't open pty %s\n", pty_name);
			exit(1);
		}
		dup2(tfd, STDIN_FILENO);
		dup2(tfd, STDOUT_FILENO);
		dup2(tfd, STDERR_FILENO);
		/*if(!(argv[0] = getenv("SHELL")))*/
			argv[0] = SHELL;
		argv[1] = NULL;
		execv(argv[0], argv);
		exit(1);
	}
	return tfd;
}
#endif /* UNIX*/

int
ReadPtyShell(int fd, char *buf, int count)
{
	return read(fd, buf, count);
}

int
WritePtyShell(int fd, char *buf, int count)
{
	return write(fd, buf, count);
}

void
ClosePtyShell(int fd)
{
	if(ttyfd != -1)
		close(fd);
}
