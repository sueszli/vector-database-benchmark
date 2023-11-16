#include "stdafx.h"
#include "Xdefine.h"

static char *glStrLogFilePath;
static CCriticalSection *glPCrtAddReport;

void ResetReport (char* strLogFileName)
{	
#ifdef __PRINT_LOG_DATA__
	glStrLogFilePath = new char[MAX_PATH];
	glPCrtAddReport = new CCriticalSection;

	strcpy_s (glStrLogFilePath, MAX_PATH, strLogFileName);
	FILE *fp;
	fopen_s (&fp, glStrLogFilePath, "w");
	fclose (fp);
#endif
}

void CloseReport ()
{
#ifdef __PRINT_LOG_DATA__
	delete[] glStrLogFilePath;
	delete glPCrtAddReport;
#endif
}

void AddReport (const char* sz,...)
{
#ifdef __PRINT_LOG_DATA__
	static char szOutput[400];
    va_list va;	
    va_start (va, sz);
    vsprintf_s (szOutput, 400, sz, va);      /* Format the string */
    va_end (va);

	if (strlen (glStrLogFilePath) < 1)
	{
		return;
	}

	glPCrtAddReport->Lock ();

	FILE *fp;
	fopen_s (&fp, glStrLogFilePath, "a");
	if (fp == NULL)
	{
		glPCrtAddReport->Unlock ();
		return;
	}

	SYSTEMTIME tt;	
	GetLocalTime (&tt);	
//	fprintf (fp, "%02d:%02d:%03d %s\n", tt.wMinute, tt.wSecond, tt.wMilliseconds, szOutput);
	fprintf (fp, "%s", szOutput);
	fclose (fp);

	glPCrtAddReport->Unlock ();
#endif
}