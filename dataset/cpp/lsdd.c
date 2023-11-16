/*
 * FILE  : lsdd.c
 * AUTHOR: Emanuele ALIBERTI
 * DATE  : 2000-08-04
 * DESC  : List DOS devices, i.e. symbolic links created
 *         in the \?? object manager's name space.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include <windef.h>
#include <winbase.h>


#include <reactos/buildno.h>

#include "../win32err.h"

#define LINKS_SIZE 32768
#define DEVICE_SIZE 8192

static const LPWSTR error_prefix = L"lsdd: ";

static char SymbolicLinks [LINKS_SIZE];
static char DosDeviceName [DEVICE_SIZE];

static char DeviceNames [DEVICE_SIZE];
static char DeviceName [DEVICE_SIZE];


BOOL
WINAPI
GetNextString (
	char *	BufferIn,
	char *	BufferOut,
	char ** Next
	)
{
	char * next = *Next;
	char * w;

	if ('\0' == *next) return FALSE;
	for (	w = BufferOut;
		('\0' != *next);
		next ++
		)
	{
		*(w ++) = *next;
	}
	*w = '\0';
	*Next = ++ next;
	return TRUE;
}


int
main (int argc, char * argv [] )
{
	printf (
		"ReactOS/Win32 %s - List DOS Devices Utility\n"
		"Written by E.Aliberti (%s)\n\n",
		KERNEL_VERSION_STR,
		__DATE__
		);

	if (0 != QueryDosDevice (
			NULL, /* dump full directory */
			SymbolicLinks,
			sizeof SymbolicLinks
			)
		)
	{
		char * NextDosDevice = SymbolicLinks;
		char * NextDevice;

		while (TRUE == GetNextString (
					SymbolicLinks,
					DosDeviceName,
					& NextDosDevice
					)
				)
		{
			printf ("%s\n", DosDeviceName);
			if (0 != QueryDosDevice (
					DosDeviceName,
					DeviceNames,
					sizeof DeviceNames
					)
				)
			{
				NextDevice = DeviceNames;
				while (TRUE == GetNextString (
							DeviceNames,
							DeviceName,
							& NextDevice
						)
					)
				{
					printf ("  -> \"%s\"\n", DeviceName);
				}
			}
			else
			{
				PrintWin32Error (
					error_prefix,
					GetLastError ()
					);
			}
			printf ("\n");
		}
	}
	else
	{
		PrintWin32Error (
			error_prefix,
			GetLastError ()
			);
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}


/* EOF */
