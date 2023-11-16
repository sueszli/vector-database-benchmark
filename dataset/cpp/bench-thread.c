#include <stdio.h>
#include <windows.h>

#define NR_THREADS (30)


DWORD WINAPI
thread_main1(LPVOID param)
{
   printf("Thread 1 running (Counter %lu)\n", PtrToUlong(param));
   SleepEx(INFINITE, TRUE);
   return 0;
}


DWORD WINAPI
thread_main2(LPVOID param)
{
   printf("Thread 2 running (Counter %lu)\n", PtrToUlong(param));
   Sleep(INFINITE);
   return 0;
}


int main (void)
{
   DWORD i=0;
   DWORD id;

#if 1
   printf("Creating %d threads...\n",NR_THREADS*2);
   for (i=0;i<NR_THREADS;i++)
     {
	CreateThread(NULL,
		     0,
		     thread_main1,
		     (LPVOID)(ULONG_PTR)i,
		     0,
		     &id);

/*	CreateThread(NULL,
		     0,
		     thread_main2,
		     (LPVOID)i,
		     0,
		     &id);*/
     }

   printf("All threads created...\n");

   /*
    * Waiting for threads is not implemented yet.
    * If you want to see all threads running, uncomment the
    * call to SuspendThread(). The test application will
    * freeze after all threads are created.
    */
/*   SuspendThread (GetCurrentThread()); */

#else

   printf("Creating thread...\n");

   hThread = CreateThread(NULL,
                          0,
                          thread_main1,
                          (LPVOID)i,
                          0,
                          &id);

   printf("Thread created. Waiting for termination...\n");

   WaitForSingleObject (hThread,
                        -1);

   CloseHandle (hThread);

   printf("Thread terminated...\n");
#endif
   printf("Exiting\n");
   return 0;
}
