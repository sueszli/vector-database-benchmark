// APCInject.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include <Windows.h>  
#include <TlHelp32.h>  
#include<iostream>
using namespace std;

#define MAX_PATH 1024
BOOL EnableDebugPrivilege();
int InjectDllWithApc(WCHAR* DllFullPath, ULONG ProcessId);

int _tmain(int argc, _TCHAR* argv[])
{
	WCHAR DllFullPath[MAX_PATH] = {0};
	ULONG32  ulProcessID = 0;

	if (EnableDebugPrivilege() == false)
	{
		return 0;
	}
	cout << "Input Process ID:" << endl;
	cin >> ulProcessID;
	
#ifdef  _WIN64		
	wcsncat_s(DllFullPath, L"E:\\Dll64.dll", 20);
#else				
	wcsncat_s(DllFullPath, L"E:\\Dll.dll", 20);
#endif
	InjectDllWithApc(DllFullPath, ulProcessID);
	return 0;
}


int InjectDllWithApc(WCHAR* DllFullPath, ULONG ProcessId)
{
	HANDLE hTatgetProcessHandle;
	hTatgetProcessHandle = OpenProcess(PROCESS_ALL_ACCESS, FALSE, ProcessId);
	if (hTatgetProcessHandle == NULL)
	{
		printf("Failed To Open Process!!\n");
		return 0;
	}
	ULONG32 ulDllLength = (ULONG32)_tcslen(DllFullPath) + 1;
	//�����ڴ�  
	WCHAR* pRemoteAddress = (WCHAR*)VirtualAllocEx(hTatgetProcessHandle, NULL, ulDllLength * sizeof(WCHAR),
		MEM_COMMIT, PAGE_READWRITE);
	if (pRemoteAddress == NULL)
	{
		printf("Alloc Virtual Address Failed!\n");
		CloseHandle(hTatgetProcessHandle);
		return 0;
	}
	//DLLд�� 
	if (WriteProcessMemory(hTatgetProcessHandle, pRemoteAddress, (LPVOID)DllFullPath, ulDllLength * sizeof(WCHAR), NULL) == FALSE)
	{
		VirtualFreeEx(hTatgetProcessHandle, pRemoteAddress, ulDllLength, MEM_DECOMMIT);
		CloseHandle(hTatgetProcessHandle);
		return 0;
	}
	THREADENTRY32 ThreadEntry32 = { 0 };
	HANDLE hThreadSnap = INVALID_HANDLE_VALUE;
	ThreadEntry32.dwSize = sizeof(THREADENTRY32);
	HANDLE hThreadHandle;
	BOOL bStatus;
	DWORD dwReturn;
	//��������
	hThreadSnap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (hThreadSnap == INVALID_HANDLE_VALUE)
	{
		return 0;
	}
	if (!Thread32First(hThreadSnap, &ThreadEntry32))
	{
		CloseHandle(hThreadSnap);
		return 1;
	}
	do
	{
		//�����߳�  
		if (ThreadEntry32.th32OwnerProcessID == ProcessId)
		{
			printf("TID:%d\n", ThreadEntry32.th32ThreadID);
			hThreadHandle = OpenThread(THREAD_ALL_ACCESS, FALSE, ThreadEntry32.th32ThreadID);
			if (hThreadHandle)
			{
				//���̲߳���APC  
				dwReturn = QueueUserAPC(
					(PAPCFUNC)LoadLibrary,
					hThreadHandle,
					(ULONG_PTR)pRemoteAddress);
				if (dwReturn > 0)
				{
					bStatus = TRUE;
				}
				//�رվ��  
				CloseHandle(hThreadHandle);
			}
		}
	} while (Thread32Next(hThreadSnap, &ThreadEntry32));
	VirtualFreeEx(hTatgetProcessHandle, pRemoteAddress, ulDllLength, MEM_DECOMMIT);
	CloseHandle(hThreadSnap);
	CloseHandle(hTatgetProcessHandle);
	return 0;
}

BOOL EnableDebugPrivilege()
{

	HANDLE TokenHandle = NULL;
	TOKEN_PRIVILEGES TokenPrivilege;
	LUID uID;

	//��Ȩ������
	if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &TokenHandle))
		//Open  �鿴 Set(10)  Close     [�ŷ�][10]   [����][20]   
	{
		return FALSE;
	}

	if (!LookupPrivilegeValue(NULL, SE_DEBUG_NAME, &uID))
	{

		CloseHandle(TokenHandle);
		TokenHandle = INVALID_HANDLE_VALUE;
		return FALSE;
	}

	TokenPrivilege.PrivilegeCount = 1;
	TokenPrivilege.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
	TokenPrivilege.Privileges[0].Luid = uID;

	//���������ǽ��е���Ȩ��
	if (!AdjustTokenPrivileges(TokenHandle, FALSE, &TokenPrivilege, sizeof(TOKEN_PRIVILEGES), NULL, NULL))
	{
		CloseHandle(TokenHandle);
		TokenHandle = INVALID_HANDLE_VALUE;
		return  FALSE;
	}

	CloseHandle(TokenHandle);
	TokenHandle = INVALID_HANDLE_VALUE;
	return TRUE;

}
