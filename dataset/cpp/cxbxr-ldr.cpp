// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
// ******************************************************************
// *
// *  This file is part of Cxbx-Reloaded.
// *
// *  Cxbx-Reloaded is free software; you can redistribute it
// *  and/or modify it under the terms of the GNU General Public
// *  License as published by the Free Software Foundation; either
// *  version 2 of the license, or (at your option) any later version.
// *
// *  This program is distributed in the hope that it will be useful,
// *  but WITHOUT ANY WARRANTY; without even the implied warranty of
// *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// *  GNU General Public License for more details.
// *
// *  You should have recieved a copy of the GNU General Public License
// *  along with this program; see the file COPYING.
// *  If not, write to the Free Software Foundation, Inc.,
// *  59 Temple Place - Suite 330, Bostom, MA 02111-1307, USA.
// *
// *  (c) 2017-2019 Patrick van Logchem <pvanlogchem@gmail.com>
// *  (c) 2019 ego720
// *
// *  All rights reserved
// *
// ******************************************************************

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
#include <Windows.h> // For LPTSTR, FormatMessage, GetSystemInfo, etc

#include "..\CxbxVersion.h"
#include "..\Common\AddressRanges.h"
#include "..\Common\ReserveAddressRanges.h"

// Reserve the first 128 MB MemLowVirtual address range without inflating the EXE size,
// by simply declaring an array, first thing, in global scope. It gets placed in the BSS segment,
// so it still uses space in RAM, but not the actual executable.
// This variable *MUST* be this large, for it to take up address space
// so that all other code and data in this module are placed outside of the
// maximum virtual memory range.
#define VM_PLACEHOLDER_SIZE MiB(128) // Enough to cover MemLowVirtual (Cihiro/Devkit)

// Note : In the old setup, we used #pragma section(".text"); __declspec(allocate(".text"))
// to put this variable at the exact image base address 0x00010000, but that resulted in
// an increase of the executable by 128 Mb. Since we won't return to the tiny loader code,
// and the executable doesn't contain any data that we refer too once we entered the emulation DLL,
// this will be all right. The only bit of data I transfer over, is on the stack, but the stack
// (of the initial thread) resides far outside of the reserved range, so that's all right too.
// Note : Old setup is removed in this commit, see git blame to look at src/common/ReservedMemory.h file.
// Which was used in cxbx.exe project.
unsigned char virtual_memory_placeholder[VM_PLACEHOLDER_SIZE] = { 0 }; // = { OPCODE_NOP_90 };

// Note : This executable is meant to be as tiny as humanly possible.
// The C++ runtime library is removed using https://stackoverflow.com/a/39220245/12170

// This Cxbx loader is tailored to bootstrapping an Xbox environment under Windows.
// This requires :
// * user-access to the lowest possible virtual memory address (0x00010000),
// * complete access to as much as possible of the rest 32 bit address space.
//
// This implies running it as a 32 bit process under WOW64 (64 bit Windows).
// This is done by initially relying on no other code than our own and
// kernel32.DLL (which the Windows kernel loads into all it's processes).
// Also, the linker options for this loader are specified such, that the
// executable will be loaded by Windows at the pre-defined address, and
// won't contain any other code not under our control.
// 
// When executed, this loader starts by validating the correct run environment,
// then reserves all memory ranges that Xbox emulation requires, and only then
// loads in the actual emulation code by dynamically loading in our library
// and transfer control to it.
//
// Note, that since the emulation code will have to overwrite the memory where
// this loader resides, no code or data may be used by the emulation code,
// nor may the emulator ever return to this code!

// Important linker flags :
// /MACHINE:X86
// /NODEFAULTLIB
// /DYNAMICBASE:NO
// /BASE:"0x00010000"
// /FIXED
// /LARGEADDRESSAWARE
// /SUBSYSTEM:CONSOLE
// /ENTRY:"rawMain"


void OutputMessage(const char* msg)
{
	if (msg != nullptr) {
		OutputDebugStringA(msg); // Send message to debugger output too

		HANDLE hConsoleOutput = GetStdHandle(STD_OUTPUT_HANDLE);
		DWORD nNumberOfCharsToWrite = 0;
		while (msg[nNumberOfCharsToWrite]) nNumberOfCharsToWrite++; // poor-man's strlen()

		// Detect output-redirection to a file
		DWORD ConsoleMode;
		if (!GetConsoleMode(hConsoleOutput, &ConsoleMode)) {
			// Note : assume the output file accepts ANSI encoded characters
			DWORD NumberOfBytesWritten;
			WriteFile(hConsoleOutput, (const void*)msg, nNumberOfCharsToWrite, &NumberOfBytesWritten, /*lpOverlapped=*/NULL);
		}
		else {
			// Write message to console output
			DWORD NumberOfCharsWritten;
			WriteConsoleA(hConsoleOutput, (const void*)msg, nNumberOfCharsToWrite, &NumberOfCharsWritten, /*lpReserved=*/NULL);
		}
	}
}

LPTSTR GetLastErrorString()
{
	DWORD err = GetLastError();

	// Translate ErrorCode to String.
	LPTSTR Error = nullptr;
	if (::FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
		NULL,
		err,
		0,
		(LPTSTR)&Error,
		0,
		NULL) == 0) {
		// Failed in translating.
	}

	return Error;
}

void FreeLastErrorString(LPTSTR Error)
{
	if (Error) {
		::LocalFree(Error);
		Error = nullptr;
	}
}

#define EMULATION_DLL "cxbxr-emu.dll"

DWORD CALLBACK rawMain()
{
	(void)virtual_memory_placeholder; // prevent optimization removing this data

	// First detect if we are running on WoW64, if not, prevent Cxbx-Reloaded from starting
	// Cxbx-Reloaded needs access to high memory, only exposed to WoW64.
	if (!VerifyWow64()) {
		OutputMessage("Cxbx-Reloaded can only run under WoW64\nThis means either a 64-bit version of Windows or Wine with a 64-bit prefix\n");
		return ERROR_BAD_ENVIRONMENT;
	}

	// We have hard-code BLOCK_SIZE to 64 KiB, check this against the system's allocation granularity.
	SYSTEM_INFO SystemInfo;

	GetSystemInfo(&SystemInfo);
	if (SystemInfo.dwAllocationGranularity != BLOCK_SIZE) {
		OutputMessage("Unsupported system allocation granularity!\n");
		return ERROR_BAD_ENVIRONMENT;
	}

	if (SystemInfo.dwPageSize != PAGE_SIZE) {
		OutputMessage("Unsupported system page size!\n");
		return ERROR_BAD_ENVIRONMENT;
	}

	if (SystemInfo.lpMaximumApplicationAddress < (void*)0xFFFEFFFF) {
		// Note : If this fails, the large-address-aware linker flag must be restored
		OutputMessage("Maximum application address too low!\n");
		return ERROR_BAD_ENVIRONMENT;
	}

	// Reserve all memory by default. This is a requirement for standalone emulation support.
	// Emulator's initial process will decide which memory ranges will be free.
	unsigned int system = SYSTEM_ALL; // Reserve all systems.

	// Marking this as static to avoid an implicit call to memset, which is not available in the loader
	static blocks_reserved_t blocks_reserved;

	if (!AttemptReserveAddressRanges(&system, blocks_reserved)) {
		// If we get here, emulation lacks important address ranges; Don't launch
		OutputMessage("None of system types' required address range(s) could be reserved!\n");
		return ERROR_NOT_ENOUGH_MEMORY;
	}

	// Only after the required memory ranges are reserved, load our emulation DLL
	HMODULE hEmulationDLL = LoadLibrary(TEXT(EMULATION_DLL));
	if (!hEmulationDLL) {
		OutputMessage("Error loading " EMULATION_DLL "\n");
		LPTSTR Error = GetLastErrorString();
		if (Error) {
			OutputMessage(Error);
			FreeLastErrorString(Error);
		}

		return ERROR_RESOURCE_NOT_FOUND;
	}

	// We cannot just pass the git version of the loader via the Emulate function. This, because if the user mixes a version which does the check (and thus expects 3 arguments)
	// with an old version which doesn't do the check (and thus only has 2 arguments), the behavior will be undefined since the new version will attempt to use a
	// non-existent argument. We instead pass the version string in the contiguous memory, which must have been successfully reserved by now or else the loader would
	// have already aborted execution. This memory is backed by the paging file, and thus its contents will always be initialized to zero. Thus, in the above scenerio
	// the check will fail because a version string cannot be zero.
	// NOTE1: the loader doesn't link against the CRT, which means we cannot just use strncpy here, and thus we use a for loop instead
	// NOTE2: we choose 0x80001000 as address because the first page is used by d3d to initialize the push buffer of the nv2a, so we avoid to write a string to it
	for (unsigned i = 0; i < GetGitVersionLength(); ++i) {
		*(reinterpret_cast<char *>(PHYSICAL_MAP1_BASE + 0x1000) + i) = GetGitVersionStr()[i];
	}

	// Find the main emulation function in our DLL
	typedef void (WINAPI *Emulate_t)(unsigned int, blocks_reserved_t);
	Emulate_t pfnEmulate = (Emulate_t)GetProcAddress(hEmulationDLL, "Emulate");
	if (!pfnEmulate) {
		OutputMessage("Entrypoint not found!\n");
		return ERROR_RESOURCE_NOT_FOUND;
	}

	// Call the main emulation function in our DLL, passing in the results
	// of the address range reservations
	pfnEmulate(system, blocks_reserved);

	// Once emulation actually started, execution may never return here
	// because all code and data that have been used up until now are
	// overwritten. From then on, only emulated code is allowed to access
	// this memory! If we *did* return here, initialization failed
	OutputMessage("Initialization failure!\n");
	return ERROR_APP_DATA_CORRUPT;
}
