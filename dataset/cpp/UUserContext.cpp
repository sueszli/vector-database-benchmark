/********************************************************************************
 * The Peacenet - bit::phoenix("software");
 * 
 * MIT License
 *
 * Copyright (c) 2018-2019 Michael VanOverbeek, Declan Hoare, Ian Clary, 
 * Trey Smith, Richard Moch, Victor Tran and Warren Harris
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * 
 * Contributors:
 *  - Michael VanOverbeek <alkaline@bitphoenixsoftware.com>
 *
 ********************************************************************************/


#include "UUserContext.h"
#include "PeacenetWorldStateActor.h"
#include "URainbowTable.h"
#include "UDesktopWidget.h"
#include "PTerminalWidget.h"
#include "UConsoleContext.h"
#include "CommonUtils.h"
#include "UProgram.h"
#include "UPeacegateFileSystem.h"

int UUserContext::StartProcess(FString Name, FString FilePath)
{
	return this->GetOwningSystem()->StartProcess(Name, FilePath, this->UserID);
}

FString UUserContext::GetProcessUsername(FPeacegateProcess InProcess)
{
	return this->GetOwningSystem()->GetProcessUsername(InProcess);
}

TArray<FPeacegateProcess> UUserContext::GetRunningProcesses()
{
	return this->GetOwningSystem()->GetRunningProcesses();
}

void UUserContext::ParseURL(FString InURL, int InDefaultPort, FString& OutUsername, FString& OutHost, int& OutPort, FString& OutPath)
{
	bool HasPort = false;
	bool HasUser = false;
	bool HasPath = false;

	UCommonUtils::ParseURL(InURL, OutUsername, OutHost, OutPort, OutPath, HasPath, HasUser, HasPort);

	if(!HasPort)
	{
		OutPort = InDefaultPort;
	}

	if(!HasUser)
	{
		OutUsername = this->GetUsername();
	}
}

FUserInfo UUserContext::GetUserInfo()
{
    return this->GetOwningSystem()->GetUserInfo(this->UserID);
}

void UUserContext::Setup(USystemContext* InOwningSystem, int InUserID)
{
    // Make sure the owning system is valid.
    check(InOwningSystem);

    this->OwningSystem = InOwningSystem;
    this->UserID = InUserID;
}

FString UUserContext::GetHostname()
{
    return this->OwningSystem->GetHostname();
}

FString UUserContext::GetUsername()
{
    return this->OwningSystem->GetUsername(this->UserID);
}

FString UUserContext::GetCharacterName()
{
    return this->OwningSystem->GetCharacter().CharacterName;
}

FString UUserContext::GetHomeDirectory()
{
    return this->OwningSystem->GetUserHomeDirectory(this->UserID);
}

UPeacegateFileSystem* UUserContext::GetFilesystem()
{
    return this->OwningSystem->GetFilesystem(this->UserID);
}

URainbowTable* UUserContext::GetRainbowTable()
{
    return this->OwningSystem->GetRainbowTable();
}

APeacenetWorldStateActor* UUserContext::GetPeacenet()
{
    return this->OwningSystem->GetPeacenet();
}

UDesktopWidget* UUserContext::GetDesktop()
{
    // TODO: User context should own the desktop, not the system context. This will allow remote desktop hacking.
    return this->OwningSystem->GetDesktop();
}

USystemContext* UUserContext::GetOwningSystem()
{
    return this->OwningSystem;
}

bool UUserContext::IsAdministrator()
{
	return this->GetUserInfo().IsAdminUser;
}

bool UUserContext::OpenProgram(FName InExecutableName, UProgram*& OutProgram, bool InCheckForExistingWindow)
{
    return this->GetOwningSystem()->OpenProgram(InExecutableName, OutProgram, InCheckForExistingWindow);
}

void UUserContext::ShowProgramOnWorkspace(UProgram* InProgram)
{
    // Check the program and desktop.
    check(InProgram);
    check(this->GetDesktop());

    // Show it on our workspace.
    this->GetDesktop()->ShowProgramOnWorkspace(InProgram);
}

UConsoleContext* UUserContext::CreateConsole(UPTerminalWidget* InTerminalWidget)
{
    // CHeck the terminal widget, our system context, etc
    check(this->GetOwningSystem());
    check(InTerminalWidget);

	UConsoleContext* SubConsole = NewObject<UConsoleContext>(this);

	// Assign it to the terminal widget.
	SubConsole->SetTerminal(InTerminalWidget);

    // TODO: Give the console context ourselves as a user context,
    // NOT OUR UNDERLYING SYSTEM AND UID.

	// User ID matches our window.
	SubConsole->Setup(this);

	SubConsole->SetWorkingDirectory(SubConsole->GetUserContext()->GetHomeDirectory());
	
	return SubConsole;

}

FString UUserContext::GetUserTypeDisplay()
{
	if(this->IsAdministrator())
	{
		return "#";
	}
	else
	{
		return "$";
	}
}

bool UUserContext::OpenFile(const FString& InPath, EFileOpenResult& OutResult)
{
	if (!this->GetFilesystem())
	{
		OutResult = EFileOpenResult::PermissionDenied;
		return false;
	}

	if (!this->GetFilesystem()->FileExists(InPath))
	{
		OutResult = EFileOpenResult::FileNotFound;
		return false;
	}

	FString Path;
	FString Extension;
	if (!InPath.Split(TEXT("."), &Path, &Extension, ESearchCase::IgnoreCase, ESearchDir::FromEnd))
	{
		OutResult = EFileOpenResult::NoSuitableProgram;
		return false;
	}

	UPeacegateProgramAsset* ProgramAsset;
	if (!this->GetOwningSystem()->GetSuitableProgramForFileExtension(Extension, ProgramAsset))
	{
		OutResult = EFileOpenResult::NoSuitableProgram;
		return false;
	}

    // TODO: Shouldn't the CreateProgram function deal with this internally?
	TSubclassOf<UWindow> WindowClass = this->GetPeacenet()->WindowClass;

	UWindow* NewWindow;
	UProgram* NewProgram = UProgram::CreateProgram(WindowClass, ProgramAsset->ProgramClass, this->GetOwningSystem(), this->UserID, NewWindow, ProgramAsset->ExecutableName.ToString());

	NewWindow->WindowTitle = ProgramAsset->AppLauncherItem.Name;
	NewWindow->Icon = ProgramAsset->AppLauncherItem.Icon;
	NewWindow->EnableMinimizeAndMaximize = ProgramAsset->AppLauncherItem.EnableMinimizeAndMaximize;

	NewProgram->FileOpened(InPath);

	return true;
}

TArray<UWallpaperAsset*> UUserContext::GetAvailableWallpapers()
{
	return this->GetOwningSystem()->GetAvailableWallpapers();
}

UTexture2D* UUserContext::GetCurrentWallpaper()
{
	return this->GetOwningSystem()->GetComputer().CurrentWallpaper;
}

TArray<FAdjacentNodeInfo> UUserContext::ScanForAdjacentNodes()
{
	check(this->GetOwningSystem());
	return this->GetOwningSystem()->ScanForAdjacentNodes();
}

void UUserContext::SetCurrentWallpaper(UWallpaperAsset* InWallpaperAsset)
{
	this->GetOwningSystem()->SetCurrentWallpaper(InWallpaperAsset);
}