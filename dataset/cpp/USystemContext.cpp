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


#include "USystemContext.h"
#include "Kismet/GameplayStatics.h"
#include "PeacenetWorldStateActor.h"
#include "UDesktopWidget.h"
#include "UPeacegateFileSystem.h"
#include "UHackable.h"
#include "CommonUtils.h"
#include "UPeacegateProgramAsset.h"
#include "UUserContext.h"
#include "FAdjacentNode.h"
#include "UProgram.h"
#include "URainbowTable.h"
#include "WallpaperAsset.h"
#include "UGraphicalTerminalCommand.h"
#include "CommandInfo.h"

FString USystemContext::GetProcessUsername(FPeacegateProcess InProcess)
{
	return this->GetUserInfo(InProcess.UID).Username;
}

int USystemContext::GetUserIDFromUsername(FString InUsername)
{
	for(auto User : this->GetComputer().Users)
	{
		if(User.Username == InUsername)
			return User.ID;
	}

	return -1;
}

int USystemContext::GetOpenConnectionCount()
{
	return this->InboundConnections.Num() + this->OutboundConnections.Num();
}

void USystemContext::AddConnection(UHackable* InConnection, bool IsInbound)
{
	check(InConnection);
	check(!InboundConnections.Contains(InConnection));
	check(!OutboundConnections.Contains(InConnection));
	
	if(IsInbound)
		InboundConnections.Add(InConnection);
	else
		OutboundConnections.Add(InConnection);

	// This lets anything running on this system know that something has connected.
	this->SystemConnected.Broadcast(InConnection, IsInbound);
}

bool USystemContext::UsernameExists(FString InUsername)
{
	auto Computer = this->GetComputer();

	for(auto& User : Computer.Users)
	{
		if(User.Username == InUsername)
			return true;
	}

	return false;
}

void USystemContext::Disconnect(UHackable* InConnection)
{
	check(InConnection);

	if(InboundConnections.Contains(InConnection))
		InboundConnections.Remove(InConnection);
	if(OutboundConnections.Contains(InConnection))
		OutboundConnections.Remove(InConnection);
}

FString ReadFirstLine(FString InText)
{
	if (InText.Contains("\n"))
	{
		int NewLineIndex;
		InText.FindChar('\n', NewLineIndex);
		return InText.Left(NewLineIndex).TrimStartAndEnd();
	}
	else
	{
		return InText.TrimStartAndEnd();
	}
}

FString USystemContext::GetHostname()
{
	if (!CurrentHostname.IsEmpty())
	{
		// Speed increase: No need to consult the filesystem for this.
		return CurrentHostname;
	}
	
	UPeacegateFileSystem* RootFS = this->GetFilesystem(0);
	if (RootFS->FileExists("/etc/hostname"))
	{
		EFilesystemStatusCode StatusCode;
		RootFS->ReadText("/etc/hostname", this->CurrentHostname, StatusCode);
		CurrentHostname = ReadFirstLine(CurrentHostname);
		return this->CurrentHostname;
	}

	CurrentHostname = "localhost";
	return CurrentHostname;
}

TArray<UPeacegateProgramAsset*> USystemContext::GetInstalledPrograms()
{
	check(Peacenet);

	TArray<UPeacegateProgramAsset*> OutArray;

	for(auto Program : this->GetPeacenet()->Programs)
	{
		if(GetPeacenet()->GameType->GameRules.DoUnlockables)
		{
			if(!GetComputer().InstalledPrograms.Contains(Program->ExecutableName) && !Program->IsUnlockedByDefault)
				continue;
		}
		OutArray.Add(Program);
	}

	return OutArray;
}

TArray<UCommandInfo*> USystemContext::GetInstalledCommands()
{
	check(this->GetPeacenet());

	TArray<UCommandInfo*> Ret;

	TArray<FName> CommandNames;
	GetPeacenet()->CommandInfo.GetKeys(CommandNames);

	for(auto Name : CommandNames)
	{
		UCommandInfo* Info = GetPeacenet()->CommandInfo[Name];

		if(GetPeacenet()->GameType->GameRules.DoUnlockables)
		{
			if(!GetComputer().InstalledCommands.Contains(Name) && !Info->UnlockedByDefault)
				continue;
		}
		Ret.Add(Info);
	}

	return Ret;
}

TArray<FAdjacentNodeInfo> USystemContext::ScanForAdjacentNodes()
{
	check(this->GetPeacenet());

	int CharID = this->GetCharacter().ID;

	TArray<FAdjacentNodeInfo> Ret;

	for(auto& OtherIdentity : this->GetPeacenet()->GetAdjacentNodes(this->GetCharacter()))
	{
		FAdjacentNodeInfo Node;
		Node.NodeName = OtherIdentity.CharacterName;
		Node.Country = OtherIdentity.Country;
		Node.Link = FAdjacentNode();
		Node.Link.NodeA = CharID;
		Node.Link.NodeB = OtherIdentity.ID;
		Ret.Add(Node);

		if(!this->GetPeacenet()->SaveGame->PlayerDiscoveredNodes.Contains(OtherIdentity.ID))
		{
			this->GetPeacenet()->SaveGame->PlayerDiscoveredNodes.Add(OtherIdentity.ID);
		}
	}

	return Ret;
}

bool USystemContext::OpenProgram(FName InExecutableName, UProgram*& OutProgram, bool InCheckForExistingWindow)
{
	check(this->GetPeacenet());
	check(this->GetDesktop());

	UPeacegateProgramAsset* PeacegateProgram = nullptr;

	if(!this->GetPeacenet()->FindProgramByName(InExecutableName, PeacegateProgram))
		return false;

	if(this->GetPeacenet()->GameType->GameRules.DoUnlockables)
	{
		if(!this->GetComputer().InstalledPrograms.Contains(InExecutableName) && !PeacegateProgram->IsUnlockedByDefault)
		{
			return false;
		}
	}

	UProgram* Program = this->GetDesktop()->SpawnProgramFromClass(PeacegateProgram->ProgramClass, PeacegateProgram->AppLauncherItem.Name, PeacegateProgram->AppLauncherItem.Icon);

	check(Program);

	this->GetDesktop()->ShowProgramOnWorkspace(Program);

	return Program;
}

UPeacegateFileSystem * USystemContext::GetFilesystem(const int UserID)
{
	if (!RegisteredFilesystems.Contains(UserID))
	{
		UPeacegateFileSystem* NewFS = UCommonUtils::CreateFilesystem(this, UserID);
		TScriptDelegate<> ModifiedDelegate;
		ModifiedDelegate.BindUFunction(this, "HandleFileSystemEvent");
		NewFS->FilesystemOperation.Add(ModifiedDelegate);
		this->RegisteredFilesystems.Add(UserID, NewFS);
		return NewFS;
	}

	return this->RegisteredFilesystems[UserID];
}

bool USystemContext::TryGetTerminalCommand(FName CommandName, ATerminalCommand *& OutCommand, FString& InternalUsage, FString& FriendlyUsage)
{
	check(Peacenet);

	if (!GetPeacenet()->ManPages.Contains(CommandName))
		return false;

	FManPage ManPage = GetPeacenet()->ManPages[CommandName];

	InternalUsage = ManPage.InternalUsage;
	FriendlyUsage = ManPage.FriendlyUsage;

	UPeacegateProgramAsset* Program = nullptr;
	if (GetPeacenet()->FindProgramByName(CommandName, Program))
	{
		if(GetPeacenet()->GameType->GameRules.DoUnlockables)
		{
			if(!GetComputer().InstalledPrograms.Contains(CommandName) && !Program->IsUnlockedByDefault)
			{
				return false;
			}
		}

 		FVector Location(0.0f, 0.0f, 0.0f);
		 FRotator Rotation(0.0f, 0.0f, 0.0f);
 		FActorSpawnParameters SpawnInfo;

		AGraphicalTerminalCommand* GraphicalCommand = this->GetPeacenet()->GetWorld()->SpawnActor<AGraphicalTerminalCommand>(Location, Rotation, SpawnInfo);
		GraphicalCommand->ProgramAsset = Program;
		GraphicalCommand->CommandInfo = Peacenet->CommandInfo[CommandName];
		OutCommand = GraphicalCommand;
		return true;
	}

	if (!GetPeacenet()->CommandInfo.Contains(CommandName))
	{
		return false;
	}

	UCommandInfo* Info = GetPeacenet()->CommandInfo[CommandName];

	if(GetPeacenet()->GameType->GameRules.DoUnlockables)
	{
		if(!GetComputer().InstalledCommands.Contains(CommandName) && !Info->UnlockedByDefault)
		{
			return false;
		}
	}

 	FVector Location(0.0f, 0.0f, 0.0f);
	FRotator Rotation(0.0f, 0.0f, 0.0f);
 	FActorSpawnParameters SpawnInfo;

	OutCommand = this->GetPeacenet()->GetWorld()->SpawnActor<ATerminalCommand>(Info->Info.CommandClass, Location, Rotation, SpawnInfo);

	OutCommand->CommandInfo = Info;

	return true;
}

FString USystemContext::GetIPAddress()
{
	check(this->GetPeacenet());

	return this->GetPeacenet()->GetIPAddress(this->GetComputer());
}

FUserInfo USystemContext::GetUserInfo(const int InUserID)
{
	if (InUserID == -1)
	{
		FUserInfo AnonInfo;
		AnonInfo.IsAdminUser = false;
		AnonInfo.Username = "<anonymous>";
		return AnonInfo;
	}

	for (FUser User : GetComputer().Users)
	{
		if (User.ID == InUserID)
		{
			FUserInfo Info;
			Info.Username = User.Username;
			Info.IsAdminUser = (User.Domain == EUserDomain::Administrator);
			return Info;
		}
	}

	return FUserInfo();
}

void USystemContext::ShowWindowOnWorkspace(UProgram * InProgram)
{
	// DEPRECATED IN FAVOUR OF UUserContext::ShowProgramOnWorkspace().
	if (Desktop && InProgram)
	{
		Desktop->ShowProgramOnWorkspace(InProgram);
	}
}

EUserDomain USystemContext::GetUserDomain(int InUserID)
{
	if (InUserID == -1)
	{
		return EUserDomain::Anonymous;
	}

	for (FUser User : GetComputer().Users)
	{
		if (User.ID == InUserID)
		{
			return User.Domain;
		}
	}

	return EUserDomain::User;
}

FString USystemContext::GetUsername(int InUserID)
{
	FUserInfo UserInfo = this->GetUserInfo(InUserID);
	return UserInfo.Username;
}

FString USystemContext::GetUserHomeDirectory(int UserID)
{
	if (this->GetUserDomain(UserID) == EUserDomain::Anonymous)
	{
		return "/";
	}

	for (FUser User : GetComputer().Users)
	{
		if (User.ID == UserID)
		{
			if (User.Domain == EUserDomain::Administrator)
				return TEXT("/root");
			return TEXT("/home/") + User.Username;
		}
	}

	return FString();
}

bool USystemContext::Authenticate(const FString & Username, const FString & Password, int & UserID)
{
	for (FUser User : GetComputer().Users)
	{
		if (User.Username == Username && User.Password == Password)
		{
			UserID = User.ID;
			return true;
		}
	}

	return false;
}

bool USystemContext::GetSuitableProgramForFileExtension(const FString & InExtension, UPeacegateProgramAsset *& OutProgram)
{
	for(auto Program : this->GetPeacenet()->Programs)
	{
		if(this->GetPeacenet()->GameType->GameRules.DoUnlockables)
		{
			if(!this->GetComputer().InstalledPrograms.Contains(Program->ExecutableName) && !Program->IsUnlockedByDefault)
			{
				continue;
			}
		}

		if (Program->SupportedFileExtensions.Contains(InExtension))
		{
			OutProgram = Program;
			return true;
		}
	}
	return false;
}

bool USystemContext::IsIPAddress(FString InIPAddress)
{
	return this->GetPeacenet()->SaveGame->ComputerIPMap.Contains(InIPAddress);
}

UDesktopWidget* USystemContext::GetDesktop()
{
	return this->Desktop;
}

FPeacenetIdentity& USystemContext::GetCharacter()
{
	check(this->GetPeacenet());

	auto MyPeacenet = this->GetPeacenet();

	int CharacterIndex = 0;
	FPeacenetIdentity Character;

	check(MyPeacenet->SaveGame->GetCharacterByID(this->CharacterID, Character, CharacterIndex));

	return MyPeacenet->SaveGame->Characters[CharacterIndex];
}

UUserContext* USystemContext::GetUserContext(int InUserID)
{
	if(this->Users.Contains(InUserID))
	{
		return this->Users[InUserID];
	}
	else
	{
		UUserContext* User = NewObject<UUserContext>(this);
		User->Setup(this, InUserID);
		Users.Add(InUserID, User);
		return User;
	}
}

FComputer& USystemContext::GetComputer()
{
	check(this->GetPeacenet());

	auto MyPeacenet = this->GetPeacenet();

	int ComputerIndex = 0;
	FComputer Computer;

	check(MyPeacenet->SaveGame->GetComputerByID(this->ComputerID, Computer, ComputerIndex));

	return MyPeacenet->SaveGame->Computers[ComputerIndex];
}

APeacenetWorldStateActor* USystemContext::GetPeacenet()
{
	return this->Peacenet;
}

URainbowTable* USystemContext::GetRainbowTable()
{
	return this->RainbowTable;
}

void USystemContext::SetupDesktop(int InUserID)
{
	check(!this->GetDesktop());

	APlayerController* PlayerController = UGameplayStatics::GetPlayerController(GetPeacenet()->GetWorld(), 0);

	this->Desktop = CreateWidget<UDesktopWidget, APlayerController>(PlayerController, this->GetPeacenet()->DesktopClass);

	check(GetDesktop());

	this->Desktop->SystemContext = this;
	this->Desktop->UserID = InUserID;
}

void USystemContext::GetFolderTree(TArray<FFolder>& OutFolderTree)
{
	OutFolderTree = GetComputer().Filesystem;
}

void USystemContext::PushFolderTree(const TArray<FFolder>& InFolderTree)
{
	GetComputer().Filesystem = InFolderTree;
}

FText USystemContext::GetTimeOfDay()
{
	return GetPeacenet()->GetTimeOfDay();
}

void USystemContext::ExecuteCommand(FString InCommand)
{
	check(this->GetDesktop());

	this->GetDesktop()->ExecuteCommand(InCommand);
}

void USystemContext::HandleFileSystemEvent(EFilesystemEventType InType, FString InPath)
{
	switch (InType)
	{
	case EFilesystemEventType::WriteFile:
		if (InPath == "/etc/hostname")
		{
			auto fs = GetFilesystem(0);
			EFilesystemStatusCode err;
			fs->ReadText("/etc/hostname", this->CurrentHostname, err);
			CurrentHostname = ReadFirstLine(CurrentHostname);
		}
		break;
	}

	// If the path is within /var we might want to check to make sure the log still exists.
	if (InPath.StartsWith("/var"))
	{
		auto RootFS = GetFilesystem(0);

		EFilesystemStatusCode Anus;

		// Does /var/log not exist?
		if (!RootFS->DirectoryExists("/var/log"))
		{
			if (!RootFS->DirectoryExists("/var"))
			{
				RootFS->CreateDirectory("/var", Anus);
			}
			RootFS->CreateDirectory("/var/log", Anus);
		}

		// Does peacegate.log not exist?
		if (!RootFS->FileExists("/var/log/peacegate.log"))
		{
			// write blank log.
			RootFS->WriteText("/var/log/peacegate.log", "");
		}

	}
}

TArray<UWallpaperAsset*> USystemContext::GetAvailableWallpapers()
{
	TArray<UWallpaperAsset*> Ret;
	for (auto Wallpaper : this->GetPeacenet()->Wallpapers)
	{
		if(Wallpaper->IsDefault || Wallpaper->UnlockedByDefault || this->GetComputer().UnlockedWallpapers.Contains(Wallpaper->InternalID))
		{
			Ret.Add(Wallpaper);
		}
	}
	return Ret;
}

void USystemContext::SetCurrentWallpaper(UWallpaperAsset* InWallpaperAsset)
{
	// Make sure it's not null.
	check(InWallpaperAsset);

	// If it's unlocked by default or already unlocked, we just set it.
	if(InWallpaperAsset->UnlockedByDefault || InWallpaperAsset->IsDefault || this->GetComputer().UnlockedWallpapers.Contains(InWallpaperAsset->InternalID))
	{
		// Set the wallpaper.
		this->GetComputer().CurrentWallpaper = InWallpaperAsset->WallpaperTexture;
	}
	else
	{
		// BETA TODO: Announce item unlock.
		this->GetComputer().UnlockedWallpapers.Add(InWallpaperAsset->InternalID);

		// Set the wallpaper.
		this->GetComputer().CurrentWallpaper = InWallpaperAsset->WallpaperTexture;
	}
}

void USystemContext::UpdateSystemFiles()
{
	// This function updates the system based on save data and in-game assets.
	//
	// A.K.A: This is the function that updates things like what wallpapers are installed.

	// So first we need a root fs context.
	UPeacegateFileSystem* RootFS = this->GetFilesystem(0);

	EFilesystemStatusCode Anus;

	// Does /var/log not exist?
	if (!RootFS->DirectoryExists("/var/log"))
	{
		if (!RootFS->DirectoryExists("/var"))
		{
			RootFS->CreateDirectory("/var", Anus);
		}
		RootFS->CreateDirectory("/var/log", Anus);
	}

	// Does peacegate.log not exist?
	if (!RootFS->FileExists("/var/log/peacegate.log"))
	{
		// write blank log.
		RootFS->WriteText("/var/log/peacegate.log", "");
	}

	// This is also where we init our rainbow table.
	this->RainbowTable = NewObject<URainbowTable>(this);
	this->RainbowTable->Setup(this, "/etc/rainbow_table.db", true);
}

void USystemContext::Setup(int InComputerID, int InCharacterID, APeacenetWorldStateActor* InPeacenet)
{
	check(InPeacenet);

	// assign all our IDs and Peacenet.
	this->ComputerID = InComputerID;
	this->CharacterID = InCharacterID;
	this->Peacenet = InPeacenet;

	// Now we need a filesystem.
	UPeacegateFileSystem* fs = this->GetFilesystem(0);

	// Any FS errors are reported here.
	EFilesystemStatusCode fsStatus = EFilesystemStatusCode::OK;

	// Create /home if it doesn't exist.
	if(!fs->DirectoryExists("/home"))
		fs->CreateDirectory("/home", fsStatus);

	// Go through every user on the system.
	for(auto& user : this->GetComputer().Users)
	{
		// Get the home directory for the user.
		FString home = this->GetUserHomeDirectory(user.ID);

		// If the user's home directory doesn't exist, create it.
		if(!fs->DirectoryExists(home))
		{
			fs->CreateDirectory(home, fsStatus);
		}

		// These sub-directories are important too.
		TArray<FString> homeDirs = {
			"Desktop",
			"Documents",
			"Downloads",
			"Music",
			"Pictures",
			"Videos"
		};

		for(auto subDir : homeDirs)
		{
			if(!fs->DirectoryExists(home + "/" + subDir))
				fs->CreateDirectory(home + "/" + subDir, fsStatus);
		}
	}
}

TArray<FPeacegateProcess> USystemContext::GetRunningProcesses()
{
	return this->Processes;
}

int USystemContext::StartProcess(FString Name, FString FilePath, int UserID)
{
	int NewPID = 0;
	for(auto Process : this->GetRunningProcesses())
	{
		if(NewPID <= Process.PID)
			NewPID = Process.PID + 1;
	}

	FPeacegateProcess NewProcess;
	NewProcess.PID = NewPID;
	NewProcess.UID = UserID;
	NewProcess.ProcessName = Name;
	NewProcess.FilePath = FilePath;
	this->Processes.Add(NewProcess);
	return NewProcess.PID;
}

void USystemContext::FinishProcess(int ProcessID)
{
	for(int i = 0; i < Processes.Num(); i++)
	{
		FPeacegateProcess p = Processes[i];
		if(p.PID == ProcessID)
		{
			this->Processes.RemoveAt(i);
			return;
		}
	}
}

bool USystemContext::IsEnvironmentVariableSet(FString InVariable)
{
	// Simply returns whether the computer has an environment variable set.
	return this->GetComputer().EnvironmentVariables.Contains(InVariable);
}

bool USystemContext::GetEnvironmentVariable(FString InVariable, FString& OutValue)
{
	// Check if we have the variable:
	if(this->IsEnvironmentVariableSet(InVariable))
	{
		// Retrieve the value.
		OutValue = this->GetComputer().EnvironmentVariables[InVariable];
		return true;
	}
	return false;
}

void USystemContext::SetEnvironmentVariable(FString InVariable, FString InValue)
{
	// If we have a variable with the same name, we set it. Else, we add it.
	if(this->IsEnvironmentVariableSet(InVariable))
	{
		this->GetComputer().EnvironmentVariables[InVariable] = InValue;
	}
	else
	{
		this->GetComputer().EnvironmentVariables.Add(InVariable, InValue);
	}
}

void USystemContext::UnsetEnvironmentVariable(FString InVariable)
{
	// If we have the variable set, remove it.
	if(this->IsEnvironmentVariableSet(InVariable))
	{
		this->GetComputer().EnvironmentVariables.Remove(InVariable);
	}
}
