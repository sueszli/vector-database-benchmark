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


#include "UPeacenetGameInstance.h"
#include "UGameTypeAsset.h"
#include "PeacenetWorldStateActor.h"
#include "UPeacenetSaveGame.h"
#include "FComputer.h"
#include "CommonUtils.h"
#include "FFolder.h"
#include "Base64.h"
#include "FPeacenetIdentity.h"
#include "AssetRegistry/Public/IAssetRegistry.h"
#include "AssetRegistry/Public/AssetRegistryModule.h"
#include "Kismet/GameplayStatics.h"

void UPeacenetGameInstance::CreateWorld(FString InCharacterName, UPeacenetGameTypeAsset* InGameType)
{
	check(InGameType);

	if(APeacenetWorldStateActor::HasExistingOS())
	{
		UGameplayStatics::DeleteGameInSlot("PeacegateOS", 0);
	}

	// Create a new save game.
	UPeacenetSaveGame* SaveGame = NewObject<UPeacenetSaveGame>();

	// The save file needs to keep a record of the game type.
	SaveGame->GameTypeName = InGameType->Name;

	// Setting this value causes the game to generate NPCs and other procedural
	// stuff when we get to spinning up the Peacenet world state actor later on.
	SaveGame->IsNewGame = true;

	// Now we parse the player name so we have a username and hostname.
	FString Username;
	FString Hostname;
	UCommonUtils::ParseCharacterName(InCharacterName, Username, Hostname);

	// We want to create a new computer for the player.
	FComputer PlayerComputer;
	
	// Assign the computer's metadata values.
	PlayerComputer.ID = 0; // This is what the game identifies the computer as internally. IP addresses, domains, etc. map to these IDs.
	PlayerComputer.OwnerType = EComputerOwnerType::Player; // This determines how the procedural generation system interacts with this entity later on.

	// Format the computer's filesystem.
	UFileUtilities::FormatFilesystem(PlayerComputer.Filesystem);

	// Create a new folder called "etc".
	FFolder EtcFolder;
	EtcFolder.FolderID = 1;
	EtcFolder.FolderName = "etc";
	EtcFolder.ParentID = 0;
	EtcFolder.IsReadOnly = false;

	// Add a file called "hostname" inside this folder.
	FFile HostnameFile;
	HostnameFile.FileName = "hostname";
	HostnameFile.FileContent = FBase64::Encode(Hostname);

	// Write the file to the etc folder.
	EtcFolder.Files.Add(HostnameFile);

	// Write the folder to the computer FS.
	PlayerComputer.Filesystem.Add(EtcFolder);

	// How's that for manual file I/O? We just set the computer's hostname...
	// BEFORE PEACEGATE WAS ACTIVATED.

	// Now we create the root user and root folder.
	// We do the same for the player user.
	FUser RootUser;
	RootUser.ID = 0;
	RootUser.Username = "root";
	RootUser.Password = "";
	RootUser.Domain = EUserDomain::Administrator;

	FUser PlayerUser;
	PlayerUser.ID = 1;
	PlayerUser.Username = Username;
	PlayerUser.Password = "";
	PlayerUser.Domain = EUserDomain::PowerUser;

	// Now Peacegate can identify as these users.
	PlayerComputer.Users.Add(RootUser);
	PlayerComputer.Users.Add(PlayerUser);
	
	// But they still need home directories.
	// Create a new folder called "etc".
	FFolder RootFolder;
	RootFolder.FolderID = 2;
	RootFolder.FolderName = "root";
	RootFolder.ParentID = 0;
	RootFolder.IsReadOnly = false;

	FFolder HomeFolder;
	HomeFolder.FolderID = 3;
	HomeFolder.FolderName = "home";
	HomeFolder.ParentID = 0;
	HomeFolder.IsReadOnly = false;

	FFolder UserFolder;
	UserFolder.FolderID = 4;
	UserFolder.FolderName = Username;
	UserFolder.ParentID = 3;
	UserFolder.IsReadOnly = false;
	HomeFolder.SubFolders.Add(UserFolder.FolderID);

	// Write the three new folders to the disk.
	PlayerComputer.Filesystem.Add(RootFolder);
	PlayerComputer.Filesystem.Add(HomeFolder);
	PlayerComputer.Filesystem.Add(UserFolder);

	// Wallpaper needs to be nullptr to prevent a crash.
	PlayerComputer.CurrentWallpaper = nullptr;
	
	// Next thing we need to do is assign these folder IDs as sub folders to the root.
	PlayerComputer.Filesystem[0].SubFolders.Add(EtcFolder.FolderID);
	PlayerComputer.Filesystem[0].SubFolders.Add(RootFolder.FolderID);
	PlayerComputer.Filesystem[0].SubFolders.Add(HomeFolder.FolderID);
	

	// Now, we add the computer to the save file.
	SaveGame->Computers.Add(PlayerComputer);

	// Set the player UID to that of the non-root user on that computer.
	// This makes Peacenet auto-login to this user account
	// when the desktop loads.
	SaveGame->PlayerUserID = PlayerUser.ID;

	// Now we create a Peacenet Identity.
	FPeacenetIdentity PlayerIdentity;

	// Set it up as a player identity so procedural generation doesn't touch it.
	PlayerIdentity.ID = 0;
	PlayerIdentity.CharacterType = EIdentityType::Player;

	// The player identity needs to own their computer for the
	// game to auto-possess it.
	PlayerIdentity.ComputerID = PlayerComputer.ID;

	// Set the name of the player.
	PlayerIdentity.CharacterName = InCharacterName;

	// Set default skill and reputation values.
	PlayerIdentity.Skill = 0;
	PlayerIdentity.Reputation = 0.f;

	// The game type's rules tells us what country to spawn this character in.
	PlayerIdentity.Country = InGameType->GameRules.SpawnCountry;

	// Add the character to the save file.
	SaveGame->Characters.Add(PlayerIdentity);

	// Set the player's location on the map to the origin.
	SaveGame->SetEntityPosition(PlayerIdentity.ID, FVector2D(0.f, 0.f));

	// This makes the game auto-possess the character we just created.
	SaveGame->PlayerCharacterID = PlayerIdentity.ID;

	// Player should know their own existence.
	SaveGame->PlayerDiscoveredNodes.Add(PlayerIdentity.ID);

	// Save the game.
	UGameplayStatics::SaveGameToSlot(SaveGame, "PeacegateOS", 0);
}

TArray<UPeacenetGameTypeAsset*> const& UPeacenetGameInstance::GetGameTypes() const
{
	return this->GameTypes;
}

UPeacenetSettings * UPeacenetGameInstance::GetSettings()
{
	return this->Settings;
}

void UPeacenetGameInstance::SaveSettings()
{
	// Save the settings save to disk.
	UGameplayStatics::SaveGameToSlot(this->Settings, "PeacenetSettings", 0);

	this->SettingsApplied.Broadcast(this->Settings);
}

void UPeacenetGameInstance::LoadSettings()
{
	// Load it in.
	this->Settings = Cast<UPeacenetSettings>(UGameplayStatics::LoadGameFromSlot("PeacenetSettings", 0));
}

void UPeacenetGameInstance::Init()
{
	// Do we have a settings save?
	if (UGameplayStatics::DoesSaveGameExist("PeacenetSettings", 0))
	{
		this->LoadSettings();
	}
	else
	{
		// Create a new save.
		this->Settings = NewObject<UPeacenetSettings>();

		this->SaveSettings();
	}

		// Get the Asset Registry
	FAssetRegistryModule& AssetRegistryModule = FModuleManager::LoadModuleChecked<FAssetRegistryModule>("AssetRegistry");

	// A place to store computer type asset data
	TArray<FAssetData> Assets;

	if (!AssetRegistryModule.Get().GetAssetsByClass("PeacenetGameTypeAsset", Assets, true))
		check(false);

	for (auto& Asset : Assets)
	{
		this->GameTypes.Add(Cast<UPeacenetGameTypeAsset>(Asset.GetAsset()));
	}
}

void UPeacenetGameInstance::Shutdown()
{
	// Unreal Engine's about to shut down.	

	this->SaveSettings();
}