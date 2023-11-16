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


#include "UDesktopWidget.h"
#include "PeacenetWorldStateActor.h"
#include "USystemContext.h"
#include "FComputer.h"
#include "UUserContext.h"
#include "PTerminalWidget.h"
#include "UGameTypeAsset.h"
#include "FPeacenetIdentity.h"
#include "CommonUtils.h"
#include "WallpaperAsset.h"
#include "UProgram.h"
#include "UConsoleContext.h"

void UDesktopWidget::CloseActiveProgram()
{
	this->EventActiveProgramClose.Broadcast();
}

USystemContext* UDesktopWidget::GetSystemContext()
{
	return this->SystemContext;
}

UProgram * UDesktopWidget::SpawnProgramFromClass(TSubclassOf<UProgram> InClass, const FText& InTitle, UTexture2D* InIcon)
{
	UWindow* OutputWindow = nullptr;

	UProgram* Program = UProgram::CreateProgram(this->SystemContext->GetPeacenet()->WindowClass, InClass, this->SystemContext, this->UserID, OutputWindow, InTitle.ToString());

	OutputWindow->WindowTitle = InTitle;
	OutputWindow->Icon = InIcon;
	OutputWindow->EnableMinimizeAndMaximize = false;

	return Program;
}

void UDesktopWidget::NativeConstruct()
{
	// Reset the app launcher.
	this->ResetAppLauncher();

	// Grab the user's home directory.
	this->UserHomeDirectory = this->SystemContext->GetUserHomeDirectory(this->UserID);

	// Get the filesystem context for this user.
	this->Filesystem = this->SystemContext->GetFilesystem(this->UserID);

	// Make sure that we intercept filesystem write operations that pertain to us.
	TScriptDelegate<> FSOperation;
	FSOperation.BindUFunction(this, "OnFilesystemOperation");

	this->Filesystem->FilesystemOperation.Add(FSOperation);
	this->SystemContext->GetFilesystem(0)->FilesystemOperation.Add(FSOperation);

	// Set the default wallpaper if our computer doesn't have one.
	if(!this->GetSystemContext()->GetComputer().CurrentWallpaper)
	{
		// Go through all wallpaper assets.
		for(auto Wallpaper : this->GetSystemContext()->GetPeacenet()->Wallpapers)
		{
			// Is it the default wallpaper?
			if(Wallpaper->IsDefault)
			{
				// Assign the texture to the computer.
				this->GetSystemContext()->GetComputer().CurrentWallpaper = Wallpaper->WallpaperTexture;
			}
		}
	}

	Super::NativeConstruct();
}

UConsoleContext * UDesktopWidget::CreateConsole(UPTerminalWidget* InTerminal)
{
	// DEPRECATED IN FAVOUR OF UUserContext::CreateConsole().
	return this->SystemContext->GetUserContext(this->UserID)->CreateConsole(InTerminal);
}

EGovernmentAlertStatus UDesktopWidget::GetAlertStatus()
{
	return this->AlertInfo.Status;
}

void UDesktopWidget::SetWallpaper(UTexture2D* InTexture)
{
	this->GetSystemContext()->GetComputer().CurrentWallpaper = InTexture;
}

void UDesktopWidget::OnFilesystemOperation(EFilesystemEventType InType, FString InPath)
{
}

void UDesktopWidget::NativeTick(const FGeometry& MyGeometry, float InDeltaTime)
{
	check(this->SystemContext);
	check(this->SystemContext->GetPeacenet());

	// Keep track of the current game rules.
	this->GameRules = this->SystemContext->GetPeacenet()->GameType->GameRules;

	// Fetch government alert status.
	this->AlertInfo = this->SystemContext->GetPeacenet()->GetAlertInfo(this->SystemContext->GetCharacter().ID);

	this->MyCharacter = this->SystemContext->GetCharacter();
	this->MyComputer = this->SystemContext->GetComputer();

	// If a notification isn't currently active...
	if (!bIsWaitingForNotification)
	{
		// Do we have notifications left in the queue?
		if (this->NotificationQueue.Num())
		{
			// Grab the first notification.
			FDesktopNotification Note = this->NotificationQueue[0];

			// Remove it from the queue - we have a copy.
			this->NotificationQueue.RemoveAt(0);

			// We're waiting again.
			bIsWaitingForNotification = true;

			// Set the notification values.
			this->NotificationTitle = Note.Title;
			this->NotificationMessage = Note.Message;
			this->NotificationIcon = Note.Icon;

			// Fire the event.
			this->OnShowNotification();
		}
	}

	// update username.
	this->CurrentUsername = this->SystemContext->GetUsername(this->UserID);

	// update hostname
	this->CurrentHostname = this->SystemContext->GetHostname();

	// And now the Peacenet name.
	this->CurrentPeacenetName = this->MyCharacter.CharacterName;

	this->TimeOfDay = this->SystemContext->GetPeacenet()->GetTimeOfDay();

	// Set our wallpaper.
	this->WallpaperTexture = this->GetSystemContext()->GetComputer().CurrentWallpaper;

	Super::NativeTick(MyGeometry, InDeltaTime);
}

void UDesktopWidget::ResetAppLauncher()
{
	// Clear the main menu.
	this->ClearAppLauncherMainMenu();

	// Collect app launcher categories.
	TArray<FString> CategoryNames;
	for (auto Program : this->SystemContext->GetInstalledPrograms())
	{
		if (!CategoryNames.Contains(Program->AppLauncherItem.Category.ToString()))
		{
			CategoryNames.Add(Program->AppLauncherItem.Category.ToString());
			this->AddAppLauncherMainMenuItem(Program->AppLauncherItem.Category);
		}
	}
}

int UDesktopWidget::GetOpenConnectionCount()
{
	return this->SystemContext->GetOpenConnectionCount();
}

FString UDesktopWidget::GetIPAddress()
{
	return this->GetSystemContext()->GetIPAddress();
}

void UDesktopWidget::EnqueueNotification(const FText & InTitle, const FText & InMessage, UTexture2D * InIcon)
{
	FDesktopNotification note;
	note.Title = InTitle;
	note.Message = InMessage;
	note.Icon = InIcon;
	this->NotificationQueue.Add(note);
}

void UDesktopWidget::ResetDesktopIcons()
{
}

void UDesktopWidget::ShowAppLauncherCategory(const FString& InCategoryName)
{
	// Clear the sub-menu.
	this->ClearAppLauncherSubMenu();

	// Add all the programs.
	for (auto Program : this->SystemContext->GetInstalledPrograms())
	{
		if (Program->AppLauncherItem.Category.ToString() != InCategoryName)
			continue;

		// because Blueprints hate strings being passed by-value.
		FString ProgramName = Program->ExecutableName.ToString();

		// Add the item. Woohoo.
		this->AddAppLauncherSubMenuItem(Program->AppLauncherItem.Name, Program->AppLauncherItem.Description, ProgramName, Program->AppLauncherItem.Icon);
	}
}

bool UDesktopWidget::OpenProgram(const FName InExecutableName, UProgram*& OutProgram)
{
	return this->SystemContext->OpenProgram(InExecutableName, OutProgram);
}

void UDesktopWidget::FinishShowingNotification()
{
	bIsWaitingForNotification = false;
}