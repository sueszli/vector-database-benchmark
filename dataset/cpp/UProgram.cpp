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


#include "UProgram.h"
#include "UConsoleContext.h"
#include "UWindow.h"
#include "USystemContext.h"
#include "UUserContext.h"
#include "CommonUtils.h"
#include "GameFramework/PlayerController.h"
#include "Kismet/GameplayStatics.h"
#include "PeacenetWorldStateActor.h"
#include "UPeacegateFileSystem.h"

UUserContext* UProgram::GetUserContext()
{
	return this->Window->GetUserContext();
}

void UProgram::PushNotification(const FText & InNotificationMessage)
{
	this->GetUserContext()->GetDesktop()->EnqueueNotification(this->Window->WindowTitle, InNotificationMessage, this->Window->Icon);
}

void UProgram::RequestPlayerAttention(bool PlaySound)
{
	this->PlayerAttentionNeeded.Broadcast(PlaySound);
}

UProgram* UProgram::CreateProgram(const TSubclassOf<UWindow> InWindow, const TSubclassOf<UProgram> InProgramClass, USystemContext* InSystem, const int InUserID, UWindow*& OutWindow, FString InProcessName, bool DoContextSetup)
{
	// Preventative: make sure the system context isn't null.
	check(InSystem);

	// TODO: Take in a user context instead of a system context and user ID.
	check(InSystem->GetPeacenet());

	// Grab a user context and check if it's valid.
	UUserContext* User = InSystem->GetUserContext(InUserID);

	check(User);

	APlayerController* MyPlayer = UGameplayStatics::GetPlayerController(InSystem->GetPeacenet()->GetWorld(), 0);

	// The window is what contains the program's UI.
	UWindow* Window = CreateWidget<UWindow, APlayerController>(MyPlayer, InWindow);

	// Construct the actual program.
	UProgram* ProgramInstance = CreateWidget<UProgram, APlayerController>(MyPlayer, InProgramClass);

	// Program and window are friends with each other
	ProgramInstance->Window = Window;

	// Window gets our user context.
	Window->SetUserContext(User);

	// Start the process for the program.
	ProgramInstance->ProcessID = User->StartProcess(InProcessName, InProcessName);

	// Make sure we get notified when the window closes.
	TScriptDelegate<> CloseDelegate;
	CloseDelegate.BindUFunction(ProgramInstance, "OwningWindowClosed");

	Window->NativeWindowClosed.Add(CloseDelegate);

	// Set up the program's contexts if we're told to.
	if (DoContextSetup)
	{
		ProgramInstance->SetupContexts();
		ProgramInstance->GetUserContext()->ShowProgramOnWorkspace(ProgramInstance);
	}

	// Return the window and program.
	OutWindow = Window;

	return ProgramInstance;
}

void UProgram::OwningWindowClosed()
{
    // Finish up our process.
    this->GetUserContext()->GetOwningSystem()->FinishProcess(this->ProcessID);
}


void UProgram::ActiveProgramCloseEvent()
{
	if (this->Window->HasAnyUserFocus() || this->Window->HasFocusedDescendants() || this->Window->HasKeyboardFocus())
	{
		this->Window->Close();
	}
}

void UProgram::ShowInfoWithCallbacks(const FText & InTitle, const FText & InMessage, const EInfoboxIcon InIcon, const EInfoboxButtonLayout ButtonLayout, const bool ShowTextInput, const FInfoboxDismissedEvent & OnDismissed, const FInfoboxInputValidator & ValidatorFunction)
{
	Window->ShowInfoWithCallbacks(InTitle, InMessage, InIcon, ButtonLayout, ShowTextInput, OnDismissed, ValidatorFunction);
}

void UProgram::ShowInfo(const FText & InTitle, const FText & InMessage, const EInfoboxIcon InIcon)
{
	Window->ShowInfo(InTitle, InMessage, InIcon);
}

void UProgram::AskForFile(const FString InBaseDirectory, const FString InFilter, const EFileDialogType InDialogType, const FFileDialogDismissedEvent & OnDismissed)
{
	Window->AskForFile(InBaseDirectory, InFilter, InDialogType, OnDismissed);
}

void UProgram::SetupContexts()
{
	// Add ourself to the window's client slot.
	this->Window->AddWindowToClientSlot(this);

	this->NativeProgramLaunched();
}

void UProgram::SetWindowMinimumSize(FVector2D InSize)
{
	Window->SetClientMinimumSize(InSize);
}

void UProgram::NativeProgramLaunched() {}

