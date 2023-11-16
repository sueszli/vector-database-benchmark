// Fill out your copyright notice in the Description page of Project Settings.


#include "MPMenuWidget.h"
#include "Components/Button.h"
#include "OnlineSubsystem.h"
#include "Interfaces/OnlineSessionInterface.h"
#include "OnlineSessionSettings.h"
#include "MultiplayerSessionsSubsystem.h"

void UMPMenuWidget::MenuSetup(int32 numberOfPublicConnections, FString matchType, FString lobbyPath)
{
	PathToLobby = FString::Printf(TEXT("%s?listen"), *lobbyPath);

	NumPublicConnections = numberOfPublicConnections;
	MatchType = matchType;
	
	AddToViewport();
	SetVisibility(ESlateVisibility::Visible);
	bIsFocusable = true;

	UWorld* world = GetWorld();
	if (world)
	{
		APlayerController* ctr = world->GetFirstPlayerController();
		if (ctr)
		{
			FInputModeUIOnly inputModeData;
			inputModeData.SetWidgetToFocus(TakeWidget());
			inputModeData.SetLockMouseToViewportBehavior(EMouseLockMode::DoNotLock);
			ctr->SetInputMode(inputModeData);
			ctr->SetShowMouseCursor(true);
		}
	}

	UGameInstance* gameInstance = GetGameInstance();
	if (gameInstance)
	{
		MultiplayerSessionsSubsystem = gameInstance->GetSubsystem<UMultiplayerSessionsSubsystem>();
	}

	if (MultiplayerSessionsSubsystem)
	{
		MultiplayerSessionsSubsystem->MultiplayerOnCreateSessionComplete.AddDynamic(this, &ThisClass::OnCreateSession);
		MultiplayerSessionsSubsystem->MultiplayerOnStartSessionComplete.AddDynamic(this, &ThisClass::OnStartSession);
		MultiplayerSessionsSubsystem->MultiplayerOnDestroySessionComplete.AddDynamic(this, &ThisClass::OnDestroySession);

		MultiplayerSessionsSubsystem->MultiplayerOnFindSessionsComplete.AddUObject(this, &ThisClass::OnFindSessions);
		MultiplayerSessionsSubsystem->MultiplayerOnJoinSessionComplete.AddUObject(this, &ThisClass::OnJoinSession);
	}

}

void UMPMenuWidget::MenuTeardown()
{
	RemoveFromParent();

	UWorld* world = GetWorld();
	if (world)
	{
		APlayerController* ctr = world->GetFirstPlayerController();
		if (ctr)
		{
			FInputModeGameOnly inputModeData;
			ctr->SetInputMode(inputModeData);
			ctr->SetShowMouseCursor(false);
		}
	}
}

bool UMPMenuWidget::Initialize()
{
	if (!Super::Initialize())
	{
		return false;
	}

	if (HostButton)
	{
		HostButton->OnClicked.AddDynamic(this, &ThisClass::HostButton_OnClick);
	}

	if (JoinButton)
	{
		JoinButton->OnClicked.AddDynamic(this, &ThisClass::JoinButton_OnClick);
	}

	return true;
}

void UMPMenuWidget::OnLevelRemovedFromWorld(ULevel* inLevel, UWorld* inWorld)
{
	MenuTeardown();
	Super::OnLevelRemovedFromWorld(inLevel, inWorld);
}

void UMPMenuWidget::OnCreateSession(bool bWasSuccessful)
{
	if (bWasSuccessful)
	{
		if (GEngine)
		{
			GEngine->AddOnScreenDebugMessage(
				-1,
				15.f,
				FColor::Green,
				FString(TEXT("Session created successfully!"))
			);
		}

		UWorld* world = GetWorld();
		if (world)
		{
			world->ServerTravel(PathToLobby);
		}
	}
	else
	{
		if (GEngine)
		{
			GEngine->AddOnScreenDebugMessage(
				-1,
				15.f,
				FColor::Red,
				FString(TEXT("Failed to create a session!"))
			);

			HostButton->SetIsEnabled(true);
		}
	}
	
}

void UMPMenuWidget::OnFindSessions(const TArray<FOnlineSessionSearchResult>& sessionResults, bool bWasSuccessful)
{
	if (MultiplayerSessionsSubsystem == nullptr)
	{
		return;
	}

	for (auto result : sessionResults)
	{
		FString id = result.GetSessionIdStr();

		FString settingsValue;
		result.Session.SessionSettings.Get(FName("MatchType"), settingsValue);

		if (settingsValue == MatchType)
		{
			MultiplayerSessionsSubsystem->JoinSession(result);
			return;
		}
	}

	if (!bWasSuccessful || sessionResults.Num() <= 0)
	{
		JoinButton->SetIsEnabled(true);
	}

}

void UMPMenuWidget::OnJoinSession(EOnJoinSessionCompleteResult::Type result)
{
	IOnlineSubsystem* subsys = IOnlineSubsystem::Get();
	if (subsys)
	{
		IOnlineSessionPtr sessionInterface = subsys->GetSessionInterface();
		if (sessionInterface.IsValid())
		{
			FString hostAddr;
			sessionInterface->GetResolvedConnectString(NAME_GameSession, hostAddr);

			APlayerController* ctr = GetGameInstance()->GetFirstLocalPlayerController();
			if (ctr)
			{
				ctr->ClientTravel(hostAddr, ETravelType::TRAVEL_Absolute);
			}
		}
	}

	if (result != EOnJoinSessionCompleteResult::Success)
	{
		JoinButton->SetIsEnabled(true);
	}

}

void UMPMenuWidget::OnStartSession(bool bWasSuccessful)
{
}

void UMPMenuWidget::OnDestroySession(bool bWasSuccessful)
{
}

void UMPMenuWidget::HostButton_OnClick()
{
	HostButton->SetIsEnabled(false);

	if (MultiplayerSessionsSubsystem)
	{
		MultiplayerSessionsSubsystem->CreateSession(NumPublicConnections, MatchType);
	}

}

void UMPMenuWidget::JoinButton_OnClick()
{
	JoinButton->SetIsEnabled(false);

	if (MultiplayerSessionsSubsystem)
	{
		MultiplayerSessionsSubsystem->FindSessions(10000);
	}
}
