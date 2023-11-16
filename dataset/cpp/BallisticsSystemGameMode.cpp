// Copyright Epic Games, Inc. All Rights Reserved.

#include "BallisticsSystemGameMode.h"
#include "BallisticsSystemHUD.h"
#include "BallisticsSystemCharacter.h"
#include "UObject/ConstructorHelpers.h"

ABallisticsSystemGameMode::ABallisticsSystemGameMode()
	: Super()
{
	// set default pawn class to our Blueprinted character
	static ConstructorHelpers::FClassFinder<APawn> PlayerPawnClassFinder(TEXT("/Game/FirstPersonCPP/Blueprints/FirstPersonCharacter"));
	DefaultPawnClass = PlayerPawnClassFinder.Class;

	// use our custom HUD class
	HUDClass = ABallisticsSystemHUD::StaticClass();
}
