// Copyright Epic Games, Inc. All Rights Reserved.

#include "AbilitySystemIntroGameMode.h"
#include "AbilitySystemIntroHUD.h"
#include "AbilitySystemIntroCharacter.h"
#include "UObject/ConstructorHelpers.h"

AAbilitySystemIntroGameMode::AAbilitySystemIntroGameMode()
	: Super()
{
	// set default pawn class to our Blueprinted character
	static ConstructorHelpers::FClassFinder<APawn> PlayerPawnClassFinder(TEXT("/Game/FirstPersonCPP/Blueprints/FirstPersonCharacter"));
	DefaultPawnClass = PlayerPawnClassFinder.Class;

	// use our custom HUD class
	HUDClass = AAbilitySystemIntroHUD::StaticClass();
}
