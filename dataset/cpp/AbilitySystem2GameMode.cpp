// Copyright 1998-2018 Epic Games, Inc. All Rights Reserved.

#include "AbilitySystem2GameMode.h"
#include "AbilitySystem2HUD.h"
#include "AbilitySystem2Character.h"
#include "UObject/ConstructorHelpers.h"

AAbilitySystem2GameMode::AAbilitySystem2GameMode()
	: Super()
{
	// set default pawn class to our Blueprinted character
	static ConstructorHelpers::FClassFinder<APawn> PlayerPawnClassFinder(TEXT("/Game/FirstPersonCPP/Blueprints/FirstPersonCharacter"));
	DefaultPawnClass = PlayerPawnClassFinder.Class;

	// use our custom HUD class
	HUDClass = AAbilitySystem2HUD::StaticClass();
}
