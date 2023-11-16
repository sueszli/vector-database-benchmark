// Copyright Epic Games, Inc. All Rights Reserved.

#include "TestTaskInventoryGameMode.h"
#include "TestTaskInventoryCharacter.h"
#include "UObject/ConstructorHelpers.h"

ATestTaskInventoryGameMode::ATestTaskInventoryGameMode()
{
	// set default pawn class to our Blueprinted character
	static ConstructorHelpers::FClassFinder<APawn> PlayerPawnBPClass(TEXT("/Game/ThirdPerson/Blueprints/BP_ThirdPersonCharacter"));
	if (PlayerPawnBPClass.Class != NULL)
	{
		DefaultPawnClass = PlayerPawnBPClass.Class;
	}
}
