// Copyright Epic Games, Inc. All Rights Reserved.

#include "MyProject3GameMode.h"
#include "MyProject3Character.h"
#include "UObject/ConstructorHelpers.h"

AMyProject3GameMode::AMyProject3GameMode()
{
	// set default pawn class to our Blueprinted character
	static ConstructorHelpers::FClassFinder<APawn> PlayerPawnBPClass(TEXT("/Game/ThirdPerson/Blueprints/BP_ThirdPersonCharacter"));
	if (PlayerPawnBPClass.Class != NULL)
	{
		DefaultPawnClass = PlayerPawnBPClass.Class;
	}
}
