// Fill out your copyright notice in the Description page of Project Settings.


#include "ResetMovementStateAnimNotify.h"
#include "SoulsMeleeCombatSystem/Characters/FremenCharacter.h"

void UResetMovementStateAnimNotify::Notify(USkeletalMeshComponent* MeshComp, UAnimSequenceBase* Animation,
                                    const FAnimNotifyEventReference& EventReference)
{
	Super::Notify(MeshComp, Animation, EventReference);
	if (AFremenCharacter* Fremen = Cast<AFremenCharacter>(MeshComp->GetOwner()))
	{
		Fremen->ResetMovementState();
	}
}
