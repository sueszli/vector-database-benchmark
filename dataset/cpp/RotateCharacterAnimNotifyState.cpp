// Fill out your copyright notice in the Description page of Project Settings.


#include "RotateCharacterAnimNotifyState.h"
#include "SoulsMeleeCombatSystem/Characters/FremenCharacter.h"

void URotateCharacterAnimNotifyState::NotifyTick(USkeletalMeshComponent* MeshComp, UAnimSequenceBase* Animation,
                                                 float FrameDeltaTime, const FAnimNotifyEventReference& EventReference)
{
	Super::NotifyTick(MeshComp, Animation, FrameDeltaTime, EventReference);
	if (AFremenCharacter* Fremen = Cast<AFremenCharacter>(MeshComp->GetOwner()))
	{
		const FRotator InputRotation = Fremen->GetSignificantInputRotation(0.001f);
		const FRotator CurrentRotation = Fremen->GetActorRotation();
		const FRotator InterpolatedRotator = FMath::RInterpConstantTo(CurrentRotation, InputRotation, FrameDeltaTime, Speed);
		
		Fremen->SetActorRotation(InterpolatedRotator);
	}
}
