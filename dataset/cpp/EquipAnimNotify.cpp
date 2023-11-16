// Fill out your copyright notice in the Description page of Project Settings.


#include "EquipAnimNotify.h"

#include "SoulsMeleeCombatSystem/Characters/FremenCharacter.h"
#include "SoulsMeleeCombatSystem/Components/CombatComponent.h"
#include "SoulsMeleeCombatSystem/Items/BaseWeapon.h"

void UEquipAnimNotify::Notify(USkeletalMeshComponent* MeshComp, UAnimSequenceBase* Animation,
                              const FAnimNotifyEventReference& EventReference)
{
	Super::Notify(MeshComp, Animation, EventReference);
	if (AActor* Owner = MeshComp->GetOwner())
	{
		if (const AFremenCharacter* Character = Cast<AFremenCharacter>(Owner))
		{
			UActorComponent* ActorComponent = Character->GetComponentByClass(UCombatComponent::StaticClass());
			if (UCombatComponent* CombatComponent = Cast<UCombatComponent>(ActorComponent))
			{
				ABaseWeapon* Weapon = CombatComponent->GetMainWeapon();
				Weapon->AttachActor(Weapon->IsWeaponInHand() ? Weapon->HeapSocketName : Weapon->HandSocketName);
				CombatComponent->SetCombatEnabled(Weapon->IsWeaponInHand()); //In hand means in combat and vice versa
			}
		}
	}
}
