// Fill out your copyright notice in the Description page of Project Settings.


#include "CombatComponent.h"

#include "SoulsMeleeCombatSystem/Characters/FremenAnimInstance.h"
#include "SoulsMeleeCombatSystem/Characters/FremenCharacter.h"


// Sets default values for this component's properties
UCombatComponent::UCombatComponent()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = false;

	// ...
}


// Called when the game starts
void UCombatComponent::BeginPlay()
{
	Super::BeginPlay();

	// ...
	
}

ABaseWeapon* UCombatComponent::GetMainWeapon() const
{
	return MainWeapon;
}

void UCombatComponent::SetMainWeapon(ABaseWeapon* NewWeapon)
{
	if (NewWeapon)
	{
		if (MainWeapon)
		{
			MainWeapon->OnUnequipped();
		}
		
		MainWeapon = NewWeapon;
		NewWeapon->SetOwner(GetOwner());
		MainWeapon->OnEquipped();
	}
}

bool UCombatComponent::IsCombatEnabled() const
{
	return bIsCombatEnabled;
}

bool UCombatComponent::CanAttack() const
{
	return !bIsAttacking && bIsCombatEnabled;
}

void UCombatComponent::SetCombatEnabled(bool IsCombatEnabled)
{
		if (const AFremenCharacter* Character = Cast<AFremenCharacter>(GetOwner()))
		{
			UAnimInstance* AnimInstance = Character->GetMesh()->GetAnimInstance();
			if (UFremenAnimInstance* FremenAnimInstance = Cast<UFremenAnimInstance>(AnimInstance))
			{
				FremenAnimInstance->UpdateIsCombatEnabled(IsCombatEnabled);
				bIsCombatEnabled = IsCombatEnabled;
			}
		}
}

void UCombatComponent::ResetCombat()
{
	bIsAttacking = false;
	bIsAttackSaved = false;
	AttackCount = 0;
}

