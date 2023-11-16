// Fill out your copyright notice in the Description page of Project Settings.

#include "CppAttributeSetBase.h"

UCppAttributeSetBase::UCppAttributeSetBase() 
	:Health(200.0f), 
	MaxHealth(200.0f),
	Mana(100.0f),
	MaxMana(100.0f),
	Strength(250.0f),
	MaxStrength(250.0f)
{
	



}

void UCppAttributeSetBase::PostGameplayEffectExecute(const struct FGameplayEffectModCallbackData &Data)
{
	/* Start Health */
	if (Data.EvaluatedData.Attribute.GetUProperty() == FindFieldChecked<UProperty>(UCppAttributeSetBase::StaticClass(), GET_MEMBER_NAME_CHECKED(UCppAttributeSetBase, Health)))
	{
		Health.SetCurrentValue(FMath::Clamp(Health.GetCurrentValue(), 0.0f, MaxHealth.GetCurrentValue()));
		Health.SetBaseValue(FMath::Clamp(Health.GetBaseValue(), 0.0f, MaxHealth.GetCurrentValue()));

		UE_LOG(LogTemp, Warning, TEXT("Damage, %f"), Health.GetCurrentValue());

		OnHealthChange.Broadcast(Health.GetCurrentValue(), MaxHealth.GetCurrentValue());
		
		/******* Start - Change logic to work with health attr */

		ACppCharacterBase* CharacterOwner = Cast<ACppCharacterBase>(GetOwningActor());
		
		if (Health.GetCurrentValue() == MaxHealth.GetCurrentValue())
		{
			if (CharacterOwner)
			{
				CharacterOwner->AddGameplayTag(CharacterOwner->FullHealthTag);
			}
		}
		else
		{
			if (CharacterOwner)
			{
				CharacterOwner->RemoveGameplayTag(CharacterOwner->FullHealthTag);
			}
		}

		/******* End - Change logic to work with health attr */
	
	}
	/* End Mana */

	/* Start Mana */
	if (Data.EvaluatedData.Attribute.GetUProperty() == FindFieldChecked<UProperty>(UCppAttributeSetBase::StaticClass(), GET_MEMBER_NAME_CHECKED(UCppAttributeSetBase, Mana)))
	{
		Mana.SetCurrentValue(FMath::Clamp(Mana.GetCurrentValue(), 0.0f, MaxMana.GetCurrentValue()));
		Mana.SetBaseValue(FMath::Clamp(Mana.GetBaseValue(), 0.0f, MaxMana.GetCurrentValue()));

		UE_LOG(LogTemp, Warning, TEXT("Mana, %f"), Mana.GetCurrentValue());

		OnManaChange.Broadcast(Mana.GetCurrentValue(), MaxMana.GetCurrentValue());

	}
	/* End Strength */

	/* Start Strength */
	if (Data.EvaluatedData.Attribute.GetUProperty() == FindFieldChecked<UProperty>(UCppAttributeSetBase::StaticClass(), GET_MEMBER_NAME_CHECKED(UCppAttributeSetBase, Strength)))
	{
		Strength.SetCurrentValue(FMath::Clamp(Strength.GetCurrentValue(), 0.0f, MaxStrength.GetCurrentValue()));
		Strength.SetBaseValue(FMath::Clamp(Strength.GetBaseValue(), 0.0f, MaxStrength.GetCurrentValue()));

		UE_LOG(LogTemp, Warning, TEXT("Strength, %f"), Strength.GetCurrentValue());

		OnStrengthChange.Broadcast(Strength.GetCurrentValue(), MaxStrength.GetCurrentValue());

	}
}
