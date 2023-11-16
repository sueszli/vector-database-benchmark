// Fill out your copyright notice in the Description page of Project Settings.


#include "ItemEffect.h"
#include "InventorySystemCharacter.h"
#include "MyBagItem_Weapon.h"

UItemEffect::UItemEffect()
{
	//EffectCalc = CalcType::CAdd;
	//TargetAttr = EffectAttrs::Health;
	//calcValue = 0;
}


void UItemEffect::ActivateEffect(AActor* otherActor) {
	/*if (EffectTarget == EffectTargets::ToCharacter) {
		AInventorySystemCharacter* myplayer = Cast<AInventorySystemCharacter>(otherActor);
		switch (TargetAttr) {

		case EffectAttrs::Damage: {
			switch (EffectCalc)
			{
			case CalcType::CAdd:
				myplayer->SetDamage(myplayer->GetDamage() + calcValue); break;

			case CalcType::CMul:
				myplayer->SetDamageMul(myplayer->GetDamageMul() * calcValue); break;

			case CalcType::CSet:
				myplayer->SetDamage(calcValue); break;

			default:
				break;
			}
			break;
		}

		case EffectAttrs::Health : {
			switch (EffectCalc)
			{
			case CalcType::CAdd:
				myplayer->SetCurrentHealth(myplayer->GetCurrentHealth() + calcValue); break;

			case CalcType::CMul:
				myplayer->SetCurrentHealth(myplayer->GetCurrentHealth() * calcValue); break;

			case CalcType::CSet:
				myplayer->SetCurrentHealth(calcValue); break;

			default:
				break;
			}
			break;
		}

		default: break;

		}
	}
	else {
		AMyBagItem_Weapon* myweapon = Cast<AMyBagItem_Weapon>(otherActor);
		switch (TargetAttr)
		{
		case EffectAttrs::AmmoCount: {
			switch (EffectCalc)
			{
			case CalcType::CAdd:
				myweapon->SetMaxAmmoCount(myweapon->MaxAmmoCount + calcValue); break;

			case CalcType::CMul:
				myweapon->SetMaxAmmoCount(myweapon->MaxAmmoCount * calcValue); break;

			case CalcType::CSet:
				myweapon->SetMaxAmmoCount(calcValue); break;

			default:
				break;
			}
			break;
		}
		default:
			break;
		}
	}*/
}

void UItemEffect::RemoveEffect(AActor* otherActor) {
	/*if (EffectTarget == EffectTargets::ToCharacter) {
		AInventorySystemCharacter* myplayer = Cast<AInventorySystemCharacter>(otherActor);
		switch (TargetAttr) {

		case EffectAttrs::Damage: {
			switch (EffectCalc)
			{
			case CalcType::CAdd:
				myplayer->SetDamage(myplayer->GetDamage() - calcValue); break;

			case CalcType::CMul:
				myplayer->SetDamageMul(myplayer->GetDamageMul() / calcValue); break;

			default:
				break;
			}
			break;
		}

		default: break;

		}
	}
	else {
		AMyBagItem_Weapon* myweapon = Cast<AMyBagItem_Weapon>(otherActor);
		switch (TargetAttr)
		{
		case EffectAttrs::AmmoCount: {
			switch (EffectCalc)
			{
			case CalcType::CAdd:
				myweapon->SetMaxAmmoCount(myweapon->MaxAmmoCount - calcValue); break;

			case CalcType::CSet:
				myweapon->SetMaxAmmoCount(40); break;

			default:
				break;
			}
			break;
		}
		default:
			break;
		}
	}*/
}
