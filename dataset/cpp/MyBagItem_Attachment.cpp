// Fill out your copyright notice in the Description page of Project Settings.


#include "MyBagItem_Attachment.h"
#include "MyBagItem_Weapon.h"
#include "InventorySystemCharacter.h"
#include "MyPlayerController.h"
#include "teststruct.h"

void AMyBagItem_Attachment::GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& OutLifetimeProps) const {
	Super::GetLifetimeReplicatedProps(OutLifetimeProps);
	DOREPLIFETIME(AMyBagItem_Attachment, ParentWeaponId);
}

AMyBagItem_Attachment::AMyBagItem_Attachment() {
	ParentWeaponId = -1;
	ItemEquiped = false;
}

void AMyBagItem_Attachment::EquipItem() {
	AInventorySystemCharacter* player = Cast<AInventorySystemCharacter>(ItemBelongTo);
	AMyPlayerController* PC = Cast<AMyPlayerController>(player->GetController());
	TArray<FEffectInfo> Effects = ItemInfo->Effects;
	for (int i = 0; i < Effects.Num(); i++) {
		AMyBagItem_Weapon* myweapon = Cast<AMyBagItem_Weapon>(PC->myBackPack->GetItemById(ParentWeaponId));
		AInventorySystemCharacter* myplayer = Cast<AInventorySystemCharacter>(ItemBelongTo);
		switch (Effects[i].TargetAttr)
		{
		case AttriName::AmmoCount: {
			switch (Effects[i].EffectCalc)
			{
			case CalcType::CAdd:
				myweapon->SetMaxAmmoCount(myweapon->MaxAmmoCount + Effects[i].calcValue); break;

			case CalcType::CMul:
				myweapon->SetMaxAmmoCount(myweapon->MaxAmmoCount * Effects[i].calcValue); break;

			case CalcType::CSet:
				myweapon->SetMaxAmmoCount(Effects[i].calcValue); break;

			default:
				break;
			}
			break;
		}
		case AttriName::Damage: {
			switch (Effects[i].EffectCalc) {
			case CalcType::CAdd:
				myweapon->SetDamage(myweapon->Damage + Effects[i].calcValue);
				if (myplayer->GetCurrentWeapon() == myweapon->ItemTypeId)
					myplayer->SetIntAttribute(AttriName::Damage, myweapon->Damage);
				break;
			default:break;
				break;
			}
		}
		default: {
			break;
		}
		}
	}
	SetEquipState(true);
}

void AMyBagItem_Attachment::UnEquipItem() {
	TArray<FEffectInfo> Effects = ItemInfo->Effects;
	AInventorySystemCharacter* player = Cast<AInventorySystemCharacter>(ItemBelongTo);
	AMyPlayerController* PC = Cast<AMyPlayerController>(player->GetController());
	for (int i = 0; i < Effects.Num(); i++) {
		AMyBagItem_Weapon* myweapon = Cast<AMyBagItem_Weapon>(PC->myBackPack->GetItemById(ParentWeaponId));
		AInventorySystemCharacter* myplayer = Cast<AInventorySystemCharacter>(ItemBelongTo);
		switch (Effects[i].TargetAttr)
		{
		case AttriName::AmmoCount: {
			switch (Effects[i].EffectCalc)
			{
			case CalcType::CAdd: {
				myweapon->SetMaxAmmoCount(myweapon->MaxAmmoCount - Effects[i].calcValue); 
				if (myweapon->AmmoInChip > myweapon->MaxAmmoCount) {
					int32 OutAmmo = myweapon->AmmoInChip - myweapon->MaxAmmoCount;
					myweapon->AmmoInChip -= OutAmmo;
					PC->myBackPack->AddAmmoToNormalSpace(myweapon->AmmoType, OutAmmo, ItemBelongTo);
				}
				break;
			}
			default:
				break;
			}
			break;
		}
		case AttriName::Damage: {
			switch (Effects[i].EffectCalc) {
			case CalcType::CAdd:
				myweapon->SetDamage(myweapon->Damage - Effects[i].calcValue);
				if (myplayer->GetCurrentWeapon() == myweapon->ItemTypeId)
					myplayer->SetIntAttribute(AttriName::Damage, myweapon->Damage);
				break;
			default:break;
				break;
			}
		}
		default: {
			break;
		}
		}
	}
	SetParentWeapon(NULL);
	SetEquipState(false);
}

bool AMyBagItem_Attachment::IsEquiped() {
	return ItemEquiped;
}

void AMyBagItem_Attachment::SetParentWeapon(int32 WeaponId) {
	GEngine->AddOnScreenDebugMessage(-1, 5.0f, FColor::Green, TEXT("SetParentId"));
	ParentWeaponId = WeaponId;
}

