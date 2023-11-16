// Fill out your copyright notice in the Description page of Project Settings.


#include "MyPlayerController.h"
#include "Net/UnrealNetwork.h"
#include "MyBagItem_Weapon.h"
AMyPlayerController::AMyPlayerController() {
	myBackPack = CreateDefaultSubobject<UBackPackComponent>(TEXT("myBackPack"));
	myBackPack->Owner = this;
	myBackPack->SetNetAddressable();
	myBackPack->SetIsReplicated(true);
}

int32 AMyPlayerController::GetItemCountById(int TypeId) {
	ABagItem* myItem = Cast<ABagItem>(myBackPack->GetItemById(TypeId));
	if (myItem != NULL) {
		return myItem->ItemCount;
	}
	else {
		return 0;
	}
}

int32 AMyPlayerController::GetWeaponDamageById(int TypeId) {
	ABagItem* myItem = Cast<ABagItem>(myBackPack->GetItemById(TypeId));
	if (myItem != NULL) {
		AMyBagItem_Weapon* myweapon = Cast<AMyBagItem_Weapon>(myItem);
		return myweapon->Damage;
	}
	else {
		return 1;
	}
}