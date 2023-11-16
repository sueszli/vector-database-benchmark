// Fill out your copyright notice in the Description page of Project Settings.


#include "MyBagItem_Consumable.h"
#include "InventorySystemCharacter.h"

AMyBagItem_Consumable::AMyBagItem_Consumable() {
}

void AMyBagItem_Consumable::UseItem() {
	TArray<FEffectInfo> Effects = ItemInfo->Effects;
	AInventorySystemCharacter* myplayer = Cast<AInventorySystemCharacter>(ItemBelongTo);
	for (int i = 0; i < Effects.Num(); i++) {
		switch (Effects[i].EffectCalc) {
		case CalcType::CAdd :
			myplayer->SetIntAttribute(Effects[i].TargetAttr, myplayer->GetIntAttribute(Effects[i].TargetAttr) + Effects[i].calcValue);
			break;
		case CalcType::CMul:
			myplayer->SetFloatAttribute(Effects[i].TargetAttr, myplayer->GetFloatAttribute(Effects[i].TargetAttr) * Effects[i].calcValue);
			break;
		case CalcType::CSet:
			myplayer->SetIntAttribute(Effects[i].TargetAttr, Effects[i].calcValue);
		default: break;

		}
	}
	DecItem(1);
}