// Fill out your copyright notice in the Description page of Project Settings.


#include "MyGameState.h"

AMyGameState::AMyGameState()
{
	static ConstructorHelpers::FObjectFinder<UDataTable> BP_ItemDB(TEXT("DataTable'/Game/Data/ItemDB.ItemDB'"));
	ItemDataTable = BP_ItemDB.Object;

	static ConstructorHelpers::FObjectFinder<UDataTable> BP_WeaponDB(TEXT("DataTable'/Game/Data/WeaponDB.WeaponDB'"));
	WeaponTable = BP_WeaponDB.Object;

	ItemInfoTable = CreateDefaultSubobject<UItemInfoTable>(TEXT("ItemInfoTable"));
	TArray<FInventoryItem*>RowsData;
	ItemDataTable->GetAllRows<FInventoryItem>(FString(TEXT("test")), RowsData);
	for (auto it : RowsData) {
		UItemInfo* nowItemInfo = NewObject<UItemInfo>();
		nowItemInfo->ItemName = it->ItemName;
		nowItemInfo->ItemTypeId = it->ItemTypeId;
		nowItemInfo->ItemDescription = it->ItemDescription;
		nowItemInfo->ItemWeight = it->ItemWeight;
		nowItemInfo->ItemMaxCount = it->ItemMaxCount;
		nowItemInfo->Effects = it->ItemEffects;
		nowItemInfo->ItemIcon = it->ItemIcon;
		ItemInfoTable->infotable.Add(nowItemInfo->ItemTypeId, nowItemInfo);
	}
	TArray<FWeaponInfo*>WeaponRowsData;
	WeaponTable->GetAllRows<FWeaponInfo>(FString(TEXT("test")), WeaponRowsData);
	for (auto it : WeaponRowsData) {
		ItemInfoTable->weapontable.Add(it->ItemTypeId, *it);
	}
}

UItemInfoTable* AMyGameState::GetItemInfoTable() const
{
	return ItemInfoTable;
}