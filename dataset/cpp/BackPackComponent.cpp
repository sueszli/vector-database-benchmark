// Fill out your copyright notice in the Description page of Project Settings.


#include "BackPackComponent.h"
#include "MyGameState.h"
#include "teststruct.h"
#include "MyBagItem_Ammo.h"
#include "MyBagItem_Attachment.h"
#include "MyBagItem_Avatar.h"
#include "MyBagItem_Consumable.h"
#include "MyBagItem_Weapon.h"
#include "Kismet/GameplayStatics.h"
#include "MyPlayerController.h"
#include "PickUps.h"
#include "InventorySystemCharacter.h"
#include "engine/actorchannel.h"

// Sets default values for this component's properties
UBackPackComponent::UBackPackComponent()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = true;

	Owner = NULL;
	BackPackSize = 150;
	CurrentWeight = 0;
	WeaponSlot.Init(NULL, 2);
	AttachmentSlot.Init(NULL, 4);
	AvatarSlot.Init(NULL, 3);
	// ...
}


// Called when the game starts
void UBackPackComponent::BeginPlay()
{
	Super::BeginPlay();

	// ...
	
}


// Called every frame
void UBackPackComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	// ...
}

void UBackPackComponent::GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& OutLifetimeProps)const {
	Super::GetLifetimeReplicatedProps(OutLifetimeProps);

	DOREPLIFETIME_CONDITION(UBackPackComponent, CurrentWeight, COND_OwnerOnly);
	DOREPLIFETIME_CONDITION(UBackPackComponent, NormalSpace, COND_OwnerOnly);
	DOREPLIFETIME_CONDITION(UBackPackComponent, BackPackSize, COND_OwnerOnly);
	DOREPLIFETIME_CONDITION(UBackPackComponent, WeaponSlot, COND_OwnerOnly);
	DOREPLIFETIME_CONDITION(UBackPackComponent, AttachmentSlot, COND_OwnerOnly);
	DOREPLIFETIME_CONDITION(UBackPackComponent, EquipedSlotId, COND_OwnerOnly);
}

bool UBackPackComponent::ReplicateSubobjects(class UActorChannel* Channel, class FOutBunch* Bunch, FReplicationFlags* RepFlags){
	bool WroteSomething = Super::ReplicateSubobjects(Channel, Bunch, RepFlags);
	return WroteSomething;
}

AActor* UBackPackComponent::GetItemById(int32 TypeId) {
	for (int i = 0; i < NormalSpace.Num(); i++)
		if (NormalSpace[i] != NULL && NormalSpace[i]->ItemTypeId == TypeId)
			return NormalSpace[i];
	return NULL;
}

bool UBackPackComponent::IsNewItem(int32 ItemTypeId) {
	return GetItemById(ItemTypeId) == NULL;
}

bool UBackPackComponent::CanAddItem(int32 ItemTypeId, int32 Count) {
	AMyGameState* GameState = Cast<AMyGameState>(GetWorld()->GetGameState());
	UItemInfoTable* ItemTable = GameState->GetItemInfoTable();
	UItemInfo* ItemToAdd = ItemTable->GetItemInfoById(ItemTypeId);

	if (IsNewItem(ItemTypeId)) {
		return CurrentWeight + ItemToAdd->ItemWeight * Count <= BackPackSize;
	}
	else {
		ABagItem* oldItem = Cast<ABagItem>(GetItemById(ItemTypeId));
		bool cond1 = CurrentWeight + ItemToAdd->ItemWeight * Count <= BackPackSize;
		bool cond2 = oldItem->ItemCount + Count <= ItemToAdd->ItemMaxCount;
		return (cond1 && cond2);
	}
	
	return true;
}

void UBackPackComponent::AddItem(int32 ItemTypeId, int32 Count, AActor* otherActor) {
	if (IsNewItem(ItemTypeId))
		AddNewItem(ItemTypeId, Count, otherActor);
	else AddOldItem(ItemTypeId, Count);
}

void UBackPackComponent::AddNewItem(int32 ItemTypeId, int32 Count, AActor* otherActor) {
	AMyGameState* GameState = Cast<AMyGameState>(GetWorld()->GetGameState());
	UItemInfoTable* ItemTable = GameState->GetItemInfoTable();
	UItemInfo* ItemToAdd = ItemTable->GetItemInfoById(ItemTypeId);

	ABagItem* newitem;
	switch (ItemTypeId / 100)
	{
	case 1: 
	{
		AMyBagItem_Ammo* ammoitem = GetWorld()->SpawnActor<AMyBagItem_Ammo>();
		newitem = Cast<ABagItem>(ammoitem);
		break;
	}
	case 2:
	{
		AMyBagItem_Attachment* attitem = GetWorld()->SpawnActor<AMyBagItem_Attachment>();
		newitem = Cast<ABagItem>(attitem);
		break;
	}
	case 3:
	{
		AMyBagItem_Avatar* avaitem = GetWorld()->SpawnActor<AMyBagItem_Avatar>();
		newitem = Cast<ABagItem>(avaitem);
		break;
	}
	case 4:
	{
		AMyBagItem_Consumable* conitem = GetWorld()->SpawnActor<AMyBagItem_Consumable>();
		newitem = Cast<ABagItem>(conitem);
		break;
	}
	case 5:
	{
		AMyBagItem_Weapon* weaponitem = GetWorld()->SpawnActor<AMyBagItem_Weapon>();
		FWeaponInfo WeaponToAdd = ItemTable->GetWeaponInfoById(ItemTypeId);
		weaponitem->Damage = WeaponToAdd.Damage;
		weaponitem->AmmoType = WeaponToAdd.AmmoType;
		weaponitem->AmmoInChip = 0;
		weaponitem->MaxAmmoCount = WeaponToAdd.MaxAmmoCount;
		weaponitem->AttachmentType = WeaponToAdd.AttachmentType;
		newitem = Cast<ABagItem>(weaponitem);
		break;
	}
	default:
	{
		newitem = GetWorld()->SpawnActor<ABagItem>();
		break;
	}
	}
	newitem->ItemInfo = ItemToAdd;
	newitem->ItemTypeId = ItemTypeId;
	newitem->SetItemOwner(otherActor);
	newitem->AddItem(Count);
	NormalSpace.Add(newitem);
	CurrentWeight += Count * ItemToAdd->ItemWeight;
}

void UBackPackComponent::AddOldItem(int32 ItemTypeId, int32 Count) {
	ABagItem* oldItem = Cast<ABagItem>(GetItemById(ItemTypeId));
	oldItem->AddItem(Count);
	CurrentWeight += Count * ( oldItem->ItemInfo->ItemWeight );
}

void UBackPackComponent::UseItem_Implementation(int32 ItemTypeId) {
	ABagItem* ThisItem = Cast<ABagItem>(GetItemById(ItemTypeId));
	if ((ItemTypeId / 100) % 10 == 4) {
		AMyBagItem_Consumable* myItem = Cast<AMyBagItem_Consumable>(ThisItem);
		myItem->UseItem();
		CurrentWeight -= myItem->ItemInfo->ItemWeight;
		if (myItem->ItemCount <= 0) {
			NormalSpace.Remove(ThisItem);
			myItem->Destroy();
		}
	}
	else if ((ItemTypeId / 100) % 10 == 1) {
		AMyBagItem_Ammo* myItem = Cast<AMyBagItem_Ammo>(ThisItem);
		myItem->UseItem();
		CurrentWeight -= myItem->ItemInfo->ItemWeight;
		if (myItem->ItemCount <= 0) {
			NormalSpace.Remove(ThisItem);
			myItem->Destroy();
		}
	}
}

void UBackPackComponent::DropItem_Implementation(int32 ItemTypeId, int32 Count) {
	ABagItem* ThisItem = Cast<ABagItem>(GetItemById(ItemTypeId));
	if (ThisItem == NULL)return;
	if (ThisItem->ItemCount < Count)return;
	ThisItem->ItemCount = ThisItem->ItemCount - Count;
	CurrentWeight -= Count * ThisItem->ItemInfo->ItemWeight;
	AInventorySystemCharacter* myplayer = Cast<AInventorySystemCharacter>(ThisItem->ItemBelongTo);
	if (ThisItem->ItemCount <= 0) {
		NormalSpace.Remove(ThisItem);
		ThisItem->Destroy();
	}
	myplayer->GenerateNewPickUp(ItemTypeId, Count);
}


void UBackPackComponent::AddToAttachmentSlot_Implementation(int32 ItemTypeId, int32 pos) {
	ABagItem* ThisItem = Cast<ABagItem>(GetItemById(ItemTypeId));
	if (ThisItem == NULL) return ;
	if (WeaponSlot[pos] == NULL) return ;
	AMyBagItem_Weapon* ParentWeapon = Cast<AMyBagItem_Weapon>(WeaponSlot[pos]);
	pos = pos << 1 | ((ItemTypeId / 10) & 1);
	if (ParentWeapon->AttachmentType[(ItemTypeId / 10) & 1] != ItemTypeId)return ;//检查是否可以放进来

	AMyBagItem_Attachment* AttItem = Cast<AMyBagItem_Attachment>(ThisItem);
	if (AttachmentSlot[pos] != NULL) {
		RemoveFromAttachmentSlot(pos);
	}
	
	AttachmentSlot[pos] = AttItem;
	AttItem->SetParentWeapon(ParentWeapon->ItemTypeId);
	AttItem->SetInSlotState(pos);
	AttItem->EquipItem();
}

void UBackPackComponent::AddToWeaponSlot_Implementation(int32 ItemTypeId, int32 pos) {
	ABagItem* ThisItem = Cast<ABagItem>(GetItemById(ItemTypeId));
	if (ThisItem == NULL) return ;
	AMyBagItem_Weapon* WeapItem = Cast<AMyBagItem_Weapon>(ThisItem);
	if (WeaponSlot[pos] != NULL) {
		RemoveFromWeaponSlot(pos);
	}
	WeapItem->SetInSlotState(pos);
	WeaponSlot[pos] = ThisItem;
	if (pos == EquipedSlotId)
		EquipWeapon(WeapItem->ItemTypeId);
	return ;
}

void UBackPackComponent::RemoveFromWeaponSlot_Implementation(int32 pos) {
	if (WeaponSlot[pos] == NULL) return;
	if (AttachmentSlot[pos << 1]) RemoveFromAttachmentSlot(pos << 1);
	if (AttachmentSlot[pos << 1 | 1])RemoveFromAttachmentSlot(pos << 1 | 1);

	AMyBagItem_Weapon* OldWeapItem = Cast<AMyBagItem_Weapon>(WeaponSlot[pos]);
	if (OldWeapItem->IsEquiped())
		UnEquipWeapon(OldWeapItem->ItemTypeId);
	OldWeapItem->SetInSlotState(-1);
	WeaponSlot[pos] = NULL;
}

void UBackPackComponent::RemoveFromAttachmentSlot_Implementation(int32 pos) {
	if (AttachmentSlot[pos] == NULL) return;

	AMyBagItem_Attachment* OldAttItem = Cast<AMyBagItem_Attachment>(AttachmentSlot[pos]);
	OldAttItem->UnEquipItem();
	OldAttItem->SetInSlotState(-1);
	AttachmentSlot[pos] = NULL;
}


void UBackPackComponent::EquipWeapon_Implementation(int WeaponId) {
	ABagItem* ThisItem = Cast<ABagItem>(GetItemById(WeaponId));
	ThisItem->EquipItem();
	AMyPlayerController* pc = Cast<AMyPlayerController>(Owner);
	AInventorySystemCharacter* player = Cast<AInventorySystemCharacter>(pc->GetCharacter());
	player->RefreshWeaponMesh();

}

void UBackPackComponent::UnEquipWeapon_Implementation(int WeaponId) {
	ABagItem* ThisItem = Cast<ABagItem>(GetItemById(WeaponId));
	ThisItem->UnEquipItem();
	AMyPlayerController* pc = Cast<AMyPlayerController>(Owner);
	AInventorySystemCharacter* player = Cast<AInventorySystemCharacter>(pc->GetCharacter());
	player->RefreshWeaponMesh();
}


void UBackPackComponent::ReloadWeapon_Implementation(int WeaponId) {
	AMyBagItem_Weapon* ThisItem = Cast<AMyBagItem_Weapon>(GetItemById(WeaponId));
	ThisItem->Reload();
}

void UBackPackComponent::ChangeEquipedSlot_Implementation() {
	if (WeaponSlot[EquipedSlotId] != NULL) {
		AMyBagItem_Weapon* OldWeapon = Cast<AMyBagItem_Weapon>(WeaponSlot[EquipedSlotId]);
		UnEquipWeapon(OldWeapon->ItemTypeId);
	}
	EquipedSlotId = 1 - EquipedSlotId;
	if (WeaponSlot[EquipedSlotId] != NULL) {
		AMyBagItem_Weapon* NewWeapon = Cast<AMyBagItem_Weapon>(WeaponSlot[EquipedSlotId]);
		EquipWeapon(NewWeapon->ItemTypeId);
	}
}

void UBackPackComponent::OnRep_RefreshUI() {
	AMyPlayerController* pc = Cast<AMyPlayerController>(Owner);
	AInventorySystemCharacter* player = Cast<AInventorySystemCharacter>(pc->GetCharacter());
	player->RefreshBagUI();
}

void UBackPackComponent::AddAmmoToNormalSpace(int32 ItemTypeId, int32 Count, AActor* otherActor) {
	if (IsNewItem(ItemTypeId)) {
		AMyGameState* GameState = Cast<AMyGameState>(GetWorld()->GetGameState());
		UItemInfoTable* ItemTable = GameState->GetItemInfoTable();
		UItemInfo* ItemToAdd = ItemTable->GetItemInfoById(ItemTypeId);

		ABagItem* newitem;
		AMyBagItem_Ammo* ammoitem = GetWorld()->SpawnActor<AMyBagItem_Ammo>();
		newitem = Cast<ABagItem>(ammoitem);

		newitem->ItemInfo = ItemToAdd;
		newitem->ItemTypeId = ItemTypeId;
		newitem->SetItemOwner(otherActor);
		newitem->AddItem(Count);
		NormalSpace.Add(newitem);
	}
	else {
		AMyBagItem_Ammo* OldItem = Cast<AMyBagItem_Ammo>(GetItemById(ItemTypeId));
		OldItem->AddItem(Count);
	}
}


void UBackPackComponent::RemoveFromAvatarSlot(int32 pos) {
	return;
}

bool UBackPackComponent::AddToAvatarSlot(int32 ItemTypeId, int32 pos) {
	return true;
}