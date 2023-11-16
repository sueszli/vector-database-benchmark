// Fill out your copyright notice in the Description page of Project Settings.


#include "BagItem.h"
#include "MyGameState.h"
#include "InventorySystemCharacter.h"
// Sets default values
ABagItem::ABagItem()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
	bReplicates = true;

	//RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("aaa"));
	bAlwaysRelevant = true;
	ItemBelongTo = NULL;
	ItemCount = 0;
	ItemInSlot = -1;
}

// Called when the game starts or when spawned
void ABagItem::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void ABagItem::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

void ABagItem::GetLifetimeReplicatedProps(TArray< FLifetimeProperty >& OutLifetimeProps) const
{
	Super::GetLifetimeReplicatedProps(OutLifetimeProps);
	DOREPLIFETIME(ABagItem, ItemCount);
	DOREPLIFETIME(ABagItem, ItemInSlot);
	DOREPLIFETIME(ABagItem, ItemEquiped);
	DOREPLIFETIME(ABagItem, ItemBelongTo);
	DOREPLIFETIME(ABagItem, ItemTypeId);
}

void ABagItem::EquipItem() {

}

void ABagItem::UnEquipItem() {

}

bool ABagItem::IsEquiped() {
	return false;
}

void ABagItem::UseItem() {

}

int32 ABagItem::GetWeight() {
	return ItemInfo->ItemWeight;
}

int32 ABagItem::GetMaxCount() {
	return ItemInfo->ItemMaxCount;
}

FString ABagItem::GetName() {
	return ItemInfo->ItemName;
}

FString ABagItem::GetDescription() {
	return ItemInfo->ItemDescription;
}

void ABagItem::AddItem(int32 Count) {
	ItemCount += Count;
}

void ABagItem::DecItem(int32 Count) {
	ItemCount -= Count;
}

void ABagItem::SetItemOwner(AActor* player) {
	ItemBelongTo = player;
}

void ABagItem::SetEquipState(bool mystate) {
	ItemEquiped = mystate;
}

void ABagItem::SetInSlotState(int32 mystate) {
	ItemInSlot = mystate;
}

void ABagItem::OnRep_ItemTypeId() {
	AMyGameState* GameState = Cast<AMyGameState>(GetWorld()->GetGameState());
	UItemInfoTable* ItemTable = GameState->GetItemInfoTable();
	ItemInfo = ItemTable->GetItemInfoById(ItemTypeId);
}

void ABagItem::OnRep_RefreshBagUI() {
	AInventorySystemCharacter* player = Cast<AInventorySystemCharacter>(ItemBelongTo);
	player->RefreshBagUI();
}