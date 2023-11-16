// Fill out your copyright notice in the Description page of Project Settings.


#include "PickUps.h"
#include "Net/UnrealNetwork.h"

// Sets default values
APickUps::APickUps()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
	bReplicates = true;
	CollisionComp = CreateDefaultSubobject<UCapsuleComponent>(TEXT("CollisionComp"));
	CollisionComp->InitCapsuleSize(40.0f, 50.0f);
	RootComponent = CollisionComp;
	ItemMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("ItemMesh"));
	ItemMesh->SetupAttachment(RootComponent);

	ItemTypeId = -1;
	ItemCount = 0;
}

// Called when the game starts or when spawned
void APickUps::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void APickUps::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

void APickUps::GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& OutLifetimeProps) const {
	Super::GetLifetimeReplicatedProps(OutLifetimeProps);
	DOREPLIFETIME(APickUps, ItemTypeId);
	DOREPLIFETIME(APickUps, ItemCount);
}

void APickUps::SetItem(int32 Id, int32 Count) {
	if (GetLocalRole() < ROLE_Authority) {
		ServerSetItem(Id, Count);
	}
	else {
		ItemTypeId = Id;
		ItemCount = Count;
	}
}

void APickUps::ServerSetItem_Implementation(int32 Id, int32 Count) {
	SetItem(Id, Count);
}