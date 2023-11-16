// Fill out your copyright notice in the Description page of Project Settings.


#include "BaseWeapon.h"

#include "Components/SphereComponent.h"
#include "GameFramework/Character.h"
#include "Kismet/GameplayStatics.h"
#include "SoulsMeleeCombatSystem/Characters/FremenCharacter.h"
#include "SoulsMeleeCombatSystem/Components/CollisionTraceComponent.h"
#include "SoulsMeleeCombatSystem/Components/CombatComponent.h"
#include "SoulsMeleeCombatSystem/Utils/Logger.h"

// Sets default values
ABaseWeapon::ABaseWeapon()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = false;
	USceneComponent* Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	SetRootComponent(Root);
	
	MeshComponent = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Mesh"));
	MeshComponent->SetupAttachment(Root);

	InteractionSphere = CreateDefaultSubobject<USphereComponent>(TEXT("Sphere"));
	InteractionSphere->SetupAttachment(Root);

	CollisionTraceComponent = CreateDefaultSubobject<UCollisionTraceComponent>(TEXT("CollisionTrace"));
	AddOwnedComponent(CollisionTraceComponent);
}

// Called when the game starts or when spawned
void ABaseWeapon::BeginPlay()
{
	Super::BeginPlay();
	
	if (CollisionTraceComponent)
	{
		CollisionTraceComponent->OnHit.AddDynamic(this, &ABaseWeapon::WeaponHit);
	}
}

void ABaseWeapon::Interact(AActor* Caller)
{
	if (const AFremenCharacter* Character = Cast<AFremenCharacter>(Caller))
	{
		UActorComponent* ActorComponent = Character->GetComponentByClass(UCombatComponent::StaticClass());
		if (UCombatComponent* CombatComponent = Cast<UCombatComponent>(ActorComponent))
		{
			CombatComponent->SetMainWeapon(this);
			// make weapon no longer interactable
			InteractionSphere->BodyInstance.SetObjectType(ECC_WorldDynamic);
		}
	}
}

void ABaseWeapon::OnEquipped()
{
	IEquippable::OnEquipped();
	AttachActor(HeapSocketName);
	CollisionTraceComponent->SetCollisionMesh(GetItemMesh());
	CollisionTraceComponent->AddActorToIgnore(GetOwner());
}

void ABaseWeapon::OnUnequipped()
{
	IEquippable::OnUnequipped();
}

void ABaseWeapon::AttachActor(FName SocketNameToAttach)
{
	auto Character = Cast<ACharacter>(GetOwner());
	if (Character == nullptr)
	{
		Logger::Log(ELogLevel::WARNING, FString::Printf(TEXT("Can't attach %s, owner cast to character failed."), *this->GetName()));
		return;
	}
	bIsHandEquipped = SocketNameToAttach == HandSocketName;
	AttachToComponent(Character->GetMesh(), FAttachmentTransformRules::SnapToTargetIncludingScale, SocketNameToAttach);
}

UPrimitiveComponent* ABaseWeapon::GetItemMesh()
{
	return MeshComponent;
}

bool ABaseWeapon::IsWeaponInHand() const
{
	return bIsHandEquipped;
}

TArray<UAnimMontage*> ABaseWeapon::GetAttackMontages(EAttackType AttackType) const
{
	switch (AttackType)
	{
	case Light:
		return LightAttackMontages;
	case Heavy:
		return HeavyAttackMontages;
	case Charge:
		return ChargeAttackMontages;
	default:
		return TArray<UAnimMontage*>();
	}
}

void ABaseWeapon::WeaponHit(FHitResult HitResult)
{
	Logger::Log(ELogLevel::INFO, __FUNCTION__);
	if (const auto CombatableActor = Cast<ICombatable>(HitResult.GetActor()))
	{
		if (CombatableActor->CanReceiveDamage())
		{
			UGameplayStatics::ApplyPointDamage(HitResult.GetActor(), Damage, HitResult.ImpactNormal, HitResult, GetInstigatorController(), this,  UDamageType::StaticClass());
		}
	}
}

