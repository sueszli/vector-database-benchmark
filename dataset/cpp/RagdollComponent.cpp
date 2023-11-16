// Fill out your copyright notice in the Description page of Project Settings.


#include "Components/RagdollComponent.h"

#include "GameFramework/Character.h"
#include "Utils/Logger.h"

// Sets default values for this component's properties
URagdollComponent::URagdollComponent()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = false;

	// ...
}

// Called when the game starts
void URagdollComponent::BeginPlay()
{
	Super::BeginPlay();
	
	Init();
}


void URagdollComponent::EnableRagdoll() const
{
	if (!CapsuleComponent || !MeshComponent || !MovementComponent)
	{
		Logger::Log(ELogLevel::ERROR, "Failed to enable ragdoll.");
	}

	auto CollisionResponse = FCollisionResponseContainer();
	CollisionResponse.SetResponse(ECC_Pawn, ECR_Ignore);
	CollisionResponse.SetResponse(ECC_Camera, ECR_Ignore);
	CapsuleComponent->SetCollisionResponseToChannels(CollisionResponse);

	MovementComponent->SetMovementMode(MOVE_None);

	MeshComponent->SetCollisionProfileName(TEXT("Ragdoll"), true);
	MeshComponent->SetAllBodiesBelowSimulatePhysics(PelvisBoneName, true, true);
	MeshComponent->SetAllBodiesBelowPhysicsBlendWeight(PelvisBoneName, 1, false, true);

	// Relevant for player pawn only
	if (SpringArmComponent)
	{
		const auto AttachmentRules = FAttachmentTransformRules(EAttachmentRule::KeepWorld, true);
		SpringArmComponent->AttachToComponent(MeshComponent, AttachmentRules ,PelvisBoneName);
		SpringArmComponent->bDoCollisionTest = false;
	}
}

void URagdollComponent::Init()
{
	if (const auto Character = Cast<ACharacter>(GetOwner()))
	{
		MeshComponent = Character->GetMesh();
		CapsuleComponent = Character->GetCapsuleComponent();
		MovementComponent = Cast<UCharacterMovementComponent>(Character->GetMovementComponent());
	}

	if (const auto ActorComponent = GetOwner()->GetComponentByClass(USpringArmComponent::StaticClass()))
	{
		SpringArmComponent = Cast<USpringArmComponent>(ActorComponent);
	}

	if (!CapsuleComponent || !MeshComponent || !MovementComponent)
	{
		Logger::Log(ELogLevel::ERROR, "Initializing ragdoll component has failed, can't resolve all references from owner.");
	}
}


