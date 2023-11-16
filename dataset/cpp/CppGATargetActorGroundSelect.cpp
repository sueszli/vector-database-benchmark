// Fill out your copyright notice in the Description page of Project Settings.

#include "DrawDebugHelpers.h"

#include "CppGATargetActorGroundSelect.h"

ACppGATargetActorGroundSelect::ACppGATargetActorGroundSelect()
{
	PrimaryActorTick.bCanEverTick = true;

	Decal = CreateDefaultSubobject<UDecalComponent>("Decal");
	RootComp = CreateDefaultSubobject<USceneComponent>("RootComp");

	SetRootComponent(RootComp);
	Decal->SetupAttachment(RootComp);

	Radius = 200.0f;
	Decal->DecalSize = FVector(Radius);
}


void ACppGATargetActorGroundSelect::StartTargeting(UGameplayAbility* Ability)
{
	OwningAbility = Ability;
	MasterPC = Cast<APlayerController>(Ability->GetOwningActorFromActorInfo()->GetInstigatorController());

	Decal->DecalSize = FVector(Radius);
}

void ACppGATargetActorGroundSelect::ConfirmTargetingAndContinue()
{

	FVector ViewLocation;
	GetPlayerLocatioinPoint(ViewLocation);

	TArray<FOverlapResult> Overlaps;
	TArray<TWeakObjectPtr<AActor>> OverlapedActors;
	bool TraceComplex = false;

	FCollisionQueryParams CollisionQueryParams;
	CollisionQueryParams.bTraceComplex = TraceComplex;
	CollisionQueryParams.bReturnPhysicalMaterial = false;
	APawn* MasterPawn = MasterPC->GetPawn();

	if (MasterPawn)
	{
		CollisionQueryParams.AddIgnoredActor(MasterPawn->GetUniqueID());
	}

	bool TryOverlap = GetWorld()->OverlapMultiByObjectType
	(
		Overlaps, 
		ViewLocation, 
		FQuat::Identity, 
		FCollisionObjectQueryParams(ECC_Pawn), 
		FCollisionShape::MakeSphere(Radius), 
		CollisionQueryParams
	);

	if (TryOverlap)
	{
		for (int32 i = 0; i<Overlaps.Num(); i++) 
		{
			APawn* PawnOverlaped = Cast<APawn>(Overlaps[i].GetActor());
			
			if (PawnOverlaped && !OverlapedActors.Contains(PawnOverlaped)) 
			{
				OverlapedActors.Add(PawnOverlaped);
			}
		}
	}

	FGameplayAbilityTargetData_LocationInfo* CenterLocation = new FGameplayAbilityTargetData_LocationInfo();
	
	if (Decal) CenterLocation->TargetLocation.LiteralTransform = Decal->GetComponentTransform();

	if (OverlapedActors.Num() > 0)
	{
		FGameplayAbilityTargetDataHandle TargetData = StartLocation.MakeTargetDataHandleFromActors(OverlapedActors);
		TargetData.Add(CenterLocation);
		TargetDataReadyDelegate.Broadcast(TargetData);
	}
	else 
	{
		TargetDataReadyDelegate.Broadcast(FGameplayAbilityTargetDataHandle(CenterLocation));
	}
}

void ACppGATargetActorGroundSelect::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);

	FVector LookPoint;
	GetPlayerLocatioinPoint(LookPoint);

	Decal->SetWorldLocation(LookPoint);

}

bool ACppGATargetActorGroundSelect::GetPlayerLocatioinPoint(FVector& OutViewPoint)
{
	FVector ViewPoint;
	FRotator ViewRotation;

	MasterPC->GetPlayerViewPoint(ViewPoint, ViewRotation);

	FHitResult HitResult;
	FCollisionQueryParams QueryParams;

	QueryParams.bTraceComplex = true;

	APawn* MasterPawn = MasterPC->GetPawn();

	if (MasterPawn)
	{
		QueryParams.AddIgnoredActor(MasterPawn->GetUniqueID());
	}


	bool TryTrace = GetWorld()->LineTraceSingleByChannel(HitResult, ViewPoint, ViewPoint + ViewRotation.Vector() * 10000.0f, ECC_Visibility, QueryParams);

	if (TryTrace)
	{
		OutViewPoint = HitResult.ImpactPoint;
	}
	else
	{
		OutViewPoint = FVector();
	}

	return TryTrace;
}
