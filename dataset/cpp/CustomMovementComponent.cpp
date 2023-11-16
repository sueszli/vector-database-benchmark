// Fill out your copyright notice in the Description page of Project Settings.


#include "Components/CustomMovementComponent.h"

#include "MotionWarpingComponent.h"
#include "ClimbingSystem/ClimbingSystemCharacter.h"
#include "ClimbingSystem/DebugHelper.h"
#include "Components/CapsuleComponent.h"
#include "GameFramework/Character.h"
#include "Kismet/KismetMathLibrary.h"
#include "Kismet/KismetSystemLibrary.h"

void UCustomMovementComponent::BeginPlay()
{
	Super::BeginPlay();

	OwnerAnimInstance = CharacterOwner->GetMesh()->GetAnimInstance();
	OwnerAnimInstance->OnMontageEnded.AddDynamic(this, &UCustomMovementComponent::OnMontageEnded);
	OwnerAnimInstance->OnMontageBlendingOut.AddDynamic(this, &UCustomMovementComponent::OnMontageEnded);

	OwnerPlayerCharacter = Cast<AClimbingSystemCharacter>(CharacterOwner);
}

void UCustomMovementComponent::TickComponent(float DeltaTime, ELevelTick TickType,
											 FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
	CanClimbDownLedge();
}

FVector UCustomMovementComponent::ConstrainAnimRootMotionVelocity(const FVector& RootMotionVelocity,
	const FVector& CurrentVelocity) const
{
	const bool bIsPlayingRMMontage = IsFalling() && OwnerAnimInstance && OwnerAnimInstance->IsAnyMontagePlaying();

	if(bIsPlayingRMMontage)
		return RootMotionVelocity;
	
	return Super::ConstrainAnimRootMotionVelocity(RootMotionVelocity, CurrentVelocity);
}

void UCustomMovementComponent::PhysCustom(float deltaTime, int32 Iterations)
{
	if(IsClimbing())
	{
		PhysClimb(deltaTime, Iterations);
	}
	
	Super::PhysCustom(deltaTime, Iterations);
}


float UCustomMovementComponent::GetMaxSpeed() const
{
	if(IsClimbing())
		return MaxClimbSpeed;	
	
	return Super::GetMaxSpeed();
}

float UCustomMovementComponent::GetMaxAcceleration() const
{
	if(IsClimbing())
		return MaxClimbAcceleration;	
	
	return Super::GetMaxSpeed();
}

TArray<FHitResult> UCustomMovementComponent::DoCapsuleTraceMultiByObject(const FVector& Start, const FVector& End,
                                                                       bool bShowDebugShape, bool bDrawPersistantShapes)
{
	TArray<FHitResult> OutCapsuleTraceHitResults;

	EDrawDebugTrace::Type DebugTraceType = EDrawDebugTrace::None;

	if(bShowDebugShape)
		DebugTraceType =bDrawPersistantShapes ? EDrawDebugTrace::Persistent : EDrawDebugTrace::ForOneFrame;

	UKismetSystemLibrary::CapsuleTraceMultiForObjects(
		this,
		Start,
		End,
		ClimbCapsuleTraceRadius,
		ClimbCapsuleTraceHalfHeight,
		ClimbableSurfaceTraceTypes,
		false,
		TArray<AActor*>(),
		DebugTraceType,
		OutCapsuleTraceHitResults,
		false
	);

	return OutCapsuleTraceHitResults;
}

FHitResult UCustomMovementComponent::DoLineTraceSingleByObject(const FVector& Start, const FVector& End,
																bool bShowDebugShape, bool bDrawPersistantShapes)
{
	FHitResult HitResult;

	EDrawDebugTrace::Type DebugTraceType = EDrawDebugTrace::None;

	if(bShowDebugShape)
		DebugTraceType =bDrawPersistantShapes ? EDrawDebugTrace::Persistent : EDrawDebugTrace::ForOneFrame;
	
	UKismetSystemLibrary::LineTraceSingleForObjects(
		this,
		Start,
		End,
		ClimbableSurfaceTraceTypes,
		false,
		TArray<AActor*>(),
		DebugTraceType,
		HitResult,
		false
	);

	return HitResult;
}

bool UCustomMovementComponent::TraceClimbableSurfaces()
{
	const FVector StartOffset = UpdatedComponent->GetForwardVector() * 30.0f;
	const FVector Start = UpdatedComponent->GetComponentLocation() + StartOffset;
	const FVector End = Start + UpdatedComponent->GetForwardVector();
	
	ClimbableSurfacesTracedResults = DoCapsuleTraceMultiByObject(Start,End);
	return !ClimbableSurfacesTracedResults.IsEmpty();
}

FHitResult UCustomMovementComponent::TraceFromEyeHeight(float TraceDistance, float TraceStartOffset, bool bShowDebugShape, bool bDrawPersistantShapes)
{
	const FVector ComponentLocation = UpdatedComponent->GetComponentLocation();
	const FVector EyeHeightOffset = UpdatedComponent->GetUpVector() * (CharacterOwner->BaseEyeHeight + TraceStartOffset);
	
	const FVector Start = ComponentLocation + EyeHeightOffset;
	const FVector End = Start + UpdatedComponent->GetForwardVector() * TraceDistance;

	return DoLineTraceSingleByObject(Start,End, bShowDebugShape, bDrawPersistantShapes);
}

void UCustomMovementComponent::PlayClimbMontage(UAnimMontage* MontageToPlay)
{
	if(!MontageToPlay || !OwnerAnimInstance || OwnerAnimInstance->IsAnyMontagePlaying())
		return;

	OwnerAnimInstance->Montage_Play(MontageToPlay);
}

void UCustomMovementComponent::OnMontageEnded(UAnimMontage* Montage, bool bInterrupted)
{
	if(Montage == IdleToClimbMontage || Montage == ClimbDownLedgeMontage)
	{
		StartClimbing();
		StopMovementImmediately();
	}
	else if(Montage == ClimbToTopMontage || Montage == VaultMontage)
	{
		SetMovementMode(MOVE_Walking);
	}
}

void UCustomMovementComponent::SetMotionWarpTarget(const FName& InWarpTargetName, const FVector& InTargetLocation)
{
	if(!OwnerPlayerCharacter)
		return;

	OwnerPlayerCharacter->GetMotionWarpingComponent()->AddOrUpdateWarpTargetFromLocation(InWarpTargetName, InTargetLocation);
}

void UCustomMovementComponent::ToggleClimbing(bool bEnableClimb)
{
	if(bEnableClimb)
	{
		if(CanStartClimbing())
		{
			PlayClimbMontage(IdleToClimbMontage);
		}
		else if(CanClimbDownLedge())
		{
			PlayClimbMontage(ClimbDownLedgeMontage);
		}
		else
		{
			TryStartVaulting();
		}
	}
	
	if(!bEnableClimb)
	{
		StopClimbing();
	}
}

void UCustomMovementComponent::RequestHopping()
{
	const FVector UnrotatedLastInputVector = UKismetMathLibrary::Quat_UnrotateVector(UpdatedComponent->GetComponentQuat(),GetLastInputVector());
	const FVector UnrotatedLastInputDirection = UnrotatedLastInputVector.GetSafeNormal();
	const float DotVerticalResult = FVector::DotProduct(UnrotatedLastInputDirection, FVector::UpVector);
	const float DotHorizontalResult = FVector::DotProduct(UnrotatedLastInputDirection, FVector::RightVector);

	if(DotVerticalResult > 0.9f)
	{
		HandleHop(HopDirection::UP);
	}
	else if(DotVerticalResult < -0.9f)
	{
		HandleHop(HopDirection::DOWN);
	}
	else if(DotHorizontalResult > 0.9f)
	{
		HandleHop(HopDirection::RIGHT);
	}
	else if(DotHorizontalResult < -0.9f)
	{
		HandleHop(HopDirection::LEFT);
	}
	
}

void UCustomMovementComponent::HandleHop(HopDirection Direction)
{
	FVector HopTargetPoint;
	
	if(CheckCanHop(Direction, HopTargetPoint))
	{
		SetMotionWarpTarget(FName("HopTargetPoint"), HopTargetPoint);

		UAnimMontage* HopMontage = nullptr;

		switch (Direction)
		{
			case HopDirection::UP:
				HopMontage = HopUpMontage;
				break;
			case HopDirection::DOWN :
				HopMontage = HopDownMontage;
				break;
			case HopDirection::RIGHT :
				HopMontage = HopRightMontage;
				break;
			case HopDirection::LEFT :
				HopMontage = HopLeftMontage;
				break;
		}
			
		PlayClimbMontage(HopMontage);
	}
	else
	{
		
	}
}

bool UCustomMovementComponent::CheckCanHop(HopDirection Direction, FVector& OutHopTargetPosition)
{
	FVector TraceDirection = FVector::ZeroVector;
	FVector HeightOffset = FVector::ZeroVector;

	switch (Direction)
	{
	case HopDirection::UP:
		TraceDirection = FVector::UpVector;
		break;
	case HopDirection::DOWN :
		TraceDirection = FVector::DownVector;
		HeightOffset = TraceDirection * 150.0f;
		break;
	case HopDirection::RIGHT :
		TraceDirection = UpdatedComponent->GetRightVector() * 2.0f;
		HeightOffset = FVector::DownVector * GetCharacterOwner()->GetDefaultHalfHeight();
		break;
	case HopDirection::LEFT :
		TraceDirection = -UpdatedComponent->GetRightVector() * 2.0f;
		HeightOffset = FVector::DownVector * GetCharacterOwner()->GetDefaultHalfHeight();
		break;
	}
	
	FVector TraceStart = UpdatedComponent->GetComponentLocation() + TraceDirection * 100.0f + HeightOffset;
	FVector TraceEnd = TraceStart + UpdatedComponent->GetForwardVector() * 100.0f;
	FHitResult HopHitResult = DoLineTraceSingleByObject(TraceStart,TraceEnd);

	if(HopHitResult.bBlockingHit)
	{
		OutHopTargetPosition = HopHitResult.ImpactPoint;
		return true;
	}
	
	return false;	
}

bool UCustomMovementComponent::CanStartClimbing()
{
	if(IsFalling())
		return false;

	if(!TraceClimbableSurfaces())
		return false;

	if(!TraceFromEyeHeight(100.0f).bBlockingHit)
		return false;

	return true;
}

bool UCustomMovementComponent::CanClimbDownLedge()
{
	if(IsFalling())
		return false;

	const FVector ComponentLocation = UpdatedComponent->GetComponentLocation();
	const FVector ComponentForward = UpdatedComponent->GetForwardVector();
	const FVector DownVector = -UpdatedComponent->GetUpVector();

	const FVector WalkableSurfaceTraceStart = ComponentLocation + ComponentForward * ClimbDownWalkableSurfaceTraceOffset;
	const FVector WalkableSurfaceTraceEnd = WalkableSurfaceTraceStart + DownVector * 100.0f;

	FHitResult WalkableSurfaceHit = DoLineTraceSingleByObject(WalkableSurfaceTraceStart,WalkableSurfaceTraceEnd);

	const FVector LedgeTraceStart  = WalkableSurfaceHit.TraceStart +  ComponentForward * ClimbDownLedgeTraceOffset;
	const FVector LedgeTraceEnd  = LedgeTraceStart + DownVector * 200.0f;
	
	FHitResult LedgeTraceHit = DoLineTraceSingleByObject(LedgeTraceStart,LedgeTraceEnd);
	
	if(WalkableSurfaceHit.bBlockingHit && !LedgeTraceHit.bBlockingHit)
	{
		return true;
	}

	return false;
}

void UCustomMovementComponent::TryStartVaulting()
{
	FVector VaultStartPosition;
	FVector VaultEndPosition;
	
	if(CanStartVaulting(VaultStartPosition, VaultEndPosition))
	{
		SetMotionWarpTarget(FName("VaultStartPosition"), VaultStartPosition);
		SetMotionWarpTarget(FName("VaultEndPosition"), VaultEndPosition);
		StartClimbing();
		PlayClimbMontage(VaultMontage);
	}
}

bool UCustomMovementComponent::CanStartVaulting(FVector& OutVaultStartPosition, FVector& OutVaultEndPosition)
{
	if(IsFalling())
		return false;

	OutVaultStartPosition = FVector::ZeroVector;
	OutVaultEndPosition = FVector::ZeroVector;

	const FVector ComponentLocation = UpdatedComponent->GetComponentLocation();
	const FVector ComponentForward = UpdatedComponent->GetForwardVector();
	const FVector UpVector = UpdatedComponent->GetUpVector();
	const FVector DownVector = -UpVector;

	for (int32 i = 0; i < 5; i++)
	{
		const FVector Start = ComponentLocation + UpVector * 100.0f + ComponentForward * 80.0f * (i + 1);
		const FVector End = Start + DownVector * 100.0f * (i + 1);

		FHitResult VaultTraceHit = DoLineTraceSingleByObject(Start, End, false, false);

		if(VaultTraceHit.bBlockingHit)
		{
			if(i == 0)
				OutVaultStartPosition = VaultTraceHit.ImpactPoint;
			else if(i==3)
				OutVaultEndPosition = VaultTraceHit.ImpactPoint;
		}
	}

	if(OutVaultStartPosition != FVector::ZeroVector && OutVaultEndPosition != FVector::ZeroVector)
		return true;

	return false;
}

bool UCustomMovementComponent::CheckHasReachedFloor()
{
	const FVector DownVector = -UpdatedComponent->GetUpVector();
	const FVector StartOffset = DownVector * 50.0f;

	const FVector Start = UpdatedComponent->GetComponentLocation() + StartOffset;
	const FVector End = Start+DownVector;

	TArray<FHitResult> PossibleFloorHits = DoCapsuleTraceMultiByObject(Start,End);

	if(PossibleFloorHits.IsEmpty())
		return false;

	for (const FHitResult& PossibleFloorHit : PossibleFloorHits)
	{
		const bool bFloorReached = FVector::Parallel(-PossibleFloorHit.ImpactNormal, FVector::UpVector) && GetUnrotatedClimbVelocity().Z < -10.0f;

		if(bFloorReached)
			return true;
	}

	return false;
}

bool UCustomMovementComponent::CheckShouldStopClimbing()
{
	if(ClimbableSurfacesTracedResults.IsEmpty())
		return true;

	const float DotResult = FVector::DotProduct(CurrentClimbableSurfaceNormal, FVector::UpVector);
	const float DegreeDiff = FMath::RadiansToDegrees(FMath::Acos(DotResult));

	Debug::Print(TEXT("Degree Diff: ") + FString::SanitizeFloat(DegreeDiff), FColor::Cyan, 1);
	return DegreeDiff <= 60.0f;
}

bool UCustomMovementComponent::CheckHasReachedLedge()
{
	FHitResult LedgeHitResult = TraceFromEyeHeight( 100.0f, 50.0f);

	if(!LedgeHitResult.bBlockingHit)
	{
		const FVector WalkableSurfaceTraceStart = LedgeHitResult.TraceEnd;
		const FVector DownVector = -UpdatedComponent->GetUpVector();
		const FVector WalkableSurfaceTraceEnd = WalkableSurfaceTraceStart + DownVector * 100.0f;

		FHitResult WalkableSurfaceHitResult = DoLineTraceSingleByObject(WalkableSurfaceTraceStart, WalkableSurfaceTraceEnd);
		
		if(WalkableSurfaceHitResult.bBlockingHit && GetUnrotatedClimbVelocity().Z > 10.0f)
		{
			return true;
		}
	}
	
	return false;
}

void UCustomMovementComponent::StartClimbing()
{
	SetMovementMode(MOVE_Custom, ECustomMovementMode::MOVE_Climb);
}

void UCustomMovementComponent::StopClimbing()
{
	SetMovementMode(MOVE_Falling);
}

void UCustomMovementComponent::PhysClimb(float deltaTime, int32 Iterations)
{
	if (deltaTime < MIN_TICK_TIME)
	{
		return;
	}

	TraceClimbableSurfaces();
	ProcessClimbableSurfaceInfo();

	if(CheckHasReachedFloor() || CheckShouldStopClimbing())
	{
		StopClimbing();
	}
	
	RestorePreAdditiveRootMotionVelocity();

	if( !HasAnimRootMotion() && !CurrentRootMotion.HasOverrideVelocity() )
	{
		CalcVelocity(deltaTime, 0.0f, true, MaxBrakeClimbDeceleration);
	}

	ApplyRootMotionToVelocity(deltaTime);

	FVector OldLocation = UpdatedComponent->GetComponentLocation();
	const FVector Adjusted = Velocity * deltaTime;
	FHitResult Hit(1.f);
	SafeMoveUpdatedComponent(Adjusted, GetClimbRotation(deltaTime), true, Hit);

	if (Hit.Time < 1.f)
	{
		HandleImpact(Hit, deltaTime, Adjusted);
		SlideAlongSurface(Adjusted, (1.f - Hit.Time), Hit.Normal, Hit, true);
	}

	if(!HasAnimRootMotion() && !CurrentRootMotion.HasOverrideVelocity() )
	{
		Velocity = (UpdatedComponent->GetComponentLocation() - OldLocation) / deltaTime;
	}

	SnapMovementToClimbableSurfaces(deltaTime);
	
	if(CheckHasReachedLedge())
	{
		PlayClimbMontage(ClimbToTopMontage);	
	}
}

void UCustomMovementComponent::ProcessClimbableSurfaceInfo()
{
	CurrentClimbableSurfaceLocation = FVector::ZeroVector;
	CurrentClimbableSurfaceNormal = FVector::ZeroVector;

	if(ClimbableSurfacesTracedResults.IsEmpty())
		return;

	for (const FHitResult& TracedHitResult : ClimbableSurfacesTracedResults)
	{
		CurrentClimbableSurfaceLocation += TracedHitResult.ImpactPoint;
		CurrentClimbableSurfaceNormal += TracedHitResult.ImpactNormal;
	}

	CurrentClimbableSurfaceLocation /= ClimbableSurfacesTracedResults.Num();
	CurrentClimbableSurfaceNormal = CurrentClimbableSurfaceNormal.GetSafeNormal();
}


FQuat UCustomMovementComponent::GetClimbRotation(float DeltaTime)
{
	const FQuat CurrentQuat = UpdatedComponent->GetComponentQuat();

	if(HasAnimRootMotion() || CurrentRootMotion.HasOverrideVelocity())
		return CurrentQuat;

	const FQuat TargetQuat = FRotationMatrix::MakeFromX(-CurrentClimbableSurfaceNormal).ToQuat();
	return FMath::QInterpTo(CurrentQuat, TargetQuat, DeltaTime, 5.0f);
}

void UCustomMovementComponent::SnapMovementToClimbableSurfaces(float DeltaTime)
{
	const FVector ComponentForward = UpdatedComponent->GetForwardVector();
	const FVector ComponentLocation = UpdatedComponent->GetComponentLocation();

	const FVector ProjectedCharacterToSurface = (CurrentClimbableSurfaceLocation - ComponentLocation).ProjectOnTo(ComponentForward);
	const FVector SnapVector = -CurrentClimbableSurfaceNormal * ProjectedCharacterToSurface.Length();

	UpdatedComponent->MoveComponent(SnapVector * DeltaTime * MaxClimbSpeed,
									UpdatedComponent->GetComponentQuat(),
									true);
}

bool UCustomMovementComponent::IsClimbing() const
{
	return MovementMode == MOVE_Custom && CustomMovementMode == ECustomMovementMode::MOVE_Climb;
}

void UCustomMovementComponent::OnMovementModeChanged(EMovementMode PreviousMovementMode, uint8 PreviousCustomMode)
{
	if(IsClimbing())
	{
		bOrientRotationToMovement = false;
		CharacterOwner->GetCapsuleComponent()->SetCapsuleHalfHeight(48.0f);
		OnEnterClimbState.ExecuteIfBound();
	}
	
	if(PreviousMovementMode == MOVE_Custom && PreviousCustomMode == ECustomMovementMode::MOVE_Climb)
	{
		bOrientRotationToMovement = true;
		CharacterOwner->GetCapsuleComponent()->SetCapsuleHalfHeight(96.0f);

		const FRotator DirtyRotation = UpdatedComponent->GetComponentRotation();
		const FRotator CleanStandRotation = FRotator(0.0f, DirtyRotation.Yaw, 0.0f);
		UpdatedComponent->SetRelativeRotation(CleanStandRotation);
		
		StopMovementImmediately();

		OnExitClimbState.ExecuteIfBound();
	}
	
	Super::OnMovementModeChanged(PreviousMovementMode, PreviousCustomMode);
}

FVector UCustomMovementComponent::GetUnrotatedClimbVelocity() const
{
	return UKismetMathLibrary::Quat_UnrotateVector(UpdatedComponent->GetComponentQuat(), Velocity);
}