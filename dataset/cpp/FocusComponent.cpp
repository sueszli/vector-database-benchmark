// Fill out your copyright notice in the Description page of Project Settings.


#include "Components/FocusComponent.h"

#include "CombatComponent.h"
#include "Camera/CameraComponent.h"
#include "Characters/Focusable.h"
#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "Kismet/GameplayStatics.h"
#include "Kismet/KismetMathLibrary.h"
#include "Kismet/KismetSystemLibrary.h"
#include "Utils/Logger.h"

// Sets default values for this component's properties
UFocusComponent::UFocusComponent()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = true;

	// ...
}


// Called when the game starts
void UFocusComponent::BeginPlay()
{
	Super::BeginPlay();
	
	SetComponentTickEnabled(false);
	OwnerCharacter = Cast<ACharacter>(GetOwner());
	if (OwnerCharacter != nullptr)
	{
		OwnerController = OwnerCharacter->GetController();
		if (UActorComponent* Actor = OwnerCharacter->GetComponentByClass(UCameraComponent::StaticClass()))
		{
			FollowCamera = Cast<UCameraComponent>(Actor);
		}
	}
	else
	{
		Logger::Log(ELogLevel::ERROR, "Casting component owner to ACharacter has failed!");
	}
}


// Called every frame
void UFocusComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	if (bIsEndOfFocusTransition)
	{
		const FRotator CurrentRotation = OwnerController->GetControlRotation();
		const FRotator DesiredRotation(0.f, CurrentRotation.Yaw ,CurrentRotation.Roll);

		// Interpolate rotation according to speed
		const auto InterpolatedRotation = UKismetMathLibrary::RInterpTo(CurrentRotation, DesiredRotation, UGameplayStatics::GetWorldDeltaSeconds(GetWorld()), Speed) ;
			
		// Set rotation values
		const float Pitch = InterpolatedRotation.Pitch;
		const float Yaw = CurrentRotation.Yaw;
		const float Roll = CurrentRotation.Roll;

		// Set owner rotation
		OwnerController->SetControlRotation(FRotator(Pitch, Yaw, Roll));
	}
	
	if (!bIsInFocus)
	{
		return;
	}
	
	UpdateFocus();
}

void UFocusComponent::ToggleFocus()
{
	if (bIsInFocus)
	{
		bIsInFocus = false;
		bIsEndOfFocusTransition = true;
		FocusTarget->OnFocused(false);
		FocusTarget = nullptr;
		ActorInFocus = nullptr;
		SetRotationMode(OrientToMovement);
		FTimerDelegate ChangeCameraPitchDelegate;
		ChangeCameraPitchDelegate.BindLambda([this]
		{
			SetComponentTickEnabled(false);
			bIsEndOfFocusTransition = false;
			GetWorld()->GetTimerManager().ClearTimer(EndOfFocusTimerHandle);
		});
		GetWorld()->GetTimerManager().SetTimer(EndOfFocusTimerHandle, ChangeCameraPitchDelegate, 1, false);
	}
	else
	{
		Focus();
	}
}


void UFocusComponent::Focus()
{
	IFocusable* OutFocusable;
	if (FindTarget(&OutFocusable) && OutFocusable->CanBeFocused())
	{
		bIsInFocus = true;
		bIsEndOfFocusTransition = false;
		FocusTarget = OutFocusable;
		ActorInFocus = ActorInFocus = Cast<AActor>(OutFocusable);
		UpdateOwnerRotationMode();
		SetComponentTickEnabled(true);
		OutFocusable->OnFocused(true);
	}
}

void UFocusComponent::ChangeRotation() const
{
	constexpr int PitchModifier = 100;
	const FVector SourceLocation = GetOwner()->GetActorLocation();
	const FVector TargetLocation = ActorInFocus->GetActorLocation() - FVector(0, 0, PitchModifier);
			
	// Calculate the rotation from the source location to the target location
	const FRotator LockAtRotation = UKismetMathLibrary::FindLookAtRotation(SourceLocation, TargetLocation);

	// Interpolate rotation according to speed
	const auto InterpolatedRotation = UKismetMathLibrary::RInterpTo(OwnerController->GetControlRotation(), LockAtRotation, UGameplayStatics::GetWorldDeltaSeconds(GetWorld()), Speed) ;
			
	// Set rotation values
	const float Pitch = InterpolatedRotation.Pitch;
	const float Yaw = InterpolatedRotation.Yaw;
	const float Roll = OwnerController->GetControlRotation().Roll;;

	// Set owner rotation
	OwnerController->SetControlRotation(FRotator(Pitch, Yaw, Roll));
}

void UFocusComponent::UpdateFocus()
{
	if (OwnerCharacter && OwnerController && ActorInFocus)
	{
		const float CurrentDistance = GetOwner()->GetDistanceTo(ActorInFocus);
		if (FocusTarget->CanBeFocused() && CurrentDistance < FocusDistance)
		{
			ChangeRotation();
			return;
		}
	}
	
	ToggleFocus(); // Toggle focus off
}

bool UFocusComponent::FindTarget(IFocusable** OutFocusable) const
{
	FVector StartLocation = GetOwner()->GetActorLocation();
	FVector EndLocation = StartLocation + FollowCamera->GetForwardVector() * FocusDistance;
	float Radius = 100.0f; 
	TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes = {UEngineTypes::ConvertToObjectType(ECC_Pawn)};
	TArray<AActor*> ActorsToIgnore = {GetOwner()};
	FHitResult OutHit; 

	if (UKismetSystemLibrary::SphereTraceSingleForObjects(GetWorld(), StartLocation, EndLocation, Radius, ObjectTypes,
		false, ActorsToIgnore, DebugTrace,OutHit,true)
		&& OutHit.GetActor() != nullptr)
	{
		if (IFocusable* Focusable = Cast<IFocusable>(OutHit.GetActor()))
		{
			*OutFocusable = Focusable;
			return true;
		}
	}

	// Sphere trace didn't hit anything that implements IFocusable
	return false;
}

void UFocusComponent::SetRotationMode(ERelativeOrientation OrientTo) const
{
	switch (OrientTo)
	{
		case OrientToCamera:
			OwnerCharacter->bUseControllerRotationYaw = true;
			OwnerCharacter->GetCharacterMovement()->bUseControllerDesiredRotation = true;
			OwnerCharacter->GetCharacterMovement()->bOrientRotationToMovement = false;
			break;
		
		case OrientToMovement:
			OwnerCharacter->bUseControllerRotationYaw = false;
			OwnerCharacter->GetCharacterMovement()->bUseControllerDesiredRotation = false;
			OwnerCharacter->GetCharacterMovement()->bOrientRotationToMovement = true;
			break;
	}
}

void UFocusComponent::UpdateOwnerRotationMode() const
{
	const auto CombatComponent = Cast<UCombatComponent>(OwnerCharacter->GetComponentByClass(UCombatComponent::StaticClass()));
	if (CombatComponent && CombatComponent->IsCombatEnabled() && bIsInFocus)
	{
		SetRotationMode(OrientToCamera);
	}
	else
	{
		SetRotationMode(OrientToMovement);
	}
}

