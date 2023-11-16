// Fill out your copyright notice in the Description page of Project Settings.


#include "FremenCharacter.h" 

#include "SoulsMeleeCombatSystem/Items/BaseWeapon.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "Kismet/GameplayStatics.h"
#include "Kismet/KismetSystemLibrary.h"
#include "NiagaraFunctionLibrary.h"
#include "MotionWarpingComponent.h"
#include "Items/Interactable.h"
#include "Utils/Logger.h"
#include "Components/CombatComponent.h"
#include "Components/FocusComponent.h"
#include "Components/RagdollComponent.h"
#include "Components/WidgetComponent.h"

// Sets default values
AFremenCharacter::AFremenCharacter()
{
 	// Set this character to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = false;
	
	// Rotation is determent by the camera and not by the controller.
	bUseControllerRotationPitch = false;
	bUseControllerRotationYaw = false;
	bUseControllerRotationRoll = false;

	// Configure character movement
	GetCharacterMovement()->bOrientRotationToMovement = true; // Character moves in the direction of input...	
	GetCharacterMovement()->RotationRate = FRotator(0.0f, 500.0f, 0.0f); // ...at this rotation rate

	CombatComponent = CreateDefaultSubobject<UCombatComponent>(TEXT("CombatComponent"));
	AddOwnedComponent(CombatComponent);

	RagdollComponent = CreateDefaultSubobject<URagdollComponent>(TEXT("RagdollComponent"));
	AddOwnedComponent(RagdollComponent);

	FocusComponent = CreateDefaultSubobject<UFocusComponent>(TEXT("FocusComponent"));
	AddOwnedComponent(FocusComponent);
	
	MotionWarpingComponent = CreateDefaultSubobject<UMotionWarpingComponent>(TEXT("MotionWarpingComponent"));
	AddOwnedComponent(MotionWarpingComponent);
	
	InFocusWidget = CreateDefaultSubobject<UWidgetComponent>(TEXT("InFocusWidget"));
	InFocusWidget->SetupAttachment(RootComponent);

	CharacterStateMachine = StateMachine(Idle);
}

void AFremenCharacter::InstallStateMachineHandlers()
{
	CharacterStateMachine.RegisterStateHandler(Idle, TSet{Attacking, Dodging, Disabled, GeneralAction});
	CharacterStateMachine.RegisterStateHandler(GeneralAction, TSet{Idle});
	CharacterStateMachine.RegisterStateHandler(Attacking, TSet{Idle});
	CharacterStateMachine.RegisterStateHandler(Dodging, TSet{Idle});
	CharacterStateMachine.RegisterStateHandler(Disabled, TSet{Idle, Attacking, Dodging, Dead, GeneralAction});
	CharacterStateMachine.RegisterStateHandler(Dead, TSet{Idle, Disabled, Attacking, Dodging, Dead, GeneralAction}, this, &AFremenCharacter::PerformDeath);
}

// Called when the game starts or when spawned
void AFremenCharacter::BeginPlay()
{
	Super::BeginPlay();
	
	InstallStateMachineHandlers();
	OnTakePointDamage.AddUniqueDynamic(this, &AFremenCharacter::OnReceivePointDamage);
	TrySpawnMainWeapon();
}

// Called to bind functionality to input
void AFremenCharacter::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
	Super::SetupPlayerInputComponent(PlayerInputComponent);
	
	PlayerInputComponent->BindAxis(TEXT("MoveForward"), this, &AFremenCharacter::MoveForward);
    PlayerInputComponent->BindAxis(TEXT("MoveRight"), this, &AFremenCharacter::MoveRight);
    PlayerInputComponent->BindAxis(TEXT("LookUp"), this, &AFremenCharacter::LookUp);
    PlayerInputComponent->BindAxis(TEXT("LookRight"), this, &AFremenCharacter::LookRight);

	PlayerInputComponent->BindAction(TEXT("ToggleWeapon"), IE_Pressed, this, &AFremenCharacter::ToggleWeapon);
	PlayerInputComponent->BindAction(TEXT("Attack"), IE_Released, this, &AFremenCharacter::Attack);
	PlayerInputComponent->BindAction(TEXT("Attack"), IE_Pressed, this, &AFremenCharacter::StartChargeAttack);
	PlayerInputComponent->BindAction(TEXT("HeavyAttack"), IE_Pressed, this, &AFremenCharacter::HeavyAttack);
	PlayerInputComponent->BindAction(TEXT("Interact"), IE_Pressed, this, &AFremenCharacter::Interact);
	PlayerInputComponent->BindAction(TEXT("Dodge"), IE_Pressed, this, &AFremenCharacter::Dodge);
	PlayerInputComponent->BindAction(TEXT("Focus"), IE_Pressed, this, &AFremenCharacter::Focus);
}

void AFremenCharacter::MoveForward(float AxisValue)
{
	if (Controller != nullptr && AxisValue != 0.0f)
	{
		// find out which way is forward
		const FRotator Rotation = Controller->GetControlRotation();
		const FRotator YawRotation(0, Rotation.Yaw, 0);

		// get forward vector
		const FVector Direction = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::X);
		AddMovementInput(Direction, AxisValue);
	}
}

void AFremenCharacter::MoveRight(float AxisValue)
{
	if ( (Controller != nullptr) && (AxisValue != 0.0f) )
	{
		// find out which way is right
		const FRotator Rotation = Controller->GetControlRotation();
		const FRotator YawRotation(0, Rotation.Yaw, 0);
	
		// get right vector 
		const FVector Direction = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::Y);
		// add movement in that direction
		AddMovementInput(Direction, AxisValue);
	}
}

void AFremenCharacter::LookUp(float AxisValue)
{
	AddControllerPitchInput(AxisValue * RotationRate * GetWorld()->GetDeltaSeconds());
}

void AFremenCharacter::LookRight(float AxisValue)
{
	AddControllerYawInput(AxisValue * RotationRate * GetWorld()->GetDeltaSeconds());
}

void AFremenCharacter::ToggleWeapon()
{
	Logger::Log(ELogLevel::INFO, __FUNCTION__);
	if (CharacterStateMachine.MoveToState(GeneralAction))
	{
		if (const ABaseWeapon* MainWeapon = CombatComponent->GetMainWeapon())
		{
			if (UAnimMontage* Montage = CombatComponent->IsCombatEnabled() ? MainWeapon->SheatheWeaponMontage : MainWeapon->DrawWeaponMontage)
			{
				Logger::Log(ELogLevel::INFO, FString::Printf(TEXT("play %s"), *Montage->GetName()));

				PlayAnimMontage(Montage);
			}
		}
		else
		{
			CharacterStateMachine.MoveToState(Idle);
		}
	}
	
}

void AFremenCharacter::Interact()
{
	Logger::Log(ELogLevel::INFO, __FUNCTION__);
	TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypesArray;
	ObjectTypesArray.Init(UEngineTypes::ConvertToObjectType(ECC_GameTraceChannel1), 1);

	TArray<AActor*> OutArray;
	TArray<AActor*> IgnoreActors;
	IgnoreActors.Init(this, 1);
	
	if (UKismetSystemLibrary::SphereOverlapActors(GetWorld(), GetActorLocation(), 500.f, ObjectTypesArray, nullptr, IgnoreActors, OutArray))
	{
		if (IInteractable* Item = Cast<IInteractable>(OutArray[0]))
		{
			Item->Interact(this);
			Logger::Log(ELogLevel::INFO, FString::Printf(TEXT("Interact with %s"), *OutArray[0]->GetName()));
		}
	}
}

void AFremenCharacter::Dodge()
{
	Logger::Log(ELogLevel::INFO, __FUNCTION__);
	
	if (CharacterStateMachine.MoveToState(Dodging))
	{
		PerformDodge();
	}
}

void AFremenCharacter::Focus()
{
	Logger::Log(ELogLevel::INFO, __FUNCTION__);
	if (CombatComponent->IsCombatEnabled())
	{
		FocusComponent->ToggleFocus();
	}
}

void AFremenCharacter::Attack()
{
	Logger::Log(ELogLevel::INFO, __FUNCTION__);
	if (CombatComponent && !CombatComponent->IsCombatEnabled())
	{
		return;
	}
	
	if (CharacterStateMachine.GetCurrentState() == Attacking )
	{
		CombatComponent->bIsAttackSaved = true;
		
	} else if (CharacterStateMachine.MoveToState(Attacking))
	{
		const EAttackType AttackType = bIsChargedAttackReady ? Charge : Light;
		PerformAttack(AttackType);
	}

	ClearChargeAttack();
}

void AFremenCharacter::ClearChargeAttack()
{
	Logger::Log(ELogLevel::INFO, __FUNCTION__);

	if (GetWorldTimerManager().IsTimerActive(ChargeAttackTimerHandle))
	{
		GetWorldTimerManager().ClearTimer(ChargeAttackTimerHandle);
	}
	
	bIsChargedAttackReady = false;
}

void AFremenCharacter::HeavyAttack()
{
	Logger::Log(ELogLevel::INFO, __FUNCTION__);
	
	ClearChargeAttack();
	
	if (CombatComponent && !CombatComponent->IsCombatEnabled())
	{
		return;
	}

	if (CharacterStateMachine.GetCurrentState() == Attacking )
	{
		return;
	}

	if (CharacterStateMachine.MoveToState(Attacking))
	{
		PerformAttack(Heavy, false);
	}
}

void AFremenCharacter::StartChargeAttack()
{
	FTimerDelegate SetChargeAttackReadyCallback;
	SetChargeAttackReadyCallback.BindLambda([this]
	{
		Logger::Log(ELogLevel::INFO, "Charged Attack Ready!");
		bIsChargedAttackReady = true;
	});

	GetWorldTimerManager().SetTimer(ChargeAttackTimerHandle, SetChargeAttackReadyCallback, ChargeAttackDuration, false);
}

void AFremenCharacter::AttackContinue()
{
	if (!CombatComponent)
	{
		return;
	}

	CharacterStateMachine.MoveToState(Idle);
	CombatComponent->bIsAttacking = false;
	
	if (CombatComponent->bIsAttackSaved)
	{
		CombatComponent->bIsAttackSaved = false;
		if (CombatComponent->IsCombatEnabled())
		{
			PerformAttack(Light);
		}
	}
}

void AFremenCharacter::ResetMovementState()
{
	CharacterStateMachine.MoveToState(Idle);
	CombatComponent->ResetCombat();
}

bool AFremenCharacter::CanReceiveDamage()
{
	return CharacterStateMachine.GetCurrentState() != Dead;
}

bool AFremenCharacter::CanBeFocused()
{
	return CharacterStateMachine.GetCurrentState() != Dead;
}

void AFremenCharacter::OnFocused(bool bIsFocused)
{
	InFocusWidget->SetVisibility(bIsFocused);
}

FRotator AFremenCharacter::GetSignificantInputRotation(float Threshold)
{
	const FVector LastInput = GetLastMovementInputVector();
	if (LastInput.Length() >= Threshold)
	{
		return FRotationMatrix::MakeFromX(LastInput).Rotator();
	}

	return GetActorRotation();
}

void AFremenCharacter::ApplyMotionWarping(FName WarpTargetName) const
{
	const AActor* Target = FocusComponent->ActorInFocus;
		
	if (Target && GetDistanceTo(Target) < MinWarpingDistance)
	{
		// calc unit vector in the direction from target to character, multiplied by the offset distance
		auto OffsetFromTarget = GetActorLocation() - Target->GetActorLocation(); 
		OffsetFromTarget.Normalize();
		OffsetFromTarget *= WarpingTargetOffsetFactor;	
			
		MotionWarpingComponent->AddOrUpdateWarpTargetFromLocation(WarpTargetName, Target->GetActorLocation() + OffsetFromTarget);
	}
	else
	{
		MotionWarpingComponent->RemoveWarpTarget(WarpTargetName);
	}
}

void AFremenCharacter::PerformAttack(EAttackType AttackType ,bool IsRandom)
{
	TArray<UAnimMontage*> MontagesArray = CombatComponent->GetMainWeapon()->GetAttackMontages(AttackType);
	const int Index = IsRandom ? FMath::RandRange(0, MontagesArray.Num()) : CombatComponent->AttackCount;

	if (Index < MontagesArray.Num())
	{
		UAnimMontage* Montage = MontagesArray[Index];
		CombatComponent->bIsAttacking = true;
		CombatComponent->AttackCount ++;
		CombatComponent->AttackCount %= MontagesArray.Num();

		ApplyMotionWarping(TEXT("Attack"));
		PlayAnimMontage(Montage);
	}
}

void AFremenCharacter::PerformDodge()
{
	PlayAnimMontage(DodgeMontage);
}

void AFremenCharacter::PerformDeath()
{
	Logger::Log(ELogLevel::DEBUG, __FUNCTION__);
	
	RagdollComponent->EnableRagdoll();

	// Add impact velocity
	constexpr float InitialSpeed = 2000.f;
	const FVector HitVelocity = GetActorForwardVector() * InitialSpeed * -1;
	GetMesh()->SetPhysicsLinearVelocity(HitVelocity, false, TEXT("Pelvis"));

	// Apply physics on main weapon
	if (ABaseWeapon* MainWeapon = CombatComponent->GetMainWeapon())
	{
		MainWeapon->GetItemMesh()->SetCollisionProfileName(TEXT("PhysicsActor"), true);
		MainWeapon->GetItemMesh()->SetCollisionEnabled(ECollisionEnabled::PhysicsOnly);
		MainWeapon->GetItemMesh()->SetSimulatePhysics(true);
	}

	// Destroy character and weapon after delay time ends
	FTimerHandle DeathTimer;
	GetWorldTimerManager().SetTimer(DeathTimer, this, &AFremenCharacter::DestroyCharacter, 5.f, false);
}

void AFremenCharacter::DestroyCharacter()
{
	if (ABaseWeapon* MainWeapon = CombatComponent->GetMainWeapon())
	{
		MainWeapon->Destroy();
	}
	
	this->Destroy();
}

void AFremenCharacter::OnReceivePointDamage(AActor* DamagedActor, float Damage, AController* InstigatedBy,
                                            FVector HitLocation, UPrimitiveComponent* FHitComponent, FName BoneName, FVector ShotFromDirection,
                                            const UDamageType* DamageType, AActor* DamageCauser)
{
	Logger::Log(ELogLevel::DEBUG, __FUNCTION__);

	UGameplayStatics::PlaySoundAtLocation(GetWorld(),HitCue, HitLocation);
	UNiagaraFunctionLibrary::SpawnSystemAtLocation(GetWorld(), BloodEmitter, HitLocation, ShotFromDirection.Rotation());
	PlayAnimMontage(HitMontage);
	CharacterStateMachine.MoveToState(Disabled);
	Health -= Damage;
	
	if (Health <= 0)
	{
		CharacterStateMachine.MoveToState(Dead);
	}
}

void AFremenCharacter::TrySpawnMainWeapon()
{
	if (WeaponClass != nullptr)
	{
		if (UWorld* World = GetWorld())
		{
			FActorSpawnParameters SpawnParameters;
			
			SpawnParameters.Owner = this;
			SpawnParameters.Instigator = this;

			if (ABaseWeapon* Weapon = World->SpawnActor<ABaseWeapon>(WeaponClass, GetActorTransform(), SpawnParameters))
			{
				CombatComponent->SetMainWeapon(Weapon);
			}
		}
	}
}

