// Fill out your copyright notice in the Description page of Project Settings.

#include "Kismet/GameplayStatics.h"
#include "BallisticsSystemProjectile.h"
#include "Animation/AnimInstance.h"
#include "ThirdPersonController.h"
#include "Camera/CameraComponent.h"
#include "GameFramework/PlayerController.h"
#include "GameFramework/SpringArmComponent.h"
#include "GameFramework/CharacterMovementComponent.h"

// Sets default values
AThirdPersonController::AThirdPersonController()
{
 	// Set this character to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	skeletalMesh = CreateDefaultSubobject<USkeletalMeshComponent>(TEXT("SkeletalMesh"));
	skeletalMesh->SetupAttachment(RootComponent);

	bUseControllerRotationPitch = false;
	bUseControllerRotationYaw = false;
	bUseControllerRotationRoll = false;

	GetCharacterMovement()->bOrientRotationToMovement = true;
	GetCharacterMovement()->RotationRate = FRotator(0, 540, 0);

	FP_Gun = CreateDefaultSubobject<USkeletalMeshComponent>(TEXT("FP_Gun"));
	FP_Gun->SetOnlyOwnerSee(false);
	FP_Gun->bCastDynamicShadow = false;
	FP_Gun->CastShadow = false;
	// FP_Gun->SetupAttachment(Mesh1P, TEXT("GripPoint"));
	FP_Gun->SetupAttachment(RootComponent);

	//FP_MuzzleLocation = CreateDefaultSubobject<USceneComponent>(TEXT("MuzzleLocation"));
	//FP_MuzzleLocation->SetupAttachment(FP_Gun);
	//FP_MuzzleLocation->SetRelativeLocation(FVector(0.2f, 48.4f, -10.6f));

	turnRate = 45;
	lookRate = 45;
}

// Called when the game starts or when spawned
void AThirdPersonController::BeginPlay()
{
	Super::BeginPlay();
	FP_Gun->AttachToComponent(skeletalMesh, FAttachmentTransformRules(EAttachmentRule::SnapToTarget, true), TEXT("GripPoint"));
}

// Called every frame
void AThirdPersonController::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
}

// Called to bind functionality to input
void AThirdPersonController::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
	check(PlayerInputComponent);

	PlayerInputComponent->BindAxis("MoveForward", this, &AThirdPersonController::MoveForward);
	PlayerInputComponent->BindAxis("MoveRight", this, &AThirdPersonController::MoveRight);

	PlayerInputComponent->BindAxis("TurnRate", this, &AThirdPersonController::TurnRate);
	//PlayerInputComponent->BindAxis("LookUpRate", this, &AThirdPersonController::LookRate);

	PlayerInputComponent->BindAction("Jump", IE_Pressed, this, &ACharacter::Jump);
	PlayerInputComponent->BindAction("Jump", IE_Released, this, &ACharacter::StopJumping);
}


void AThirdPersonController::MoveForward(float value)
{
	/*forwardAxisValue = value;
	if (FMath::Abs(value) > 0.5f)
	{
		AddMovementInput(GetActorForwardVector(), value);
	}*/
}

void AThirdPersonController::MoveRight(float value)
{
	/*rightAxisValue = value;
	if (FMath::Abs(value) > 0.5f)
	{
		AddMovementInput(GetActorRightVector(), value);
	}*/
}

void AThirdPersonController::TurnRate(float Rate)
{
	/*UE_LOG(LogTemp, Warning, TEXT("%f"), &Rate);
	if(FMath::Abs(Rate)>0.5f)
		AddActorLocalRotation(FRotator(0, Rate, 0));*/
}

void AThirdPersonController::LookRate(float Rate)
{
/*	if (FMath::Abs(Rate) > 0.5f)
		AddControllerPitchInput(Rate * GetWorld()->GetDeltaSeconds() * lookRate);?*/
}