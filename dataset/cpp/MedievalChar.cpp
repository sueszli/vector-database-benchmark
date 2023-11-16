// Fill out your copyright notice in the Description page of Project Settings.
#include "MedievalChar.h"
#include "Camera/CameraComponent.h"
#include "Components/CapsuleComponent.h"
#include "Net/UnrealNetwork.h"
#include "Kismet/KismetMathLibrary.h"

// Sets default values
AMedievalChar::AMedievalChar()
{
	GetCapsuleComponent()->InitCapsuleSize(55.f, 96.0f);
 	// Set this character to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
	FirstPersonCameraComponent = CreateDefaultSubobject<UCameraComponent>(TEXT("MainFPSCamera"));
	isFighting = false;
	isAttacking = false;
	isBlocking = false;
	Mesh3P = CreateDefaultSubobject<USkeletalMeshComponent>(TEXT("Mesh3P"));
	Mesh3P->SetupAttachment(GetCapsuleComponent());
	FirstPersonCameraComponent->SetupAttachment(Mesh3P);

}

// Called when the game starts or when spawned
void AMedievalChar::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void AMedievalChar::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

// Called to bind functionality to input
void AMedievalChar::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
	Super::SetupPlayerInputComponent(PlayerInputComponent);
	PlayerInputComponent->BindAxis("Turn", this, &APawn::AddControllerYawInput);
	PlayerInputComponent->BindAxis("MoveRight", this, &AMedievalChar::MoveRight);
	PlayerInputComponent->BindAxis("MoveForward", this, &AMedievalChar::MoveForward);
	PlayerInputComponent->BindAxis("LookUp", this, &AMedievalChar::LookUp);
	PlayerInputComponent->BindAction("Fight", IE_Pressed, this, &AMedievalChar::FightStance);
	PlayerInputComponent->BindAction("Punch", IE_Pressed, this, &AMedievalChar::Punch);
	PlayerInputComponent->BindAction("Attack", IE_Pressed, this, &AMedievalChar::Attack);
	PlayerInputComponent->BindAction("Attack", IE_Released, this, &AMedievalChar::Attack);
	PlayerInputComponent->BindAction("Block", IE_Pressed, this, &AMedievalChar::Block);
	PlayerInputComponent->BindAction("Block", IE_Released, this, &AMedievalChar::Block);

}

void AMedievalChar::MoveForward(float val) {
	if (val != 0.0) {
		AddMovementInput(GetActorForwardVector(), val);
	}
}

void AMedievalChar::MoveRight(float val) {
	if (val != 0.0) {
		AddMovementInput(GetActorRightVector(), val);
	}
}

void AMedievalChar::FightStance() {
	isFighting = !isFighting;
}

void AMedievalChar::Attack() {
	isAttacking = !isAttacking; 
	ServerAttack();
}

void AMedievalChar::Block() {
	isBlocking = !isBlocking;
	ServerBlock();
}

void AMedievalChar::Punch() {
	UAnimInstance* animInstance =  Mesh3P->GetAnimInstance();
	if (animInstance != nullptr) {
		if (animInstance->GetCurrentActiveMontage() != punchAnim && isFighting) {
			animInstance->Montage_Play(punchAnim, 1.f);
		}
	}
}

void AMedievalChar::LookUp(float val)
{
	APawn::AddControllerPitchInput(val);
	ServerLookUp(val);
}

void AMedievalChar::ServerLookUp_Implementation(float val)
{
	pitch = UKismetMathLibrary::NormalizeAxis(GetControlRotation().Pitch - GetActorRotation().Pitch);
}

void AMedievalChar::ServerAttack_Implementation()
{
	isAttacking = !isAttacking;
}

void AMedievalChar::ServerBlock_Implementation()
{
	isBlocking = !isBlocking;
}


void AMedievalChar::GetLifetimeReplicatedProps(TArray< FLifetimeProperty >& OutLifetimeProps) const {
	Super::GetLifetimeReplicatedProps(OutLifetimeProps);
	DOREPLIFETIME_CONDITION(AMedievalChar, pitch, COND_SkipOwner);
	DOREPLIFETIME_CONDITION(AMedievalChar, isAttacking, COND_SkipOwner);
	DOREPLIFETIME_CONDITION(AMedievalChar, isBlocking, COND_SkipOwner);
}