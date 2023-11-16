// Copyright Epic Games, Inc. All Rights Reserved.

#include "InventorySystemCharacter.h"
#include "HeadMountedDisplayFunctionLibrary.h"
#include "Camera/CameraComponent.h"
#include "Components/CapsuleComponent.h"
#include "Components/InputComponent.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "GameFramework/Controller.h"
#include "Net/UnrealNetwork.h"
#include "GameFramework/SpringArmComponent.h"
#include "PickUps.h"
#include "MyBagItem_Weapon.h"
#include "MyPlayerController.h"

//////////////////////////////////////////////////////////////////////////
// AInventorySystemCharacter

AInventorySystemCharacter::AInventorySystemCharacter()
{
	// Set size for collision capsule
	GetCapsuleComponent()->InitCapsuleSize(42.f, 96.0f);

	bReplicates = true;
	// set our turn rates for input
	BaseTurnRate = 45.f;
	BaseLookUpRate = 45.f;

	// Don't rotate when the controller rotates. Let that just affect the camera.
	bUseControllerRotationPitch = false;
	bUseControllerRotationYaw = false;
	bUseControllerRotationRoll = false;

	// Configure character movement
	GetCharacterMovement()->bOrientRotationToMovement = true; // Character moves in the direction of input...	
	GetCharacterMovement()->RotationRate = FRotator(0.0f, 540.0f, 0.0f); // ...at this rotation rate
	GetCharacterMovement()->JumpZVelocity = 600.f;
	GetCharacterMovement()->AirControl = 0.2f;

	// Create a camera boom (pulls in towards the player if there is a collision)
	CameraBoom = CreateDefaultSubobject<USpringArmComponent>(TEXT("CameraBoom"));
	CameraBoom->SetupAttachment(RootComponent);
	CameraBoom->TargetArmLength = 300.0f; // The camera follows at this distance behind the character	
	CameraBoom->bUsePawnControlRotation = true; // Rotate the arm based on the controller

	// Create a follow camera
	FollowCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("FollowCamera"));
	FollowCamera->SetupAttachment(CameraBoom, USpringArmComponent::SocketName); // Attach the camera to the end of the boom and let the boom adjust to match the controller orientation
	FollowCamera->bUsePawnControlRotation = false; // Camera does not rotate relative to arm


	MaxHealth = 100;
	MaxEnergy = 100;
	Energy = 50;
	CurrentHealth = 50;
	CurrentWeaponId = -1;
	//CurrentWeaponId = NULL;
	Damage = 1;
	DamageMul = 1.0;
	EnergyMul = 1.0;
	// Note: The skeletal mesh and anim blueprint references on the Mesh component (inherited from Character) 
	// are set in the derived blueprint asset named MyCharacter (to avoid direct content references in C++)
}

//////////////////////////////////////////////////////////////////////////
// Input

void AInventorySystemCharacter::SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent)
{
	// Set up gameplay key bindings
	check(PlayerInputComponent);
	PlayerInputComponent->BindAction("Jump", IE_Pressed, this, &ACharacter::Jump);
	PlayerInputComponent->BindAction("Jump", IE_Released, this, &ACharacter::StopJumping);

	PlayerInputComponent->BindAxis("MoveForward", this, &AInventorySystemCharacter::MoveForward);
	PlayerInputComponent->BindAxis("MoveRight", this, &AInventorySystemCharacter::MoveRight);

	// We have 2 versions of the rotation bindings to handle different kinds of devices differently
	// "turn" handles devices that provide an absolute delta, such as a mouse.
	// "turnrate" is for devices that we choose to treat as a rate of change, such as an analog joystick
	PlayerInputComponent->BindAxis("Turn", this, &APawn::AddControllerYawInput);
	PlayerInputComponent->BindAxis("TurnRate", this, &AInventorySystemCharacter::TurnAtRate);
	PlayerInputComponent->BindAxis("LookUp", this, &APawn::AddControllerPitchInput);
	PlayerInputComponent->BindAxis("LookUpRate", this, &AInventorySystemCharacter::LookUpAtRate);

	// handle touch devices
	PlayerInputComponent->BindTouch(IE_Pressed, this, &AInventorySystemCharacter::TouchStarted);
	PlayerInputComponent->BindTouch(IE_Released, this, &AInventorySystemCharacter::TouchStopped);

	// VR headset functionality
	PlayerInputComponent->BindAction("ResetVR", IE_Pressed, this, &AInventorySystemCharacter::OnResetVR);
}


void AInventorySystemCharacter::OnResetVR()
{
	// If InventorySystem is added to a project via 'Add Feature' in the Unreal Editor the dependency on HeadMountedDisplay in InventorySystem.Build.cs is not automatically propagated
	// and a linker error will result.
	// You will need to either:
	//		Add "HeadMountedDisplay" to [YourProject].Build.cs PublicDependencyModuleNames in order to build successfully (appropriate if supporting VR).
	// or:
	//		Comment or delete the call to ResetOrientationAndPosition below (appropriate if not supporting VR)
	UHeadMountedDisplayFunctionLibrary::ResetOrientationAndPosition();
}

void AInventorySystemCharacter::TouchStarted(ETouchIndex::Type FingerIndex, FVector Location)
{
		Jump();
}

void AInventorySystemCharacter::TouchStopped(ETouchIndex::Type FingerIndex, FVector Location)
{
		StopJumping();
}

void AInventorySystemCharacter::TurnAtRate(float Rate)
{
	// calculate delta for this frame from the rate information
	AddControllerYawInput(Rate * BaseTurnRate * GetWorld()->GetDeltaSeconds());
}

void AInventorySystemCharacter::LookUpAtRate(float Rate)
{
	// calculate delta for this frame from the rate information
	AddControllerPitchInput(Rate * BaseLookUpRate * GetWorld()->GetDeltaSeconds());
}

void AInventorySystemCharacter::MoveForward(float Value)
{
	if ((Controller != nullptr) && (Value != 0.0f))
	{
		// find out which way is forward
		const FRotator Rotation = Controller->GetControlRotation();
		const FRotator YawRotation(0, Rotation.Yaw, 0);

		// get forward vector
		const FVector Direction = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::X);
		AddMovementInput(Direction, Value);
	}
}

void AInventorySystemCharacter::MoveRight(float Value)
{
	if ( (Controller != nullptr) && (Value != 0.0f) )
	{
		// find out which way is right
		const FRotator Rotation = Controller->GetControlRotation();
		const FRotator YawRotation(0, Rotation.Yaw, 0);
	
		// get right vector 
		const FVector Direction = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::Y);
		// add movement in that direction
		AddMovementInput(Direction, Value);
	}
}


/* -------------------------My Code-----------------------------*/

void AInventorySystemCharacter::GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& OutLifetimeProps)const {
	Super::GetLifetimeReplicatedProps(OutLifetimeProps);
	DOREPLIFETIME(AInventorySystemCharacter, CurrentHealth);
	DOREPLIFETIME(AInventorySystemCharacter, CurrentWeaponId);
	DOREPLIFETIME(AInventorySystemCharacter, Damage);
	DOREPLIFETIME(AInventorySystemCharacter, DamageMul);
	DOREPLIFETIME(AInventorySystemCharacter, Energy);
	DOREPLIFETIME(AInventorySystemCharacter, EnergyMul);
}


int32 AInventorySystemCharacter::GetCurrentWeapon() {
	return CurrentWeaponId;
}

void AInventorySystemCharacter::SetCurrentWeapon(int32 cw) {
	if (GetLocalRole() < ROLE_Authority) {
		ServerSetCurrentWeapon(cw);
	}
	else {
		CurrentWeaponId = cw;
	}
}

void AInventorySystemCharacter::ServerSetCurrentWeapon_Implementation(int32 cw) {
	SetCurrentWeapon(cw);
}



void AInventorySystemCharacter::GenerateNewPickUp_Implementation(int32 TypeId, int32 Count) {
	char dir[] = "/Game/BluePrints/BP_Item000.BP_Item000_C";
	int digitid[3] = { 0,0,0 };
	digitid[0] = TypeId / 100, digitid[1] = TypeId / 10 % 10, digitid[2] = TypeId % 10;
	for (int i = 0; i < 3; i++)
		dir[24 + i] = digitid[i] + '0', dir[35 + i] = digitid[i] + '0';
	FSoftClassPath softClassPath(/*TEXT(dir)*/dir);
	UClass* bpClass = softClassPath.TryLoadClass<AActor>();
	if (bpClass)
	{
		AActor* nactor = GetWorld()->SpawnActor<AActor>(bpClass);
		APickUps* newpickup = Cast<APickUps>(nactor);
		newpickup->SetItem(TypeId, Count);
		FVector myposition = GetActorLocation();
		myposition.Z = 0.0f;
		nactor->SetActorLocation(myposition + FVector(100.0f, 100.0f, 170.0f));
	}
}

void AInventorySystemCharacter::SetIntAttribute(AttriName AttName, int32 value) {
	if (GetLocalRole() < ROLE_Authority) {
		ServerSetIntAttribute(AttName, value);
	}
	else {
		switch (AttName)
		{
		case AttriName::Damage:
			Damage = value; break;
		case AttriName::Energy:
			Energy = value; break;
		case AttriName::Health:
			CurrentHealth = MaxHealth<value?MaxHealth:value; break;
		default:
			break;
		}
	}
}

void AInventorySystemCharacter::ServerSetIntAttribute_Implementation(AttriName AttName, int32 value) {
	SetIntAttribute(AttName, value);
}

void AInventorySystemCharacter::SetFloatAttribute(AttriName AttName, float value) {
	if (GetLocalRole() < ROLE_Authority) {
		ServerSetFloatAttribute(AttName, value);
	}
	else {
		switch (AttName)
		{
		case AttriName::DamageMul:
			DamageMul = value; break;
		case AttriName::EnergyMul:
			EnergyMul = value; break;
		default:
			break;
		}
	}
}

void AInventorySystemCharacter::ServerSetFloatAttribute_Implementation(AttriName AttName, float value) {
	SetFloatAttribute(AttName, value);
}

int32 AInventorySystemCharacter::GetIntAttribute(AttriName AttName) {
	switch (AttName)
	{
	case AttriName::Damage: {
		AMyPlayerController* pc = Cast<AMyPlayerController>(GetController());
		return pc->GetWeaponDamageById(CurrentWeaponId); 
		//return Damage;
		break;
	}
	case AttriName::Energy:
		return Energy; break;
	case AttriName::Health:
		return CurrentHealth; break;
	case AttriName::TrueDamage: {
		//return Damage * DamageMul; 
		AMyPlayerController* pc = Cast<AMyPlayerController>(GetController());
		return pc->GetWeaponDamageById(CurrentWeaponId) * DamageMul;
		break;
	}
	case AttriName::TrueEnergy:
		return Energy * EnergyMul; break;
	}
	return 0;
}

float AInventorySystemCharacter::GetFloatAttribute(AttriName AttName) {
	switch (AttName)
	{
	case AttriName::DamageMul:
		return DamageMul; break;
	case AttriName::EnergyMul:
		return EnergyMul; break;
	default:
		break;
	}
	return 0;
}