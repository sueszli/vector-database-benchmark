// Fill out your copyright notice in the Description page of Project Settings.
#include "OneHandedAIS.h"
#include "MedievalChar.h"
#include "Net/UnrealNetwork.h"
#include "Kismet/KismetMathLibrary.h"

UOneHandedAIS::UOneHandedAIS(){

	isAttacking = false;
	isRunning = false;
	isBlocking = false;
	aimPitch = 0;
	aimYaw = 0;
	newPitch = 0;
	pawnOwner = nullptr;
}

void UOneHandedAIS::NativeUpdateAnimation(float DeltaSeconds) {
	Super::NativeUpdateAnimation(DeltaSeconds);
	
	if (pawnOwner == nullptr) {
		return;
	}

	if (pawnOwner->IsA(AMedievalChar::StaticClass())) {
		AMedievalChar* ownerChar = Cast<AMedievalChar>(pawnOwner);
		
		isAttacking = ownerChar->isAttacking;
		isBlocking = ownerChar->isBlocking;
		if (ownerChar->IsLocallyControlled()) {
			newPitch = UKismetMathLibrary::NormalizeAxis(ownerChar->GetControlRotation().Pitch - ownerChar->GetActorRotation().Pitch);
		}
		else {
			newPitch = ownerChar->pitch;
		}
		//Running Logic
		if (ownerChar->GetVelocity().Size() > 1.0) {
			isRunning = true;
		}
		else { 
			isRunning = false; 
		}

		//AO Values
		/*if (ownerChar->GetControlRotation().Pitch >= 180.0) {
			aimPitch = ownerChar->GetControlRotation().Pitch - 360;
		}
		else {
			aimPitch = ownerChar->GetControlRotation().Pitch;
		}

		if (ownerChar->GetControlRotation().Yaw >= 90) {
			aimYaw = ownerChar->GetControlRotation().Yaw - 360;
		}
		else {
			aimYaw = ownerChar->GetControlRotation().Yaw;
		}*/
	}

}

void UOneHandedAIS::NativeInitializeAnimation() {
	Super::NativeInitializeAnimation();
	pawnOwner = TryGetPawnOwner();
}

void UOneHandedAIS::GetLifetimeReplicatedProps(TArray< FLifetimeProperty >& OutLifetimeProps) const {
	Super::GetLifetimeReplicatedProps(OutLifetimeProps);
	DOREPLIFETIME(UOneHandedAIS, aimPitch);
}