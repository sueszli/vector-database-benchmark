// Fill out your copyright notice in the Description page of Project Settings.


#include "BaseCharacter.h"

// Sets default values
ABaseCharacter::ABaseCharacter()
{
 	// Set this character to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

// Called when the game starts or when spawned
void ABaseCharacter::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void ABaseCharacter::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

// Called to bind functionality to input
void ABaseCharacter::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
	Super::SetupPlayerInputComponent(PlayerInputComponent);

}

// Implement Calculate Health
void ABaseCharacter::calculateHealth(int i_minusDamage)
{
	if (i_minusDamage >= 0)
	{
		if (i_minusDamage >= i_health)
		{
			i_health= 0;
		}
		else { i_health-= i_minusDamage; }
	}
	calculateDead();
	calculateCanAffectHealth();
}

// Implement Calculate Dead
void ABaseCharacter::calculateDead()
{
	if (i_health <= 0) 
	{
		b_isDead= true;
	}
	else { b_isDead= false; }
}

void ABaseCharacter::calculateCanAffectHealth()
{
	if (b_isDead == true) 
	{
		b_canAffectHealth= false; 
		malfeasanceExplosions();
	}
	else { b_canAffectHealth= true; }
}

void ABaseCharacter::addMalfeasanceStack()
{
	if (i_stacksOfMalfeasance < 5)
	{
		//add a stack of malfeasance
		i_stacksOfMalfeasance++;

		//if more than 5 stack, explode the stacks
		if (i_stacksOfMalfeasance >= 5)
		{
			if (b_isDead == false)
			{
				malfeasanceExplosions();
			}
		}
	}
}

#if WITH_EDITOR
// Implement editor only code used when changing values
void ABaseCharacter::PostEditChangeProperty(FPropertyChangedEvent &PropertyChangedEvent)
{
	b_isDead= false;
	i_health= 100;

	Super::PostEditChangeProperty(PropertyChangedEvent);

	calculateDead();
}
#endif //WITH_EDITOR

