// Copyright Epic Games, Inc. All Rights Reserved.

#include "BallisticsSystemProjectile.h"
#include "GameFramework/ProjectileMovementComponent.h"
#include "Components/SphereComponent.h"

ABallisticsSystemProjectile::ABallisticsSystemProjectile()
{
	// Use a sphere as a simple collision representation
	CollisionComp = CreateDefaultSubobject<USphereComponent>(TEXT("SphereComp"));
	CollisionComp->InitSphereRadius(5.0f);
	CollisionComp->BodyInstance.SetCollisionProfileName("Projectile");
	CollisionComp->OnComponentHit.AddDynamic(this, &ABallisticsSystemProjectile::OnHit); // set up a notification for when this component hits something blocking

	// Players can't walk on it
	CollisionComp->SetWalkableSlopeOverride(FWalkableSlopeOverride(WalkableSlope_Unwalkable, 0.f));
	CollisionComp->CanCharacterStepUpOn = ECB_No;

	// Set as root component
	RootComponent = CollisionComp;

	// Use a ProjectileMovementComponent to govern this projectile's movement
	ProjectileMovement = CreateDefaultSubobject<UProjectileMovementComponent>(TEXT("ProjectileComp"));
	ProjectileMovement->UpdatedComponent = CollisionComp;
	ProjectileMovement->InitialSpeed = 3000.f;
	ProjectileMovement->MaxSpeed = 3000.f;
	ProjectileMovement->bRotationFollowsVelocity = true;
	ProjectileMovement->bShouldBounce = true;

	// Die after 3 seconds by default
	InitialLifeSpan = 30.0f;
}

void ABallisticsSystemProjectile::OnHit(UPrimitiveComponent *HitComp, AActor *OtherActor, UPrimitiveComponent *OtherComp, FVector NormalImpulse, const FHitResult &Hit)
{
	// Only add impulse and destroy projectile if we hit a physics
	if ((OtherActor != nullptr) && (OtherActor != this) && (OtherComp != nullptr) && OtherComp->IsSimulatingPhysics())
	{
		OtherComp->AddImpulseAtLocation(GetVelocity() * 100.0f, GetActorLocation());

		Destroy();
	}
}

float ABallisticsSystemProjectile::BulletCoefficientCalculator(float mass, float drag, float crossSection)
{
	float temp = crossSection * drag;
	UE_LOG(LogTemp, Warning, TEXT("%f"), temp);
	return mass / temp;
}

void ABallisticsSystemProjectile::Initialize(EBulletCaliber BulletCaliber)
{
	switch (BulletCaliber)
	{
		case EBulletCaliber::S_Caliber_556mm:
		{
			Mass = 0.02;
			crossSectionArea = 0.015;
			dragCoefficient = 0.05;
		}
		case EBulletCaliber::S_Caliber_762mm:
		{
			Mass = 0.0181;
			crossSectionArea = 0.176;
			dragCoefficient = 0.05;
		}
		case EBulletCaliber::S_Caliber_9mm:
		{
			Mass = 0.012;
			crossSectionArea = 0.02;
			dragCoefficient = 0.05;
		}
		case EBulletCaliber::S_PaintBall:
		{
			Mass = 0.007;
			crossSectionArea = 0.363;
			dragCoefficient = 0.47f;
		}
	}
	BulletCoefficient = BulletCoefficientCalculator(Mass, dragCoefficient, crossSectionArea);
}