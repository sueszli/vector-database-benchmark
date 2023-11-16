// Fill out your copyright notice in the Description page of Project Settings.


#include "GunScript.h"

// Sets default values
AGunScript::AGunScript()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
	switch (BarrelType)
	{
	case EBarrelType::S_BarrelType_A: {
		BarrelLength = 10;
	}
	case EBarrelType::S_BarrelType_B: {
		BarrelLength = 12;
	}
	case EBarrelType::S_BarrelType_C: {
		BarrelLength = 15;
	}
	}
}

// Called when the game starts or when spawned
void AGunScript::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void AGunScript::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}