// Fill out your copyright notice in the Description page of Project Settings.


#include "MouseDeltaValues.h"

void AMouseDeltaValues::BeginPlay()
{
}

void AMouseDeltaValues::Tick(float DeltaTime)
{
	GetInputMouseDelta(mouseDeltaX, mouseDeltaY);
}
