// Fill out your copyright notice in the Description page of Project Settings.


#include "Equippable.h"

void IEquippable::OnEquipped()
{
	SetIsEquipped(true);
}

void IEquippable::OnUnequipped()
{
	SetIsEquipped(false);
}

void IEquippable::SetIsEquipped(bool IsEquipped)
{
	bIsEquipped = IsEquipped;
}

// Add default functionality here for any IIEquippable functions that are not pure virtual.
bool IEquippable::IsEquipped()
{
	return bIsEquipped;
}
