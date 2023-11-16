#include "teststruct.h"

FInventoryItem::FInventoryItem()
{
	this->ItemTypeId = -1;
	this->ItemName = "No Name";
	this->ItemWeight = 1;
	this->ItemMaxCount = 1;
	this->ItemDescription = "No Description";
}

FWeaponInfo::FWeaponInfo() {
	ItemTypeId = -1;
	Damage = 0;
	MaxAmmoCount = 0;
	AmmoType = -1;
	AttachmentType.Init(-1, 2);
}