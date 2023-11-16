// Fill out your copyright notice in the Description page of Project Settings.


#include "ItemInfoTable.h"

UItemInfo::UItemInfo() {

}

UItemInfoTable::UItemInfoTable() {

}

UItemInfo* UItemInfoTable::GetItemInfoById(int32 id) {
	if (infotable.Contains(id))return infotable[id];
	return NULL;
}

FWeaponInfo UItemInfoTable::GetWeaponInfoById(int32 id) {
	if (weapontable.Contains(id)) return weapontable[id];
	else return FWeaponInfo();
}