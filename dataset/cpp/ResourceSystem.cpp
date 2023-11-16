// Fill out your copyright notice in the Description page of Project Settings.

#include "ResourceSystem.h"
#include "Kismet/GameplayStatics.h"


// Sets default values for this component's properties
 UResourceSystem::UResourceSystem()
{

    PrimaryComponentTick.bCanEverTick = false;

}

void UResourceSystem::GetResourceFromDataTable(FName ResourceRowName)
{
    if (!ResourceDataTable)
    {
        UE_LOG(LogTemp, Warning, TEXT("ResourceDataTable is not set."));
    }

    FBuildResource* Row = ResourceDataTable->FindRow<FBuildResource>(ResourceRowName, TEXT("Lookup Build Resource"));
      
    if (!Row)
    {
        UE_LOG(LogTemp, Warning, TEXT("Resource row not found: %s"), *ResourceRowName.ToString()); Resource = *Row;
    }

    Resource = *Row;
}

FBuildResource UResourceSystem::GetResourceData(FName ResourceRowName) const
{
    const FBuildResource* CachedResource = CachedResources.Find(ResourceRowName);
    if (CachedResource)
    {
        return *CachedResource;
    }

    FBuildResource Result;
    if (!ResourceDataTable)
    {
        UE_LOG(LogTemp, Warning, TEXT("ResourceDataTable is not set."));
        return Result;
    }

    FBuildResource* Row = ResourceDataTable->FindRow<FBuildResource>(ResourceRowName, TEXT("Lookup Build Resource"));

    if (!Row)
    {
        UE_LOG(LogTemp, Warning, TEXT("Resource row not found: %s"), *ResourceRowName.ToString());
        return Result;
    }

    Result = *Row;
    const_cast<UResourceSystem*>(this)->CachedResources.Add(ResourceRowName, Result);
    return Result;
}


void UResourceSystem::SavePlayerResources()
{
    UResourceSave* SaveGameInstance = Cast<UResourceSave>(UGameplayStatics::CreateSaveGameObject(UResourceSave::StaticClass()));


    if (SaveGameInstance)
    {
        SaveGameInstance->OwnedResources = OwnedResources;
        SaveGameInstance->SaveSlotName = SaveSlotName;
        SaveGameInstance->UserIndex = UserIndex;

        UGameplayStatics::SaveGameToSlot(SaveGameInstance, SaveGameInstance->SaveSlotName, SaveGameInstance->UserIndex);

    }
}

void UResourceSystem::LoadPlayerResources()
{
    if (UGameplayStatics::DoesSaveGameExist(SaveSlotName, UserIndex))
    {
        UResourceSave* SaveGameInstance = Cast<UResourceSave>(UGameplayStatics::LoadGameFromSlot(SaveSlotName, UserIndex));

        if (SaveGameInstance)
        {
            OwnedResources = SaveGameInstance->OwnedResources;
        }
    }
}

void UResourceSystem::AddResourceToInventory(FName& ResourceToAdd, const int32 NumToAdd)
{
    FResourceQuantity* ResourceAdded = OwnedResources.Find(ResourceToAdd);

    if (!ResourceAdded)
    {
        FResourceQuantity NewResource;
        NewResource.ResourceData.Name = ResourceToAdd.ToString();
        NewResource.Quantity = NumToAdd;
        OwnedResources.Add(ResourceToAdd, NewResource);
    }
    else
    {
        ResourceAdded->Quantity += NumToAdd;
    }
}

