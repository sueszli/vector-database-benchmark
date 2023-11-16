#include "BuildComponent.h"
#include "BuildActor.h"
#include "MyProject3Character.h"
#include "Engine/World.h"
#include "Kismet/KismetSystemLibrary.h"

UBuildComponent::UBuildComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
	BuildingActorClass = ABuildActor::StaticClass();
	BuildGhostMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("BuildMeshComponent"));
}

void UBuildComponent::BeginPlay()
{
	Super::BeginPlay();

	Owner = Cast<AMyProject3Character>(GetOwner());

	if (Owner != nullptr)
	{
		Camera = Owner->FindComponentByClass<UCameraComponent>();
		SkeletalMeshComponent = Owner->FindComponentByClass<USkeletalMeshComponent>();
	}

	BuildGhostMesh->SetHiddenInGame(false);
	BuildGhostMesh->SetCollisionEnabled(ECollisionEnabled::QueryOnly);
	BuildGhostMesh->SetCollisionResponseToAllChannels(ECR_Ignore);
	BuildGhostMesh->SetCastShadow(false);
	BuildGhostMesh->SetOnlyOwnerSee(true);

	if (Owner != nullptr && Owner->GetRootComponent() != nullptr)
	{
		BuildGhostMesh->SetupAttachment(Owner->GetRootComponent());
	}

	InitializeBuildingMeshDataArray();

	bIsBuildModeOn = false;

	ChangeMesh();

	GetWorld()->GetTimerManager().SetTimer(UpdateBuildComponentTimerHandle, this, &UBuildComponent::UpdateBuildComponent, UpdateInterval, true);
}

void UBuildComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
}

void UBuildComponent::SpawnBuilding()
{
	if (BuildingActorClass && BuildGhostMesh->IsVisible() && BuildingMesh && CachedLineTraceResult.bBlockingHit)
	{
		bool bHasEnoughResources = true;

		if (bHasEnoughResources)
		{
			FTransform InstanceTransform = BuildGhostMesh->GetComponentTransform();

			ABuildActor*& NewBuildingInstanceActor = BuildingActors.FindOrAdd(BuildingMesh);

			if (!NewBuildingInstanceActor)
			{
				NewBuildingInstanceActor = GetWorld()->SpawnActor<ABuildActor>(BuildingActorClass, FVector::ZeroVector, FRotator::ZeroRotator);
				NewBuildingInstanceActor->BaseInstancedMesh->SetStaticMesh(BuildingMesh);
			}

			NewBuildingInstanceActor->BaseInstancedMesh->AddInstance(InstanceTransform);

			SetBuildMode(false);
			BuildGhostMesh->SetHiddenInGame(true);
		}
	}
}

void UBuildComponent::RotateBuilding(float DeltaRotation)
{
	if (bIsBuildModeOn)
	{
		float NewYaw = BuildGhostMesh->GetComponentRotation().Yaw + DeltaRotation * RotationSpeed;
		BuildGhostMesh->SetWorldRotation(FRotator(0.0f, NewYaw, 0.0f));
	}
}

void UBuildComponent::ChangeMesh()
{
	int32 NumRows = BuildingMeshDataArray.Num();
	if (NumRows > 0)
	{
		CurrentRowIndex = (CurrentRowIndex + 1) % NumRows;
		FBuildingMeshData& NextRow = BuildingMeshDataArray[CurrentRowIndex];

		BuildingMesh = NextRow.buildMesh;
		GhostMesh = NextRow.buildGhostMesh;
		BuildGhostMesh->SetStaticMesh(BuildingMesh);
		BuildGhostMesh->SetMaterial(0, ValidBuildMaterial);
		socketTagList = &NextRow.socketNames;
		UE_LOG(LogTemp, Warning, TEXT("Set Mesh: %s"), socketTagList);
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("Mesh not found in TArray"));
	}
}

void UBuildComponent::DeleteBuild()
{
	if (CachedLineTraceResult.bBlockingHit)
	{
		ABuildActor* BuildComp = Cast<ABuildActor>(CachedLineTraceResult.GetActor());
		if (BuildComp)
		{
			BuildComp->DestroyInstance(CachedLineTraceResult);
		}
	}
}

FHitResult UBuildComponent::PerformLineTrace(const int LineTraceRange, bool bDebug)
{
	FVector SpawnLocation = Camera->GetComponentLocation();
	FVector SpawnDirection = Camera->GetForwardVector();
	FVector SpawnEnd = SpawnDirection * LineTraceRange + SpawnLocation;

	FHitResult HitResult;
	FCollisionQueryParams TraceParams;
	TraceParams.AddIgnoredActor(Owner);
	TraceParams.bTraceComplex = true;
	TraceParams.bReturnPhysicalMaterial = true;

	bool bHit = GetWorld()->SweepSingleByChannel(HitResult, SpawnLocation, SpawnEnd, FQuat::Identity, ECC_Camera, FCollisionShape::MakeCapsule(SnappingSensitivity, SnappingSensitivity), TraceParams);

	if (bDebug)
	{
		FColor DebugLineColor = bHit ? FColor::Red : FColor::Green;
		float DebugLineLifetime = 5.0f;
		uint8 DepthPriority = 0;
		float LineThickness = 2.0f;
		DrawDebugCapsule(GetWorld(), SpawnLocation + 0.5 * (SpawnEnd - SpawnLocation), 10.0f, 50.0f, FQuat::Identity, DebugLineColor, true, DebugLineLifetime, DepthPriority, LineThickness);
	}

	return HitResult;
}

ABuildActor* UBuildComponent::GetHitBuildingActor(const FHitResult& HitResult)
{
	return Cast<ABuildActor>(HitResult.GetActor());
}

UStaticMesh* UBuildComponent::GetDataFromDataTable(FName RowName)
{
	if (!BuildingMeshDataTable)
	{
		return nullptr;
	}

	FBuildingMeshData* Row = BuildingMeshDataTable->FindRow<FBuildingMeshData>(RowName, TEXT("MESH"));

	if (Row)
	{
		return Row->buildMesh;
	}

	return nullptr;
}

void UBuildComponent::InitializeBuildingMeshDataArray()
{
	if (BuildingMeshDataTable)
	{
		TArray<FName> RowNames = BuildingMeshDataTable->GetRowNames();
		for (const FName& RowName : RowNames)
		{
			FBuildingMeshData* Row = BuildingMeshDataTable->FindRow<FBuildingMeshData>(RowName, TEXT("MESH"));
			if (Row)
			{
				BuildingMeshDataArray.Add(*Row);
			}
		}
	}
}

void UBuildComponent::UpdateBuildComponent()
{
	if (bIsBuildModeOn)
	{
		CachedLineTraceResult = PerformLineTrace(BuildingTraceRange, Debug);
		FVector SpawnLocation;
		FRotator SpawnRotation = BuildGhostMesh->GetComponentRotation();

		if (CachedLineTraceResult.bBlockingHit)
		{
			ABuildActor* HitBuildingActor = GetHitBuildingActor(CachedLineTraceResult);
			if (HitBuildingActor && bEnableSnapping)
			{
				SpawnLocation = GetSpawnLocationWithSocketAttachment(HitBuildingActor, SpawnRotation);
			}
			else
			{
				SpawnLocation = GetSpawnLocationWithSnapping();
			}
		}

		SetBuildGhostMeshTransform(SpawnLocation, SpawnRotation);
		BuildGhostMesh->SetHiddenInGame(false);
	}
	else
	{
		BuildGhostMesh->SetHiddenInGame(true);
	}
}

FVector UBuildComponent::GetSpawnLocationWithSocketAttachment(ABuildActor* HitBuildingActor, FRotator& OutSpawnRotation)
{
	FVector SpawnLocation;
	TOptional<FTransform> SocketToAttach = HitBuildingActor->GetHitSocketTransform(CachedLineTraceResult, *socketTagList);


	if (SocketToAttach.IsSet())
	{
		FTransform SocketTransform = SocketToAttach.GetValue();
		SpawnLocation = SocketTransform.GetLocation();
		OutSpawnRotation = SocketTransform.GetRotation().Rotator();
	}
	else
	{
		SpawnLocation = GetSpawnLocationWithSnapping();
	}
	return SpawnLocation;
}

FVector UBuildComponent::GetSpawnLocationWithSnapping()
{
	FVector SpawnLocation;

	if (bEnableSnapping)
	{
		FVector GridSize(GridSizeInput, GridSizeInput, GridSizeInput);
		FVector SnappedLocation = FVector(
			FMath::GridSnap(CachedLineTraceResult.Location.X, GridSize.X),
			FMath::GridSnap(CachedLineTraceResult.Location.Y, GridSize.Y),
			FMath::GridSnap(CachedLineTraceResult.Location.Z, GridSize.Z)
		);
		SpawnLocation = SnappedLocation;
	}
	else
	{
		SpawnLocation = CachedLineTraceResult.Location;
	}

	return SpawnLocation;
}

void UBuildComponent::SetBuildGhostMeshTransform(const FVector& SpawnLocation, const FRotator& SpawnRotation)
{
	FTransform ResultTransform;
	ResultTransform.SetLocation(SpawnLocation);
	ResultTransform.SetRotation(SpawnRotation.Quaternion());
	BuildGhostMesh->SetWorldTransform(ResultTransform);
}
