// Copyright Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
#include "MyProject3/BuildComponent.h"
#include "Engine/Classes/Engine/EngineTypes.h"
#include "Engine/Classes/Engine/HitResult.h"
#include "MyProject3/BuildActor.h"
PRAGMA_DISABLE_DEPRECATION_WARNINGS
void EmptyLinkFunctionForGeneratedCodeBuildComponent() {}
// Cross Module References
	COREUOBJECT_API UClass* Z_Construct_UClass_UClass();
	COREUOBJECT_API UScriptStruct* Z_Construct_UScriptStruct_FRotator();
	COREUOBJECT_API UScriptStruct* Z_Construct_UScriptStruct_FVector();
	ENGINE_API UClass* Z_Construct_UClass_AActor_NoRegister();
	ENGINE_API UClass* Z_Construct_UClass_UActorComponent();
	ENGINE_API UClass* Z_Construct_UClass_UCameraComponent_NoRegister();
	ENGINE_API UClass* Z_Construct_UClass_UDataTable_NoRegister();
	ENGINE_API UClass* Z_Construct_UClass_UMaterialInterface_NoRegister();
	ENGINE_API UClass* Z_Construct_UClass_USkeletalMeshComponent_NoRegister();
	ENGINE_API UClass* Z_Construct_UClass_UStaticMesh_NoRegister();
	ENGINE_API UClass* Z_Construct_UClass_UStaticMeshComponent_NoRegister();
	ENGINE_API UScriptStruct* Z_Construct_UScriptStruct_FHitResult();
	ENGINE_API UScriptStruct* Z_Construct_UScriptStruct_FTimerHandle();
	MYPROJECT3_API UClass* Z_Construct_UClass_ABuildActor_NoRegister();
	MYPROJECT3_API UClass* Z_Construct_UClass_UBuildComponent();
	MYPROJECT3_API UClass* Z_Construct_UClass_UBuildComponent_NoRegister();
	MYPROJECT3_API UScriptStruct* Z_Construct_UScriptStruct_FBuildingMeshData();
	UPackage* Z_Construct_UPackage__Script_MyProject3();
// End Cross Module References
	DEFINE_FUNCTION(UBuildComponent::execGetDataFromDataTable)
	{
		P_GET_PROPERTY(FNameProperty,Z_Param_RowName);
		P_FINISH;
		P_NATIVE_BEGIN;
		*(UStaticMesh**)Z_Param__Result=P_THIS->GetDataFromDataTable(Z_Param_RowName);
		P_NATIVE_END;
	}
	DEFINE_FUNCTION(UBuildComponent::execSetBuildGhostMeshTransform)
	{
		P_GET_STRUCT_REF(FVector,Z_Param_Out_SpawnLocation);
		P_GET_STRUCT_REF(FRotator,Z_Param_Out_SpawnRotation);
		P_FINISH;
		P_NATIVE_BEGIN;
		P_THIS->SetBuildGhostMeshTransform(Z_Param_Out_SpawnLocation,Z_Param_Out_SpawnRotation);
		P_NATIVE_END;
	}
	DEFINE_FUNCTION(UBuildComponent::execGetHitBuildingActor)
	{
		P_GET_STRUCT_REF(FHitResult,Z_Param_Out_HitResult);
		P_FINISH;
		P_NATIVE_BEGIN;
		*(ABuildActor**)Z_Param__Result=P_THIS->GetHitBuildingActor(Z_Param_Out_HitResult);
		P_NATIVE_END;
	}
	DEFINE_FUNCTION(UBuildComponent::execGetSpawnLocationWithSnapping)
	{
		P_FINISH;
		P_NATIVE_BEGIN;
		*(FVector*)Z_Param__Result=P_THIS->GetSpawnLocationWithSnapping();
		P_NATIVE_END;
	}
	DEFINE_FUNCTION(UBuildComponent::execGetSpawnLocationWithSocketAttachment)
	{
		P_GET_OBJECT(ABuildActor,Z_Param_HitBuildingActor);
		P_GET_STRUCT_REF(FRotator,Z_Param_Out_OutSpawnRotation);
		P_FINISH;
		P_NATIVE_BEGIN;
		*(FVector*)Z_Param__Result=P_THIS->GetSpawnLocationWithSocketAttachment(Z_Param_HitBuildingActor,Z_Param_Out_OutSpawnRotation);
		P_NATIVE_END;
	}
	DEFINE_FUNCTION(UBuildComponent::execInitializeBuildingMeshDataArray)
	{
		P_FINISH;
		P_NATIVE_BEGIN;
		P_THIS->InitializeBuildingMeshDataArray();
		P_NATIVE_END;
	}
	DEFINE_FUNCTION(UBuildComponent::execUpdateBuildComponent)
	{
		P_FINISH;
		P_NATIVE_BEGIN;
		P_THIS->UpdateBuildComponent();
		P_NATIVE_END;
	}
	DEFINE_FUNCTION(UBuildComponent::execGetBuildMode)
	{
		P_FINISH;
		P_NATIVE_BEGIN;
		*(bool*)Z_Param__Result=P_THIS->GetBuildMode();
		P_NATIVE_END;
	}
	DEFINE_FUNCTION(UBuildComponent::execSetBuildMode)
	{
		P_GET_UBOOL(Z_Param_bSetBuildMode);
		P_FINISH;
		P_NATIVE_BEGIN;
		P_THIS->SetBuildMode(Z_Param_bSetBuildMode);
		P_NATIVE_END;
	}
	DEFINE_FUNCTION(UBuildComponent::execPerformLineTrace)
	{
		P_GET_PROPERTY(FIntProperty,Z_Param_LineTraceRange);
		P_GET_UBOOL(Z_Param_bDebug);
		P_FINISH;
		P_NATIVE_BEGIN;
		*(FHitResult*)Z_Param__Result=P_THIS->PerformLineTrace(Z_Param_LineTraceRange,Z_Param_bDebug);
		P_NATIVE_END;
	}
	DEFINE_FUNCTION(UBuildComponent::execDeleteBuild)
	{
		P_FINISH;
		P_NATIVE_BEGIN;
		P_THIS->DeleteBuild();
		P_NATIVE_END;
	}
	DEFINE_FUNCTION(UBuildComponent::execChangeMesh)
	{
		P_FINISH;
		P_NATIVE_BEGIN;
		P_THIS->ChangeMesh();
		P_NATIVE_END;
	}
	DEFINE_FUNCTION(UBuildComponent::execRotateBuilding)
	{
		P_GET_PROPERTY(FFloatProperty,Z_Param_DeltaRotation);
		P_FINISH;
		P_NATIVE_BEGIN;
		P_THIS->RotateBuilding(Z_Param_DeltaRotation);
		P_NATIVE_END;
	}
	DEFINE_FUNCTION(UBuildComponent::execSpawnBuilding)
	{
		P_FINISH;
		P_NATIVE_BEGIN;
		P_THIS->SpawnBuilding();
		P_NATIVE_END;
	}
	void UBuildComponent::StaticRegisterNativesUBuildComponent()
	{
		UClass* Class = UBuildComponent::StaticClass();
		static const FNameNativePtrPair Funcs[] = {
			{ "ChangeMesh", &UBuildComponent::execChangeMesh },
			{ "DeleteBuild", &UBuildComponent::execDeleteBuild },
			{ "GetBuildMode", &UBuildComponent::execGetBuildMode },
			{ "GetDataFromDataTable", &UBuildComponent::execGetDataFromDataTable },
			{ "GetHitBuildingActor", &UBuildComponent::execGetHitBuildingActor },
			{ "GetSpawnLocationWithSnapping", &UBuildComponent::execGetSpawnLocationWithSnapping },
			{ "GetSpawnLocationWithSocketAttachment", &UBuildComponent::execGetSpawnLocationWithSocketAttachment },
			{ "InitializeBuildingMeshDataArray", &UBuildComponent::execInitializeBuildingMeshDataArray },
			{ "PerformLineTrace", &UBuildComponent::execPerformLineTrace },
			{ "RotateBuilding", &UBuildComponent::execRotateBuilding },
			{ "SetBuildGhostMeshTransform", &UBuildComponent::execSetBuildGhostMeshTransform },
			{ "SetBuildMode", &UBuildComponent::execSetBuildMode },
			{ "SpawnBuilding", &UBuildComponent::execSpawnBuilding },
			{ "UpdateBuildComponent", &UBuildComponent::execUpdateBuildComponent },
		};
		FNativeFunctionRegistrar::RegisterFunctions(Class, Funcs, UE_ARRAY_COUNT(Funcs));
	}
	struct Z_Construct_UFunction_UBuildComponent_ChangeMesh_Statics
	{
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UBuildComponent_ChangeMesh_Statics::Function_MetaDataParams[] = {
		{ "Category", "Build Mode" },
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_UBuildComponent_ChangeMesh_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_UBuildComponent, nullptr, "ChangeMesh", nullptr, nullptr, 0, nullptr, 0, RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x04080401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_UBuildComponent_ChangeMesh_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_ChangeMesh_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_UBuildComponent_ChangeMesh()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_UBuildComponent_ChangeMesh_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_UBuildComponent_DeleteBuild_Statics
	{
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UBuildComponent_DeleteBuild_Statics::Function_MetaDataParams[] = {
		{ "Category", "Build Mode" },
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_UBuildComponent_DeleteBuild_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_UBuildComponent, nullptr, "DeleteBuild", nullptr, nullptr, 0, nullptr, 0, RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x04080401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_UBuildComponent_DeleteBuild_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_DeleteBuild_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_UBuildComponent_DeleteBuild()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_UBuildComponent_DeleteBuild_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_UBuildComponent_GetBuildMode_Statics
	{
		struct BuildComponent_eventGetBuildMode_Parms
		{
			bool ReturnValue;
		};
		static void NewProp_ReturnValue_SetBit(void* Obj);
		static const UECodeGen_Private::FBoolPropertyParams NewProp_ReturnValue;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
	void Z_Construct_UFunction_UBuildComponent_GetBuildMode_Statics::NewProp_ReturnValue_SetBit(void* Obj)
	{
		((BuildComponent_eventGetBuildMode_Parms*)Obj)->ReturnValue = 1;
	}
	const UECodeGen_Private::FBoolPropertyParams Z_Construct_UFunction_UBuildComponent_GetBuildMode_Statics::NewProp_ReturnValue = { "ReturnValue", nullptr, (EPropertyFlags)0x0010000000000580, UECodeGen_Private::EPropertyGenFlags::Bool | UECodeGen_Private::EPropertyGenFlags::NativeBool, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, sizeof(bool), sizeof(BuildComponent_eventGetBuildMode_Parms), &Z_Construct_UFunction_UBuildComponent_GetBuildMode_Statics::NewProp_ReturnValue_SetBit, METADATA_PARAMS(nullptr, 0) };
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_UBuildComponent_GetBuildMode_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UBuildComponent_GetBuildMode_Statics::NewProp_ReturnValue,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UBuildComponent_GetBuildMode_Statics::Function_MetaDataParams[] = {
		{ "Category", "Build Mode" },
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_UBuildComponent_GetBuildMode_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_UBuildComponent, nullptr, "GetBuildMode", nullptr, nullptr, sizeof(Z_Construct_UFunction_UBuildComponent_GetBuildMode_Statics::BuildComponent_eventGetBuildMode_Parms), Z_Construct_UFunction_UBuildComponent_GetBuildMode_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_GetBuildMode_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x04080401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_UBuildComponent_GetBuildMode_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_GetBuildMode_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_UBuildComponent_GetBuildMode()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_UBuildComponent_GetBuildMode_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_UBuildComponent_GetDataFromDataTable_Statics
	{
		struct BuildComponent_eventGetDataFromDataTable_Parms
		{
			FName RowName;
			UStaticMesh* ReturnValue;
		};
		static const UECodeGen_Private::FNamePropertyParams NewProp_RowName;
		static const UECodeGen_Private::FObjectPropertyParams NewProp_ReturnValue;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
	const UECodeGen_Private::FNamePropertyParams Z_Construct_UFunction_UBuildComponent_GetDataFromDataTable_Statics::NewProp_RowName = { "RowName", nullptr, (EPropertyFlags)0x0010000000000080, UECodeGen_Private::EPropertyGenFlags::Name, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(BuildComponent_eventGetDataFromDataTable_Parms, RowName), METADATA_PARAMS(nullptr, 0) };
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UFunction_UBuildComponent_GetDataFromDataTable_Statics::NewProp_ReturnValue = { "ReturnValue", nullptr, (EPropertyFlags)0x0010000000000580, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(BuildComponent_eventGetDataFromDataTable_Parms, ReturnValue), Z_Construct_UClass_UStaticMesh_NoRegister, METADATA_PARAMS(nullptr, 0) };
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_UBuildComponent_GetDataFromDataTable_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UBuildComponent_GetDataFromDataTable_Statics::NewProp_RowName,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UBuildComponent_GetDataFromDataTable_Statics::NewProp_ReturnValue,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UBuildComponent_GetDataFromDataTable_Statics::Function_MetaDataParams[] = {
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_UBuildComponent_GetDataFromDataTable_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_UBuildComponent, nullptr, "GetDataFromDataTable", nullptr, nullptr, sizeof(Z_Construct_UFunction_UBuildComponent_GetDataFromDataTable_Statics::BuildComponent_eventGetDataFromDataTable_Parms), Z_Construct_UFunction_UBuildComponent_GetDataFromDataTable_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_GetDataFromDataTable_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x00040401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_UBuildComponent_GetDataFromDataTable_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_GetDataFromDataTable_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_UBuildComponent_GetDataFromDataTable()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_UBuildComponent_GetDataFromDataTable_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_UBuildComponent_GetHitBuildingActor_Statics
	{
		struct BuildComponent_eventGetHitBuildingActor_Parms
		{
			FHitResult HitResult;
			ABuildActor* ReturnValue;
		};
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_HitResult_MetaData[];
#endif
		static const UECodeGen_Private::FStructPropertyParams NewProp_HitResult;
		static const UECodeGen_Private::FObjectPropertyParams NewProp_ReturnValue;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UBuildComponent_GetHitBuildingActor_Statics::NewProp_HitResult_MetaData[] = {
		{ "NativeConst", "" },
	};
#endif
	const UECodeGen_Private::FStructPropertyParams Z_Construct_UFunction_UBuildComponent_GetHitBuildingActor_Statics::NewProp_HitResult = { "HitResult", nullptr, (EPropertyFlags)0x0010008008000182, UECodeGen_Private::EPropertyGenFlags::Struct, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(BuildComponent_eventGetHitBuildingActor_Parms, HitResult), Z_Construct_UScriptStruct_FHitResult, METADATA_PARAMS(Z_Construct_UFunction_UBuildComponent_GetHitBuildingActor_Statics::NewProp_HitResult_MetaData, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_GetHitBuildingActor_Statics::NewProp_HitResult_MetaData)) }; // 1287526515
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UFunction_UBuildComponent_GetHitBuildingActor_Statics::NewProp_ReturnValue = { "ReturnValue", nullptr, (EPropertyFlags)0x0010000000000580, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(BuildComponent_eventGetHitBuildingActor_Parms, ReturnValue), Z_Construct_UClass_ABuildActor_NoRegister, METADATA_PARAMS(nullptr, 0) };
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_UBuildComponent_GetHitBuildingActor_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UBuildComponent_GetHitBuildingActor_Statics::NewProp_HitResult,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UBuildComponent_GetHitBuildingActor_Statics::NewProp_ReturnValue,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UBuildComponent_GetHitBuildingActor_Statics::Function_MetaDataParams[] = {
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_UBuildComponent_GetHitBuildingActor_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_UBuildComponent, nullptr, "GetHitBuildingActor", nullptr, nullptr, sizeof(Z_Construct_UFunction_UBuildComponent_GetHitBuildingActor_Statics::BuildComponent_eventGetHitBuildingActor_Parms), Z_Construct_UFunction_UBuildComponent_GetHitBuildingActor_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_GetHitBuildingActor_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x00440401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_UBuildComponent_GetHitBuildingActor_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_GetHitBuildingActor_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_UBuildComponent_GetHitBuildingActor()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_UBuildComponent_GetHitBuildingActor_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSnapping_Statics
	{
		struct BuildComponent_eventGetSpawnLocationWithSnapping_Parms
		{
			FVector ReturnValue;
		};
		static const UECodeGen_Private::FStructPropertyParams NewProp_ReturnValue;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
	const UECodeGen_Private::FStructPropertyParams Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSnapping_Statics::NewProp_ReturnValue = { "ReturnValue", nullptr, (EPropertyFlags)0x0010000000000580, UECodeGen_Private::EPropertyGenFlags::Struct, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(BuildComponent_eventGetSpawnLocationWithSnapping_Parms, ReturnValue), Z_Construct_UScriptStruct_FVector, METADATA_PARAMS(nullptr, 0) };
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSnapping_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSnapping_Statics::NewProp_ReturnValue,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSnapping_Statics::Function_MetaDataParams[] = {
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSnapping_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_UBuildComponent, nullptr, "GetSpawnLocationWithSnapping", nullptr, nullptr, sizeof(Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSnapping_Statics::BuildComponent_eventGetSpawnLocationWithSnapping_Parms), Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSnapping_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSnapping_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x00840401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSnapping_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSnapping_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSnapping()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSnapping_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSocketAttachment_Statics
	{
		struct BuildComponent_eventGetSpawnLocationWithSocketAttachment_Parms
		{
			ABuildActor* HitBuildingActor;
			FRotator OutSpawnRotation;
			FVector ReturnValue;
		};
		static const UECodeGen_Private::FObjectPropertyParams NewProp_HitBuildingActor;
		static const UECodeGen_Private::FStructPropertyParams NewProp_OutSpawnRotation;
		static const UECodeGen_Private::FStructPropertyParams NewProp_ReturnValue;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSocketAttachment_Statics::NewProp_HitBuildingActor = { "HitBuildingActor", nullptr, (EPropertyFlags)0x0010000000000080, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(BuildComponent_eventGetSpawnLocationWithSocketAttachment_Parms, HitBuildingActor), Z_Construct_UClass_ABuildActor_NoRegister, METADATA_PARAMS(nullptr, 0) };
	const UECodeGen_Private::FStructPropertyParams Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSocketAttachment_Statics::NewProp_OutSpawnRotation = { "OutSpawnRotation", nullptr, (EPropertyFlags)0x0010000000000180, UECodeGen_Private::EPropertyGenFlags::Struct, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(BuildComponent_eventGetSpawnLocationWithSocketAttachment_Parms, OutSpawnRotation), Z_Construct_UScriptStruct_FRotator, METADATA_PARAMS(nullptr, 0) };
	const UECodeGen_Private::FStructPropertyParams Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSocketAttachment_Statics::NewProp_ReturnValue = { "ReturnValue", nullptr, (EPropertyFlags)0x0010000000000580, UECodeGen_Private::EPropertyGenFlags::Struct, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(BuildComponent_eventGetSpawnLocationWithSocketAttachment_Parms, ReturnValue), Z_Construct_UScriptStruct_FVector, METADATA_PARAMS(nullptr, 0) };
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSocketAttachment_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSocketAttachment_Statics::NewProp_HitBuildingActor,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSocketAttachment_Statics::NewProp_OutSpawnRotation,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSocketAttachment_Statics::NewProp_ReturnValue,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSocketAttachment_Statics::Function_MetaDataParams[] = {
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSocketAttachment_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_UBuildComponent, nullptr, "GetSpawnLocationWithSocketAttachment", nullptr, nullptr, sizeof(Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSocketAttachment_Statics::BuildComponent_eventGetSpawnLocationWithSocketAttachment_Parms), Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSocketAttachment_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSocketAttachment_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x00C40401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSocketAttachment_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSocketAttachment_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSocketAttachment()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSocketAttachment_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_UBuildComponent_InitializeBuildingMeshDataArray_Statics
	{
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UBuildComponent_InitializeBuildingMeshDataArray_Statics::Function_MetaDataParams[] = {
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_UBuildComponent_InitializeBuildingMeshDataArray_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_UBuildComponent, nullptr, "InitializeBuildingMeshDataArray", nullptr, nullptr, 0, nullptr, 0, RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x00040401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_UBuildComponent_InitializeBuildingMeshDataArray_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_InitializeBuildingMeshDataArray_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_UBuildComponent_InitializeBuildingMeshDataArray()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_UBuildComponent_InitializeBuildingMeshDataArray_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_UBuildComponent_PerformLineTrace_Statics
	{
		struct BuildComponent_eventPerformLineTrace_Parms
		{
			int32 LineTraceRange;
			bool bDebug;
			FHitResult ReturnValue;
		};
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_LineTraceRange_MetaData[];
#endif
		static const UECodeGen_Private::FUnsizedIntPropertyParams NewProp_LineTraceRange;
		static void NewProp_bDebug_SetBit(void* Obj);
		static const UECodeGen_Private::FBoolPropertyParams NewProp_bDebug;
		static const UECodeGen_Private::FStructPropertyParams NewProp_ReturnValue;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UBuildComponent_PerformLineTrace_Statics::NewProp_LineTraceRange_MetaData[] = {
		{ "NativeConst", "" },
	};
#endif
	const UECodeGen_Private::FUnsizedIntPropertyParams Z_Construct_UFunction_UBuildComponent_PerformLineTrace_Statics::NewProp_LineTraceRange = { "LineTraceRange", nullptr, (EPropertyFlags)0x0010000000000082, UECodeGen_Private::EPropertyGenFlags::Int, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(BuildComponent_eventPerformLineTrace_Parms, LineTraceRange), METADATA_PARAMS(Z_Construct_UFunction_UBuildComponent_PerformLineTrace_Statics::NewProp_LineTraceRange_MetaData, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_PerformLineTrace_Statics::NewProp_LineTraceRange_MetaData)) };
	void Z_Construct_UFunction_UBuildComponent_PerformLineTrace_Statics::NewProp_bDebug_SetBit(void* Obj)
	{
		((BuildComponent_eventPerformLineTrace_Parms*)Obj)->bDebug = 1;
	}
	const UECodeGen_Private::FBoolPropertyParams Z_Construct_UFunction_UBuildComponent_PerformLineTrace_Statics::NewProp_bDebug = { "bDebug", nullptr, (EPropertyFlags)0x0010000000000080, UECodeGen_Private::EPropertyGenFlags::Bool | UECodeGen_Private::EPropertyGenFlags::NativeBool, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, sizeof(bool), sizeof(BuildComponent_eventPerformLineTrace_Parms), &Z_Construct_UFunction_UBuildComponent_PerformLineTrace_Statics::NewProp_bDebug_SetBit, METADATA_PARAMS(nullptr, 0) };
	const UECodeGen_Private::FStructPropertyParams Z_Construct_UFunction_UBuildComponent_PerformLineTrace_Statics::NewProp_ReturnValue = { "ReturnValue", nullptr, (EPropertyFlags)0x0010008000000580, UECodeGen_Private::EPropertyGenFlags::Struct, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(BuildComponent_eventPerformLineTrace_Parms, ReturnValue), Z_Construct_UScriptStruct_FHitResult, METADATA_PARAMS(nullptr, 0) }; // 1287526515
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_UBuildComponent_PerformLineTrace_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UBuildComponent_PerformLineTrace_Statics::NewProp_LineTraceRange,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UBuildComponent_PerformLineTrace_Statics::NewProp_bDebug,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UBuildComponent_PerformLineTrace_Statics::NewProp_ReturnValue,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UBuildComponent_PerformLineTrace_Statics::Function_MetaDataParams[] = {
		{ "Category", "Trace" },
		{ "Comment", "// Support Functions\n" },
		{ "CPP_Default_bDebug", "false" },
		{ "ModuleRelativePath", "BuildComponent.h" },
		{ "ToolTip", "Support Functions" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_UBuildComponent_PerformLineTrace_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_UBuildComponent, nullptr, "PerformLineTrace", nullptr, nullptr, sizeof(Z_Construct_UFunction_UBuildComponent_PerformLineTrace_Statics::BuildComponent_eventPerformLineTrace_Parms), Z_Construct_UFunction_UBuildComponent_PerformLineTrace_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_PerformLineTrace_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x04080401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_UBuildComponent_PerformLineTrace_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_PerformLineTrace_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_UBuildComponent_PerformLineTrace()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_UBuildComponent_PerformLineTrace_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_UBuildComponent_RotateBuilding_Statics
	{
		struct BuildComponent_eventRotateBuilding_Parms
		{
			float DeltaRotation;
		};
		static const UECodeGen_Private::FFloatPropertyParams NewProp_DeltaRotation;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
	const UECodeGen_Private::FFloatPropertyParams Z_Construct_UFunction_UBuildComponent_RotateBuilding_Statics::NewProp_DeltaRotation = { "DeltaRotation", nullptr, (EPropertyFlags)0x0010000000000080, UECodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(BuildComponent_eventRotateBuilding_Parms, DeltaRotation), METADATA_PARAMS(nullptr, 0) };
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_UBuildComponent_RotateBuilding_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UBuildComponent_RotateBuilding_Statics::NewProp_DeltaRotation,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UBuildComponent_RotateBuilding_Statics::Function_MetaDataParams[] = {
		{ "Category", "Build Mode" },
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_UBuildComponent_RotateBuilding_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_UBuildComponent, nullptr, "RotateBuilding", nullptr, nullptr, sizeof(Z_Construct_UFunction_UBuildComponent_RotateBuilding_Statics::BuildComponent_eventRotateBuilding_Parms), Z_Construct_UFunction_UBuildComponent_RotateBuilding_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_RotateBuilding_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x04080401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_UBuildComponent_RotateBuilding_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_RotateBuilding_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_UBuildComponent_RotateBuilding()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_UBuildComponent_RotateBuilding_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_UBuildComponent_SetBuildGhostMeshTransform_Statics
	{
		struct BuildComponent_eventSetBuildGhostMeshTransform_Parms
		{
			FVector SpawnLocation;
			FRotator SpawnRotation;
		};
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_SpawnLocation_MetaData[];
#endif
		static const UECodeGen_Private::FStructPropertyParams NewProp_SpawnLocation;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_SpawnRotation_MetaData[];
#endif
		static const UECodeGen_Private::FStructPropertyParams NewProp_SpawnRotation;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UBuildComponent_SetBuildGhostMeshTransform_Statics::NewProp_SpawnLocation_MetaData[] = {
		{ "NativeConst", "" },
	};
#endif
	const UECodeGen_Private::FStructPropertyParams Z_Construct_UFunction_UBuildComponent_SetBuildGhostMeshTransform_Statics::NewProp_SpawnLocation = { "SpawnLocation", nullptr, (EPropertyFlags)0x0010000008000182, UECodeGen_Private::EPropertyGenFlags::Struct, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(BuildComponent_eventSetBuildGhostMeshTransform_Parms, SpawnLocation), Z_Construct_UScriptStruct_FVector, METADATA_PARAMS(Z_Construct_UFunction_UBuildComponent_SetBuildGhostMeshTransform_Statics::NewProp_SpawnLocation_MetaData, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_SetBuildGhostMeshTransform_Statics::NewProp_SpawnLocation_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UBuildComponent_SetBuildGhostMeshTransform_Statics::NewProp_SpawnRotation_MetaData[] = {
		{ "NativeConst", "" },
	};
#endif
	const UECodeGen_Private::FStructPropertyParams Z_Construct_UFunction_UBuildComponent_SetBuildGhostMeshTransform_Statics::NewProp_SpawnRotation = { "SpawnRotation", nullptr, (EPropertyFlags)0x0010000008000182, UECodeGen_Private::EPropertyGenFlags::Struct, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(BuildComponent_eventSetBuildGhostMeshTransform_Parms, SpawnRotation), Z_Construct_UScriptStruct_FRotator, METADATA_PARAMS(Z_Construct_UFunction_UBuildComponent_SetBuildGhostMeshTransform_Statics::NewProp_SpawnRotation_MetaData, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_SetBuildGhostMeshTransform_Statics::NewProp_SpawnRotation_MetaData)) };
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_UBuildComponent_SetBuildGhostMeshTransform_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UBuildComponent_SetBuildGhostMeshTransform_Statics::NewProp_SpawnLocation,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UBuildComponent_SetBuildGhostMeshTransform_Statics::NewProp_SpawnRotation,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UBuildComponent_SetBuildGhostMeshTransform_Statics::Function_MetaDataParams[] = {
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_UBuildComponent_SetBuildGhostMeshTransform_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_UBuildComponent, nullptr, "SetBuildGhostMeshTransform", nullptr, nullptr, sizeof(Z_Construct_UFunction_UBuildComponent_SetBuildGhostMeshTransform_Statics::BuildComponent_eventSetBuildGhostMeshTransform_Parms), Z_Construct_UFunction_UBuildComponent_SetBuildGhostMeshTransform_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_SetBuildGhostMeshTransform_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x00C40401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_UBuildComponent_SetBuildGhostMeshTransform_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_SetBuildGhostMeshTransform_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_UBuildComponent_SetBuildGhostMeshTransform()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_UBuildComponent_SetBuildGhostMeshTransform_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_UBuildComponent_SetBuildMode_Statics
	{
		struct BuildComponent_eventSetBuildMode_Parms
		{
			bool bSetBuildMode;
		};
		static void NewProp_bSetBuildMode_SetBit(void* Obj);
		static const UECodeGen_Private::FBoolPropertyParams NewProp_bSetBuildMode;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
	void Z_Construct_UFunction_UBuildComponent_SetBuildMode_Statics::NewProp_bSetBuildMode_SetBit(void* Obj)
	{
		((BuildComponent_eventSetBuildMode_Parms*)Obj)->bSetBuildMode = 1;
	}
	const UECodeGen_Private::FBoolPropertyParams Z_Construct_UFunction_UBuildComponent_SetBuildMode_Statics::NewProp_bSetBuildMode = { "bSetBuildMode", nullptr, (EPropertyFlags)0x0010000000000080, UECodeGen_Private::EPropertyGenFlags::Bool | UECodeGen_Private::EPropertyGenFlags::NativeBool, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, sizeof(bool), sizeof(BuildComponent_eventSetBuildMode_Parms), &Z_Construct_UFunction_UBuildComponent_SetBuildMode_Statics::NewProp_bSetBuildMode_SetBit, METADATA_PARAMS(nullptr, 0) };
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_UBuildComponent_SetBuildMode_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UBuildComponent_SetBuildMode_Statics::NewProp_bSetBuildMode,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UBuildComponent_SetBuildMode_Statics::Function_MetaDataParams[] = {
		{ "Category", "Build Mode" },
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_UBuildComponent_SetBuildMode_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_UBuildComponent, nullptr, "SetBuildMode", nullptr, nullptr, sizeof(Z_Construct_UFunction_UBuildComponent_SetBuildMode_Statics::BuildComponent_eventSetBuildMode_Parms), Z_Construct_UFunction_UBuildComponent_SetBuildMode_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_SetBuildMode_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x04080401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_UBuildComponent_SetBuildMode_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_SetBuildMode_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_UBuildComponent_SetBuildMode()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_UBuildComponent_SetBuildMode_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_UBuildComponent_SpawnBuilding_Statics
	{
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UBuildComponent_SpawnBuilding_Statics::Function_MetaDataParams[] = {
		{ "Category", "Build Mode" },
		{ "Comment", "// Build Functions\n" },
		{ "ModuleRelativePath", "BuildComponent.h" },
		{ "ToolTip", "Build Functions" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_UBuildComponent_SpawnBuilding_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_UBuildComponent, nullptr, "SpawnBuilding", nullptr, nullptr, 0, nullptr, 0, RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x04080401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_UBuildComponent_SpawnBuilding_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_SpawnBuilding_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_UBuildComponent_SpawnBuilding()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_UBuildComponent_SpawnBuilding_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_UBuildComponent_UpdateBuildComponent_Statics
	{
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UBuildComponent_UpdateBuildComponent_Statics::Function_MetaDataParams[] = {
		{ "Comment", "//Used Functions\n" },
		{ "ModuleRelativePath", "BuildComponent.h" },
		{ "ToolTip", "Used Functions" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_UBuildComponent_UpdateBuildComponent_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_UBuildComponent, nullptr, "UpdateBuildComponent", nullptr, nullptr, 0, nullptr, 0, RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x00040401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_UBuildComponent_UpdateBuildComponent_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_UBuildComponent_UpdateBuildComponent_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_UBuildComponent_UpdateBuildComponent()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_UBuildComponent_UpdateBuildComponent_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	IMPLEMENT_CLASS_NO_AUTO_REGISTRATION(UBuildComponent);
	UClass* Z_Construct_UClass_UBuildComponent_NoRegister()
	{
		return UBuildComponent::StaticClass();
	}
	struct Z_Construct_UClass_UBuildComponent_Statics
	{
		static UObject* (*const DependentSingletons[])();
		static const FClassFunctionLinkInfo FuncInfo[];
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Class_MetaDataParams[];
#endif
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_bIsBuildModeOn_MetaData[];
#endif
		static void NewProp_bIsBuildModeOn_SetBit(void* Obj);
		static const UECodeGen_Private::FBoolPropertyParams NewProp_bIsBuildModeOn;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_BuildingMeshDataTable_MetaData[];
#endif
		static const UECodeGen_Private::FObjectPropertyParams NewProp_BuildingMeshDataTable;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_BuildingTraceRange_MetaData[];
#endif
		static const UECodeGen_Private::FUnsizedIntPropertyParams NewProp_BuildingTraceRange;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_bEnableSnapping_MetaData[];
#endif
		static void NewProp_bEnableSnapping_SetBit(void* Obj);
		static const UECodeGen_Private::FBoolPropertyParams NewProp_bEnableSnapping;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_GridSizeInput_MetaData[];
#endif
		static const UECodeGen_Private::FFloatPropertyParams NewProp_GridSizeInput;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_SnappingSensitivity_MetaData[];
#endif
		static const UECodeGen_Private::FFloatPropertyParams NewProp_SnappingSensitivity;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_RotationSpeed_MetaData[];
#endif
		static const UECodeGen_Private::FFloatPropertyParams NewProp_RotationSpeed;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_UpdateInterval_MetaData[];
#endif
		static const UECodeGen_Private::FFloatPropertyParams NewProp_UpdateInterval;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_Debug_MetaData[];
#endif
		static void NewProp_Debug_SetBit(void* Obj);
		static const UECodeGen_Private::FBoolPropertyParams NewProp_Debug;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_ValidBuildMaterial_MetaData[];
#endif
		static const UECodeGen_Private::FObjectPropertyParams NewProp_ValidBuildMaterial;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_InvalidLocationMaterial_MetaData[];
#endif
		static const UECodeGen_Private::FObjectPropertyParams NewProp_InvalidLocationMaterial;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_GhostMesh_MetaData[];
#endif
		static const UECodeGen_Private::FObjectPropertyParams NewProp_GhostMesh;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_BuildingMesh_MetaData[];
#endif
		static const UECodeGen_Private::FObjectPropertyParams NewProp_BuildingMesh;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_Owner_MetaData[];
#endif
		static const UECodeGen_Private::FObjectPropertyParams NewProp_Owner;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_Camera_MetaData[];
#endif
		static const UECodeGen_Private::FObjectPropertyParams NewProp_Camera;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_SkeletalMeshComponent_MetaData[];
#endif
		static const UECodeGen_Private::FObjectPropertyParams NewProp_SkeletalMeshComponent;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_BuildingActorClass_MetaData[];
#endif
		static const UECodeGen_Private::FClassPropertyParams NewProp_BuildingActorClass;
		static const UECodeGen_Private::FObjectPropertyParams NewProp_BuildingActors_ValueProp;
		static const UECodeGen_Private::FObjectPropertyParams NewProp_BuildingActors_Key_KeyProp;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_BuildingActors_MetaData[];
#endif
		static const UECodeGen_Private::FMapPropertyParams NewProp_BuildingActors;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_BuildGhostMesh_MetaData[];
#endif
		static const UECodeGen_Private::FObjectPropertyParams NewProp_BuildGhostMesh;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_CurrentRowIndex_MetaData[];
#endif
		static const UECodeGen_Private::FIntPropertyParams NewProp_CurrentRowIndex;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_CachedLineTraceResult_MetaData[];
#endif
		static const UECodeGen_Private::FStructPropertyParams NewProp_CachedLineTraceResult;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_UpdateBuildComponentTimerHandle_MetaData[];
#endif
		static const UECodeGen_Private::FStructPropertyParams NewProp_UpdateBuildComponentTimerHandle;
		static const UECodeGen_Private::FStructPropertyParams NewProp_BuildingMeshDataArray_Inner;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_BuildingMeshDataArray_MetaData[];
#endif
		static const UECodeGen_Private::FArrayPropertyParams NewProp_BuildingMeshDataArray;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
		static const FCppClassTypeInfoStatic StaticCppClassTypeInfo;
		static const UECodeGen_Private::FClassParams ClassParams;
	};
	UObject* (*const Z_Construct_UClass_UBuildComponent_Statics::DependentSingletons[])() = {
		(UObject* (*)())Z_Construct_UClass_UActorComponent,
		(UObject* (*)())Z_Construct_UPackage__Script_MyProject3,
	};
	const FClassFunctionLinkInfo Z_Construct_UClass_UBuildComponent_Statics::FuncInfo[] = {
		{ &Z_Construct_UFunction_UBuildComponent_ChangeMesh, "ChangeMesh" }, // 4137601790
		{ &Z_Construct_UFunction_UBuildComponent_DeleteBuild, "DeleteBuild" }, // 1033745826
		{ &Z_Construct_UFunction_UBuildComponent_GetBuildMode, "GetBuildMode" }, // 2808053901
		{ &Z_Construct_UFunction_UBuildComponent_GetDataFromDataTable, "GetDataFromDataTable" }, // 1623759939
		{ &Z_Construct_UFunction_UBuildComponent_GetHitBuildingActor, "GetHitBuildingActor" }, // 2190874688
		{ &Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSnapping, "GetSpawnLocationWithSnapping" }, // 4044917122
		{ &Z_Construct_UFunction_UBuildComponent_GetSpawnLocationWithSocketAttachment, "GetSpawnLocationWithSocketAttachment" }, // 447489762
		{ &Z_Construct_UFunction_UBuildComponent_InitializeBuildingMeshDataArray, "InitializeBuildingMeshDataArray" }, // 2093135076
		{ &Z_Construct_UFunction_UBuildComponent_PerformLineTrace, "PerformLineTrace" }, // 3818334090
		{ &Z_Construct_UFunction_UBuildComponent_RotateBuilding, "RotateBuilding" }, // 2381391634
		{ &Z_Construct_UFunction_UBuildComponent_SetBuildGhostMeshTransform, "SetBuildGhostMeshTransform" }, // 3382627454
		{ &Z_Construct_UFunction_UBuildComponent_SetBuildMode, "SetBuildMode" }, // 3657673962
		{ &Z_Construct_UFunction_UBuildComponent_SpawnBuilding, "SpawnBuilding" }, // 3841426521
		{ &Z_Construct_UFunction_UBuildComponent_UpdateBuildComponent, "UpdateBuildComponent" }, // 3845376821
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UBuildComponent_Statics::Class_MetaDataParams[] = {
		{ "BlueprintSpawnableComponent", "" },
		{ "ClassGroupNames", "Custom" },
		{ "IncludePath", "BuildComponent.h" },
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UBuildComponent_Statics::NewProp_bIsBuildModeOn_MetaData[] = {
		{ "Category", "Build Mode Settings" },
		{ "Comment", "// Build Variables\n" },
		{ "ModuleRelativePath", "BuildComponent.h" },
		{ "ToolTip", "Build Variables" },
	};
#endif
	void Z_Construct_UClass_UBuildComponent_Statics::NewProp_bIsBuildModeOn_SetBit(void* Obj)
	{
		((UBuildComponent*)Obj)->bIsBuildModeOn = 1;
	}
	const UECodeGen_Private::FBoolPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_bIsBuildModeOn = { "bIsBuildModeOn", nullptr, (EPropertyFlags)0x0020080000000004, UECodeGen_Private::EPropertyGenFlags::Bool | UECodeGen_Private::EPropertyGenFlags::NativeBool, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, sizeof(bool), sizeof(UBuildComponent), &Z_Construct_UClass_UBuildComponent_Statics::NewProp_bIsBuildModeOn_SetBit, METADATA_PARAMS(Z_Construct_UClass_UBuildComponent_Statics::NewProp_bIsBuildModeOn_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UBuildComponent_Statics::NewProp_bIsBuildModeOn_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingMeshDataTable_MetaData[] = {
		{ "Category", "Build Mode Settings" },
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingMeshDataTable = { "BuildingMeshDataTable", nullptr, (EPropertyFlags)0x0020080000010005, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UBuildComponent, BuildingMeshDataTable), Z_Construct_UClass_UDataTable_NoRegister, METADATA_PARAMS(Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingMeshDataTable_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingMeshDataTable_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingTraceRange_MetaData[] = {
		{ "Category", "Build Mode Settings" },
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FUnsizedIntPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingTraceRange = { "BuildingTraceRange", nullptr, (EPropertyFlags)0x0020080000010005, UECodeGen_Private::EPropertyGenFlags::Int, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UBuildComponent, BuildingTraceRange), METADATA_PARAMS(Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingTraceRange_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingTraceRange_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UBuildComponent_Statics::NewProp_bEnableSnapping_MetaData[] = {
		{ "Category", "Build Mode Settings" },
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	void Z_Construct_UClass_UBuildComponent_Statics::NewProp_bEnableSnapping_SetBit(void* Obj)
	{
		((UBuildComponent*)Obj)->bEnableSnapping = 1;
	}
	const UECodeGen_Private::FBoolPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_bEnableSnapping = { "bEnableSnapping", nullptr, (EPropertyFlags)0x0020080000010005, UECodeGen_Private::EPropertyGenFlags::Bool | UECodeGen_Private::EPropertyGenFlags::NativeBool, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, sizeof(bool), sizeof(UBuildComponent), &Z_Construct_UClass_UBuildComponent_Statics::NewProp_bEnableSnapping_SetBit, METADATA_PARAMS(Z_Construct_UClass_UBuildComponent_Statics::NewProp_bEnableSnapping_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UBuildComponent_Statics::NewProp_bEnableSnapping_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UBuildComponent_Statics::NewProp_GridSizeInput_MetaData[] = {
		{ "Category", "Build Mode Settings" },
		{ "EditCondition", "bEnableSnapping" },
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FFloatPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_GridSizeInput = { "GridSizeInput", nullptr, (EPropertyFlags)0x0020080000010005, UECodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UBuildComponent, GridSizeInput), METADATA_PARAMS(Z_Construct_UClass_UBuildComponent_Statics::NewProp_GridSizeInput_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UBuildComponent_Statics::NewProp_GridSizeInput_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UBuildComponent_Statics::NewProp_SnappingSensitivity_MetaData[] = {
		{ "Category", "Build Mode Settings" },
		{ "EditCondition", "bEnableSnapping" },
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FFloatPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_SnappingSensitivity = { "SnappingSensitivity", nullptr, (EPropertyFlags)0x0020080000010001, UECodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UBuildComponent, SnappingSensitivity), METADATA_PARAMS(Z_Construct_UClass_UBuildComponent_Statics::NewProp_SnappingSensitivity_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UBuildComponent_Statics::NewProp_SnappingSensitivity_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UBuildComponent_Statics::NewProp_RotationSpeed_MetaData[] = {
		{ "Category", "Build Mode Settings" },
		{ "ModuleRelativePath", "BuildComponent.h" },
		{ "ToolTip", "Speed of rotation when rotate mesh called." },
	};
#endif
	const UECodeGen_Private::FFloatPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_RotationSpeed = { "RotationSpeed", nullptr, (EPropertyFlags)0x0020080000010005, UECodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UBuildComponent, RotationSpeed), METADATA_PARAMS(Z_Construct_UClass_UBuildComponent_Statics::NewProp_RotationSpeed_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UBuildComponent_Statics::NewProp_RotationSpeed_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UBuildComponent_Statics::NewProp_UpdateInterval_MetaData[] = {
		{ "Category", "Build Mode Settings" },
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FFloatPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_UpdateInterval = { "UpdateInterval", nullptr, (EPropertyFlags)0x0020080000010005, UECodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UBuildComponent, UpdateInterval), METADATA_PARAMS(Z_Construct_UClass_UBuildComponent_Statics::NewProp_UpdateInterval_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UBuildComponent_Statics::NewProp_UpdateInterval_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UBuildComponent_Statics::NewProp_Debug_MetaData[] = {
		{ "Category", "Build Mode Settings" },
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	void Z_Construct_UClass_UBuildComponent_Statics::NewProp_Debug_SetBit(void* Obj)
	{
		((UBuildComponent*)Obj)->Debug = 1;
	}
	const UECodeGen_Private::FBoolPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_Debug = { "Debug", nullptr, (EPropertyFlags)0x0020080000010005, UECodeGen_Private::EPropertyGenFlags::Bool | UECodeGen_Private::EPropertyGenFlags::NativeBool, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, sizeof(bool), sizeof(UBuildComponent), &Z_Construct_UClass_UBuildComponent_Statics::NewProp_Debug_SetBit, METADATA_PARAMS(Z_Construct_UClass_UBuildComponent_Statics::NewProp_Debug_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UBuildComponent_Statics::NewProp_Debug_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UBuildComponent_Statics::NewProp_ValidBuildMaterial_MetaData[] = {
		{ "Category", "Building Settings" },
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_ValidBuildMaterial = { "ValidBuildMaterial", nullptr, (EPropertyFlags)0x0020080000010001, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UBuildComponent, ValidBuildMaterial), Z_Construct_UClass_UMaterialInterface_NoRegister, METADATA_PARAMS(Z_Construct_UClass_UBuildComponent_Statics::NewProp_ValidBuildMaterial_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UBuildComponent_Statics::NewProp_ValidBuildMaterial_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UBuildComponent_Statics::NewProp_InvalidLocationMaterial_MetaData[] = {
		{ "Category", "Building Settings" },
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_InvalidLocationMaterial = { "InvalidLocationMaterial", nullptr, (EPropertyFlags)0x0020080000010001, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UBuildComponent, InvalidLocationMaterial), Z_Construct_UClass_UMaterialInterface_NoRegister, METADATA_PARAMS(Z_Construct_UClass_UBuildComponent_Statics::NewProp_InvalidLocationMaterial_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UBuildComponent_Statics::NewProp_InvalidLocationMaterial_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UBuildComponent_Statics::NewProp_GhostMesh_MetaData[] = {
		{ "Comment", "//Caches\n" },
		{ "ModuleRelativePath", "BuildComponent.h" },
		{ "ToolTip", "Caches" },
	};
#endif
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_GhostMesh = { "GhostMesh", nullptr, (EPropertyFlags)0x0040000000000000, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UBuildComponent, GhostMesh), Z_Construct_UClass_UStaticMesh_NoRegister, METADATA_PARAMS(Z_Construct_UClass_UBuildComponent_Statics::NewProp_GhostMesh_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UBuildComponent_Statics::NewProp_GhostMesh_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingMesh_MetaData[] = {
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingMesh = { "BuildingMesh", nullptr, (EPropertyFlags)0x0040000000000000, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UBuildComponent, BuildingMesh), Z_Construct_UClass_UStaticMesh_NoRegister, METADATA_PARAMS(Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingMesh_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingMesh_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UBuildComponent_Statics::NewProp_Owner_MetaData[] = {
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_Owner = { "Owner", nullptr, (EPropertyFlags)0x0040000000000000, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UBuildComponent, Owner), Z_Construct_UClass_AActor_NoRegister, METADATA_PARAMS(Z_Construct_UClass_UBuildComponent_Statics::NewProp_Owner_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UBuildComponent_Statics::NewProp_Owner_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UBuildComponent_Statics::NewProp_Camera_MetaData[] = {
		{ "EditInline", "true" },
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_Camera = { "Camera", nullptr, (EPropertyFlags)0x0040000000080008, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UBuildComponent, Camera), Z_Construct_UClass_UCameraComponent_NoRegister, METADATA_PARAMS(Z_Construct_UClass_UBuildComponent_Statics::NewProp_Camera_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UBuildComponent_Statics::NewProp_Camera_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UBuildComponent_Statics::NewProp_SkeletalMeshComponent_MetaData[] = {
		{ "EditInline", "true" },
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_SkeletalMeshComponent = { "SkeletalMeshComponent", nullptr, (EPropertyFlags)0x0040000000080008, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UBuildComponent, SkeletalMeshComponent), Z_Construct_UClass_USkeletalMeshComponent_NoRegister, METADATA_PARAMS(Z_Construct_UClass_UBuildComponent_Statics::NewProp_SkeletalMeshComponent_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UBuildComponent_Statics::NewProp_SkeletalMeshComponent_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingActorClass_MetaData[] = {
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FClassPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingActorClass = { "BuildingActorClass", nullptr, (EPropertyFlags)0x0044000000000000, UECodeGen_Private::EPropertyGenFlags::Class, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UBuildComponent, BuildingActorClass), Z_Construct_UClass_UClass, Z_Construct_UClass_ABuildActor_NoRegister, METADATA_PARAMS(Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingActorClass_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingActorClass_MetaData)) };
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingActors_ValueProp = { "BuildingActors", nullptr, (EPropertyFlags)0x0000000000000000, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, 1, Z_Construct_UClass_ABuildActor_NoRegister, METADATA_PARAMS(nullptr, 0) };
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingActors_Key_KeyProp = { "BuildingActors_Key", nullptr, (EPropertyFlags)0x0000000000000000, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, 0, Z_Construct_UClass_UStaticMesh_NoRegister, METADATA_PARAMS(nullptr, 0) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingActors_MetaData[] = {
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FMapPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingActors = { "BuildingActors", nullptr, (EPropertyFlags)0x0040000000000000, UECodeGen_Private::EPropertyGenFlags::Map, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UBuildComponent, BuildingActors), EMapPropertyFlags::None, METADATA_PARAMS(Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingActors_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingActors_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildGhostMesh_MetaData[] = {
		{ "EditInline", "true" },
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildGhostMesh = { "BuildGhostMesh", nullptr, (EPropertyFlags)0x0040000000080008, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UBuildComponent, BuildGhostMesh), Z_Construct_UClass_UStaticMeshComponent_NoRegister, METADATA_PARAMS(Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildGhostMesh_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildGhostMesh_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UBuildComponent_Statics::NewProp_CurrentRowIndex_MetaData[] = {
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FIntPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_CurrentRowIndex = { "CurrentRowIndex", nullptr, (EPropertyFlags)0x0040000000000000, UECodeGen_Private::EPropertyGenFlags::Int, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UBuildComponent, CurrentRowIndex), METADATA_PARAMS(Z_Construct_UClass_UBuildComponent_Statics::NewProp_CurrentRowIndex_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UBuildComponent_Statics::NewProp_CurrentRowIndex_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UBuildComponent_Statics::NewProp_CachedLineTraceResult_MetaData[] = {
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FStructPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_CachedLineTraceResult = { "CachedLineTraceResult", nullptr, (EPropertyFlags)0x0040008000000000, UECodeGen_Private::EPropertyGenFlags::Struct, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UBuildComponent, CachedLineTraceResult), Z_Construct_UScriptStruct_FHitResult, METADATA_PARAMS(Z_Construct_UClass_UBuildComponent_Statics::NewProp_CachedLineTraceResult_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UBuildComponent_Statics::NewProp_CachedLineTraceResult_MetaData)) }; // 1287526515
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UBuildComponent_Statics::NewProp_UpdateBuildComponentTimerHandle_MetaData[] = {
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FStructPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_UpdateBuildComponentTimerHandle = { "UpdateBuildComponentTimerHandle", nullptr, (EPropertyFlags)0x0040000000000000, UECodeGen_Private::EPropertyGenFlags::Struct, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UBuildComponent, UpdateBuildComponentTimerHandle), Z_Construct_UScriptStruct_FTimerHandle, METADATA_PARAMS(Z_Construct_UClass_UBuildComponent_Statics::NewProp_UpdateBuildComponentTimerHandle_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UBuildComponent_Statics::NewProp_UpdateBuildComponentTimerHandle_MetaData)) }; // 4017759265
	const UECodeGen_Private::FStructPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingMeshDataArray_Inner = { "BuildingMeshDataArray", nullptr, (EPropertyFlags)0x0000000000000000, UECodeGen_Private::EPropertyGenFlags::Struct, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, 0, Z_Construct_UScriptStruct_FBuildingMeshData, METADATA_PARAMS(nullptr, 0) }; // 2923310826
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingMeshDataArray_MetaData[] = {
		{ "ModuleRelativePath", "BuildComponent.h" },
	};
#endif
	const UECodeGen_Private::FArrayPropertyParams Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingMeshDataArray = { "BuildingMeshDataArray", nullptr, (EPropertyFlags)0x0040000000000000, UECodeGen_Private::EPropertyGenFlags::Array, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UBuildComponent, BuildingMeshDataArray), EArrayPropertyFlags::None, METADATA_PARAMS(Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingMeshDataArray_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingMeshDataArray_MetaData)) }; // 2923310826
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UClass_UBuildComponent_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_bIsBuildModeOn,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingMeshDataTable,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingTraceRange,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_bEnableSnapping,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_GridSizeInput,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_SnappingSensitivity,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_RotationSpeed,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_UpdateInterval,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_Debug,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_ValidBuildMaterial,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_InvalidLocationMaterial,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_GhostMesh,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingMesh,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_Owner,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_Camera,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_SkeletalMeshComponent,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingActorClass,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingActors_ValueProp,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingActors_Key_KeyProp,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingActors,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildGhostMesh,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_CurrentRowIndex,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_CachedLineTraceResult,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_UpdateBuildComponentTimerHandle,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingMeshDataArray_Inner,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UBuildComponent_Statics::NewProp_BuildingMeshDataArray,
	};
	const FCppClassTypeInfoStatic Z_Construct_UClass_UBuildComponent_Statics::StaticCppClassTypeInfo = {
		TCppClassTypeTraits<UBuildComponent>::IsAbstract,
	};
	const UECodeGen_Private::FClassParams Z_Construct_UClass_UBuildComponent_Statics::ClassParams = {
		&UBuildComponent::StaticClass,
		"Engine",
		&StaticCppClassTypeInfo,
		DependentSingletons,
		FuncInfo,
		Z_Construct_UClass_UBuildComponent_Statics::PropPointers,
		nullptr,
		UE_ARRAY_COUNT(DependentSingletons),
		UE_ARRAY_COUNT(FuncInfo),
		UE_ARRAY_COUNT(Z_Construct_UClass_UBuildComponent_Statics::PropPointers),
		0,
		0x00B000A4u,
		METADATA_PARAMS(Z_Construct_UClass_UBuildComponent_Statics::Class_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UClass_UBuildComponent_Statics::Class_MetaDataParams))
	};
	UClass* Z_Construct_UClass_UBuildComponent()
	{
		if (!Z_Registration_Info_UClass_UBuildComponent.OuterSingleton)
		{
			UECodeGen_Private::ConstructUClass(Z_Registration_Info_UClass_UBuildComponent.OuterSingleton, Z_Construct_UClass_UBuildComponent_Statics::ClassParams);
		}
		return Z_Registration_Info_UClass_UBuildComponent.OuterSingleton;
	}
	template<> MYPROJECT3_API UClass* StaticClass<UBuildComponent>()
	{
		return UBuildComponent::StaticClass();
	}
	DEFINE_VTABLE_PTR_HELPER_CTOR(UBuildComponent);
	UBuildComponent::~UBuildComponent() {}
	struct Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_BuildComponent_h_Statics
	{
		static const FClassRegisterCompiledInInfo ClassInfo[];
	};
	const FClassRegisterCompiledInInfo Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_BuildComponent_h_Statics::ClassInfo[] = {
		{ Z_Construct_UClass_UBuildComponent, UBuildComponent::StaticClass, TEXT("UBuildComponent"), &Z_Registration_Info_UClass_UBuildComponent, CONSTRUCT_RELOAD_VERSION_INFO(FClassReloadVersionInfo, sizeof(UBuildComponent), 3676991384U) },
	};
	static FRegisterCompiledInInfo Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_BuildComponent_h_2329778701(TEXT("/Script/MyProject3"),
		Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_BuildComponent_h_Statics::ClassInfo, UE_ARRAY_COUNT(Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_BuildComponent_h_Statics::ClassInfo),
		nullptr, 0,
		nullptr, 0);
PRAGMA_ENABLE_DEPRECATION_WARNINGS
