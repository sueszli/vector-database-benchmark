// Copyright Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
#include "MyProject3/ResourceSystem.h"
#include "MyProject3/Public/ResourceSave.h"
PRAGMA_DISABLE_DEPRECATION_WARNINGS
void EmptyLinkFunctionForGeneratedCodeResourceSystem() {}
// Cross Module References
	ENGINE_API UClass* Z_Construct_UClass_UActorComponent();
	ENGINE_API UClass* Z_Construct_UClass_UDataTable_NoRegister();
	MYPROJECT3_API UClass* Z_Construct_UClass_UResourceSystem();
	MYPROJECT3_API UClass* Z_Construct_UClass_UResourceSystem_NoRegister();
	MYPROJECT3_API UScriptStruct* Z_Construct_UScriptStruct_FBuildResource();
	MYPROJECT3_API UScriptStruct* Z_Construct_UScriptStruct_FResourceQuantity();
	UPackage* Z_Construct_UPackage__Script_MyProject3();
// End Cross Module References
	DEFINE_FUNCTION(UResourceSystem::execAddResourceToInventory)
	{
		P_GET_PROPERTY_REF(FNameProperty,Z_Param_Out_ResourceToAdd);
		P_GET_PROPERTY(FIntProperty,Z_Param_NumToAdd);
		P_FINISH;
		P_NATIVE_BEGIN;
		P_THIS->AddResourceToInventory(Z_Param_Out_ResourceToAdd,Z_Param_NumToAdd);
		P_NATIVE_END;
	}
	DEFINE_FUNCTION(UResourceSystem::execSetResource)
	{
		P_GET_STRUCT_REF(FBuildResource,Z_Param_Out_NewResource);
		P_FINISH;
		P_NATIVE_BEGIN;
		P_THIS->SetResource(Z_Param_Out_NewResource);
		P_NATIVE_END;
	}
	DEFINE_FUNCTION(UResourceSystem::execGetResourceData)
	{
		P_GET_PROPERTY(FNameProperty,Z_Param_ResourceRowName);
		P_FINISH;
		P_NATIVE_BEGIN;
		*(FBuildResource*)Z_Param__Result=P_THIS->GetResourceData(Z_Param_ResourceRowName);
		P_NATIVE_END;
	}
	DEFINE_FUNCTION(UResourceSystem::execGetResource)
	{
		P_FINISH;
		P_NATIVE_BEGIN;
		*(FBuildResource*)Z_Param__Result=P_THIS->GetResource();
		P_NATIVE_END;
	}
	DEFINE_FUNCTION(UResourceSystem::execLoadPlayerResources)
	{
		P_FINISH;
		P_NATIVE_BEGIN;
		P_THIS->LoadPlayerResources();
		P_NATIVE_END;
	}
	DEFINE_FUNCTION(UResourceSystem::execSavePlayerResources)
	{
		P_FINISH;
		P_NATIVE_BEGIN;
		P_THIS->SavePlayerResources();
		P_NATIVE_END;
	}
	void UResourceSystem::StaticRegisterNativesUResourceSystem()
	{
		UClass* Class = UResourceSystem::StaticClass();
		static const FNameNativePtrPair Funcs[] = {
			{ "AddResourceToInventory", &UResourceSystem::execAddResourceToInventory },
			{ "GetResource", &UResourceSystem::execGetResource },
			{ "GetResourceData", &UResourceSystem::execGetResourceData },
			{ "LoadPlayerResources", &UResourceSystem::execLoadPlayerResources },
			{ "SavePlayerResources", &UResourceSystem::execSavePlayerResources },
			{ "SetResource", &UResourceSystem::execSetResource },
		};
		FNativeFunctionRegistrar::RegisterFunctions(Class, Funcs, UE_ARRAY_COUNT(Funcs));
	}
	struct Z_Construct_UFunction_UResourceSystem_AddResourceToInventory_Statics
	{
		struct ResourceSystem_eventAddResourceToInventory_Parms
		{
			FName ResourceToAdd;
			int32 NumToAdd;
		};
		static const UECodeGen_Private::FNamePropertyParams NewProp_ResourceToAdd;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_NumToAdd_MetaData[];
#endif
		static const UECodeGen_Private::FIntPropertyParams NewProp_NumToAdd;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
	const UECodeGen_Private::FNamePropertyParams Z_Construct_UFunction_UResourceSystem_AddResourceToInventory_Statics::NewProp_ResourceToAdd = { "ResourceToAdd", nullptr, (EPropertyFlags)0x0010000000000180, UECodeGen_Private::EPropertyGenFlags::Name, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(ResourceSystem_eventAddResourceToInventory_Parms, ResourceToAdd), METADATA_PARAMS(nullptr, 0) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UResourceSystem_AddResourceToInventory_Statics::NewProp_NumToAdd_MetaData[] = {
		{ "NativeConst", "" },
	};
#endif
	const UECodeGen_Private::FIntPropertyParams Z_Construct_UFunction_UResourceSystem_AddResourceToInventory_Statics::NewProp_NumToAdd = { "NumToAdd", nullptr, (EPropertyFlags)0x0010000000000082, UECodeGen_Private::EPropertyGenFlags::Int, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(ResourceSystem_eventAddResourceToInventory_Parms, NumToAdd), METADATA_PARAMS(Z_Construct_UFunction_UResourceSystem_AddResourceToInventory_Statics::NewProp_NumToAdd_MetaData, UE_ARRAY_COUNT(Z_Construct_UFunction_UResourceSystem_AddResourceToInventory_Statics::NewProp_NumToAdd_MetaData)) };
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_UResourceSystem_AddResourceToInventory_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UResourceSystem_AddResourceToInventory_Statics::NewProp_ResourceToAdd,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UResourceSystem_AddResourceToInventory_Statics::NewProp_NumToAdd,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UResourceSystem_AddResourceToInventory_Statics::Function_MetaDataParams[] = {
		{ "Category", "Build Mode" },
		{ "ModuleRelativePath", "ResourceSystem.h" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_UResourceSystem_AddResourceToInventory_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_UResourceSystem, nullptr, "AddResourceToInventory", nullptr, nullptr, sizeof(Z_Construct_UFunction_UResourceSystem_AddResourceToInventory_Statics::ResourceSystem_eventAddResourceToInventory_Parms), Z_Construct_UFunction_UResourceSystem_AddResourceToInventory_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_UResourceSystem_AddResourceToInventory_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x04480401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_UResourceSystem_AddResourceToInventory_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_UResourceSystem_AddResourceToInventory_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_UResourceSystem_AddResourceToInventory()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_UResourceSystem_AddResourceToInventory_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_UResourceSystem_GetResource_Statics
	{
		struct ResourceSystem_eventGetResource_Parms
		{
			FBuildResource ReturnValue;
		};
		static const UECodeGen_Private::FStructPropertyParams NewProp_ReturnValue;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
	const UECodeGen_Private::FStructPropertyParams Z_Construct_UFunction_UResourceSystem_GetResource_Statics::NewProp_ReturnValue = { "ReturnValue", nullptr, (EPropertyFlags)0x0010008000000580, UECodeGen_Private::EPropertyGenFlags::Struct, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(ResourceSystem_eventGetResource_Parms, ReturnValue), Z_Construct_UScriptStruct_FBuildResource, METADATA_PARAMS(nullptr, 0) }; // 3915769165
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_UResourceSystem_GetResource_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UResourceSystem_GetResource_Statics::NewProp_ReturnValue,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UResourceSystem_GetResource_Statics::Function_MetaDataParams[] = {
		{ "Category", "Build Mode" },
		{ "ModuleRelativePath", "ResourceSystem.h" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_UResourceSystem_GetResource_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_UResourceSystem, nullptr, "GetResource", nullptr, nullptr, sizeof(Z_Construct_UFunction_UResourceSystem_GetResource_Statics::ResourceSystem_eventGetResource_Parms), Z_Construct_UFunction_UResourceSystem_GetResource_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_UResourceSystem_GetResource_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x54080401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_UResourceSystem_GetResource_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_UResourceSystem_GetResource_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_UResourceSystem_GetResource()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_UResourceSystem_GetResource_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_UResourceSystem_GetResourceData_Statics
	{
		struct ResourceSystem_eventGetResourceData_Parms
		{
			FName ResourceRowName;
			FBuildResource ReturnValue;
		};
		static const UECodeGen_Private::FNamePropertyParams NewProp_ResourceRowName;
		static const UECodeGen_Private::FStructPropertyParams NewProp_ReturnValue;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
	const UECodeGen_Private::FNamePropertyParams Z_Construct_UFunction_UResourceSystem_GetResourceData_Statics::NewProp_ResourceRowName = { "ResourceRowName", nullptr, (EPropertyFlags)0x0010000000000080, UECodeGen_Private::EPropertyGenFlags::Name, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(ResourceSystem_eventGetResourceData_Parms, ResourceRowName), METADATA_PARAMS(nullptr, 0) };
	const UECodeGen_Private::FStructPropertyParams Z_Construct_UFunction_UResourceSystem_GetResourceData_Statics::NewProp_ReturnValue = { "ReturnValue", nullptr, (EPropertyFlags)0x0010008000000580, UECodeGen_Private::EPropertyGenFlags::Struct, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(ResourceSystem_eventGetResourceData_Parms, ReturnValue), Z_Construct_UScriptStruct_FBuildResource, METADATA_PARAMS(nullptr, 0) }; // 3915769165
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_UResourceSystem_GetResourceData_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UResourceSystem_GetResourceData_Statics::NewProp_ResourceRowName,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UResourceSystem_GetResourceData_Statics::NewProp_ReturnValue,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UResourceSystem_GetResourceData_Statics::Function_MetaDataParams[] = {
		{ "Category", "Build Mode" },
		{ "ModuleRelativePath", "ResourceSystem.h" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_UResourceSystem_GetResourceData_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_UResourceSystem, nullptr, "GetResourceData", nullptr, nullptr, sizeof(Z_Construct_UFunction_UResourceSystem_GetResourceData_Statics::ResourceSystem_eventGetResourceData_Parms), Z_Construct_UFunction_UResourceSystem_GetResourceData_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_UResourceSystem_GetResourceData_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x54080401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_UResourceSystem_GetResourceData_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_UResourceSystem_GetResourceData_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_UResourceSystem_GetResourceData()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_UResourceSystem_GetResourceData_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_UResourceSystem_LoadPlayerResources_Statics
	{
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UResourceSystem_LoadPlayerResources_Statics::Function_MetaDataParams[] = {
		{ "Category", "Resource Save Data" },
		{ "ModuleRelativePath", "ResourceSystem.h" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_UResourceSystem_LoadPlayerResources_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_UResourceSystem, nullptr, "LoadPlayerResources", nullptr, nullptr, 0, nullptr, 0, RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x04020401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_UResourceSystem_LoadPlayerResources_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_UResourceSystem_LoadPlayerResources_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_UResourceSystem_LoadPlayerResources()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_UResourceSystem_LoadPlayerResources_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_UResourceSystem_SavePlayerResources_Statics
	{
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UResourceSystem_SavePlayerResources_Statics::Function_MetaDataParams[] = {
		{ "Category", "Resource Save Data" },
		{ "ModuleRelativePath", "ResourceSystem.h" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_UResourceSystem_SavePlayerResources_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_UResourceSystem, nullptr, "SavePlayerResources", nullptr, nullptr, 0, nullptr, 0, RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x04020401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_UResourceSystem_SavePlayerResources_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_UResourceSystem_SavePlayerResources_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_UResourceSystem_SavePlayerResources()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_UResourceSystem_SavePlayerResources_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_UResourceSystem_SetResource_Statics
	{
		struct ResourceSystem_eventSetResource_Parms
		{
			FBuildResource NewResource;
		};
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_NewResource_MetaData[];
#endif
		static const UECodeGen_Private::FStructPropertyParams NewProp_NewResource;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UResourceSystem_SetResource_Statics::NewProp_NewResource_MetaData[] = {
		{ "NativeConst", "" },
	};
#endif
	const UECodeGen_Private::FStructPropertyParams Z_Construct_UFunction_UResourceSystem_SetResource_Statics::NewProp_NewResource = { "NewResource", nullptr, (EPropertyFlags)0x0010008008000182, UECodeGen_Private::EPropertyGenFlags::Struct, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(ResourceSystem_eventSetResource_Parms, NewResource), Z_Construct_UScriptStruct_FBuildResource, METADATA_PARAMS(Z_Construct_UFunction_UResourceSystem_SetResource_Statics::NewProp_NewResource_MetaData, UE_ARRAY_COUNT(Z_Construct_UFunction_UResourceSystem_SetResource_Statics::NewProp_NewResource_MetaData)) }; // 3915769165
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_UResourceSystem_SetResource_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UResourceSystem_SetResource_Statics::NewProp_NewResource,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UResourceSystem_SetResource_Statics::Function_MetaDataParams[] = {
		{ "Category", "Build Mode" },
		{ "ModuleRelativePath", "ResourceSystem.h" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_UResourceSystem_SetResource_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_UResourceSystem, nullptr, "SetResource", nullptr, nullptr, sizeof(Z_Construct_UFunction_UResourceSystem_SetResource_Statics::ResourceSystem_eventSetResource_Parms), Z_Construct_UFunction_UResourceSystem_SetResource_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_UResourceSystem_SetResource_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x04480401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_UResourceSystem_SetResource_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_UResourceSystem_SetResource_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_UResourceSystem_SetResource()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_UResourceSystem_SetResource_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	IMPLEMENT_CLASS_NO_AUTO_REGISTRATION(UResourceSystem);
	UClass* Z_Construct_UClass_UResourceSystem_NoRegister()
	{
		return UResourceSystem::StaticClass();
	}
	struct Z_Construct_UClass_UResourceSystem_Statics
	{
		static UObject* (*const DependentSingletons[])();
		static const FClassFunctionLinkInfo FuncInfo[];
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Class_MetaDataParams[];
#endif
		static const UECodeGen_Private::FStructPropertyParams NewProp_OwnedResources_ValueProp;
		static const UECodeGen_Private::FNamePropertyParams NewProp_OwnedResources_Key_KeyProp;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_OwnedResources_MetaData[];
#endif
		static const UECodeGen_Private::FMapPropertyParams NewProp_OwnedResources;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_SaveSlotName_MetaData[];
#endif
		static const UECodeGen_Private::FStrPropertyParams NewProp_SaveSlotName;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_UserIndex_MetaData[];
#endif
		static const UECodeGen_Private::FIntPropertyParams NewProp_UserIndex;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_ResourceDataTable_MetaData[];
#endif
		static const UECodeGen_Private::FObjectPropertyParams NewProp_ResourceDataTable;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_Resource_MetaData[];
#endif
		static const UECodeGen_Private::FStructPropertyParams NewProp_Resource;
		static const UECodeGen_Private::FStructPropertyParams NewProp_CachedResources_ValueProp;
		static const UECodeGen_Private::FNamePropertyParams NewProp_CachedResources_Key_KeyProp;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_CachedResources_MetaData[];
#endif
		static const UECodeGen_Private::FMapPropertyParams NewProp_CachedResources;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
		static const FCppClassTypeInfoStatic StaticCppClassTypeInfo;
		static const UECodeGen_Private::FClassParams ClassParams;
	};
	UObject* (*const Z_Construct_UClass_UResourceSystem_Statics::DependentSingletons[])() = {
		(UObject* (*)())Z_Construct_UClass_UActorComponent,
		(UObject* (*)())Z_Construct_UPackage__Script_MyProject3,
	};
	const FClassFunctionLinkInfo Z_Construct_UClass_UResourceSystem_Statics::FuncInfo[] = {
		{ &Z_Construct_UFunction_UResourceSystem_AddResourceToInventory, "AddResourceToInventory" }, // 26898411
		{ &Z_Construct_UFunction_UResourceSystem_GetResource, "GetResource" }, // 3538532352
		{ &Z_Construct_UFunction_UResourceSystem_GetResourceData, "GetResourceData" }, // 1195656466
		{ &Z_Construct_UFunction_UResourceSystem_LoadPlayerResources, "LoadPlayerResources" }, // 3777806380
		{ &Z_Construct_UFunction_UResourceSystem_SavePlayerResources, "SavePlayerResources" }, // 715721079
		{ &Z_Construct_UFunction_UResourceSystem_SetResource, "SetResource" }, // 4083483758
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UResourceSystem_Statics::Class_MetaDataParams[] = {
		{ "BlueprintSpawnableComponent", "" },
		{ "ClassGroupNames", "Custom" },
		{ "IncludePath", "ResourceSystem.h" },
		{ "ModuleRelativePath", "ResourceSystem.h" },
	};
#endif
	const UECodeGen_Private::FStructPropertyParams Z_Construct_UClass_UResourceSystem_Statics::NewProp_OwnedResources_ValueProp = { "OwnedResources", nullptr, (EPropertyFlags)0x0000008000000000, UECodeGen_Private::EPropertyGenFlags::Struct, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, 1, Z_Construct_UScriptStruct_FResourceQuantity, METADATA_PARAMS(nullptr, 0) }; // 2626433634
	const UECodeGen_Private::FNamePropertyParams Z_Construct_UClass_UResourceSystem_Statics::NewProp_OwnedResources_Key_KeyProp = { "OwnedResources_Key", nullptr, (EPropertyFlags)0x0000008000000000, UECodeGen_Private::EPropertyGenFlags::Name, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, 0, METADATA_PARAMS(nullptr, 0) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UResourceSystem_Statics::NewProp_OwnedResources_MetaData[] = {
		{ "Category", "Resource Save Data" },
		{ "ModuleRelativePath", "ResourceSystem.h" },
	};
#endif
	const UECodeGen_Private::FMapPropertyParams Z_Construct_UClass_UResourceSystem_Statics::NewProp_OwnedResources = { "OwnedResources", nullptr, (EPropertyFlags)0x0010008000000004, UECodeGen_Private::EPropertyGenFlags::Map, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UResourceSystem, OwnedResources), EMapPropertyFlags::None, METADATA_PARAMS(Z_Construct_UClass_UResourceSystem_Statics::NewProp_OwnedResources_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UResourceSystem_Statics::NewProp_OwnedResources_MetaData)) }; // 2626433634
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UResourceSystem_Statics::NewProp_SaveSlotName_MetaData[] = {
		{ "Category", "Resource Save Data" },
		{ "ModuleRelativePath", "ResourceSystem.h" },
	};
#endif
	const UECodeGen_Private::FStrPropertyParams Z_Construct_UClass_UResourceSystem_Statics::NewProp_SaveSlotName = { "SaveSlotName", nullptr, (EPropertyFlags)0x0020080000010005, UECodeGen_Private::EPropertyGenFlags::Str, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UResourceSystem, SaveSlotName), METADATA_PARAMS(Z_Construct_UClass_UResourceSystem_Statics::NewProp_SaveSlotName_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UResourceSystem_Statics::NewProp_SaveSlotName_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UResourceSystem_Statics::NewProp_UserIndex_MetaData[] = {
		{ "Category", "Resource Save Data" },
		{ "ModuleRelativePath", "ResourceSystem.h" },
	};
#endif
	const UECodeGen_Private::FIntPropertyParams Z_Construct_UClass_UResourceSystem_Statics::NewProp_UserIndex = { "UserIndex", nullptr, (EPropertyFlags)0x0020080000010005, UECodeGen_Private::EPropertyGenFlags::Int, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UResourceSystem, UserIndex), METADATA_PARAMS(Z_Construct_UClass_UResourceSystem_Statics::NewProp_UserIndex_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UResourceSystem_Statics::NewProp_UserIndex_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UResourceSystem_Statics::NewProp_ResourceDataTable_MetaData[] = {
		{ "Category", "Build Mode" },
		{ "ModuleRelativePath", "ResourceSystem.h" },
	};
#endif
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UClass_UResourceSystem_Statics::NewProp_ResourceDataTable = { "ResourceDataTable", nullptr, (EPropertyFlags)0x0020080000010005, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UResourceSystem, ResourceDataTable), Z_Construct_UClass_UDataTable_NoRegister, METADATA_PARAMS(Z_Construct_UClass_UResourceSystem_Statics::NewProp_ResourceDataTable_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UResourceSystem_Statics::NewProp_ResourceDataTable_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UResourceSystem_Statics::NewProp_Resource_MetaData[] = {
		{ "Category", "Build Mode" },
		{ "ModuleRelativePath", "ResourceSystem.h" },
	};
#endif
	const UECodeGen_Private::FStructPropertyParams Z_Construct_UClass_UResourceSystem_Statics::NewProp_Resource = { "Resource", nullptr, (EPropertyFlags)0x0020088000010005, UECodeGen_Private::EPropertyGenFlags::Struct, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UResourceSystem, Resource), Z_Construct_UScriptStruct_FBuildResource, METADATA_PARAMS(Z_Construct_UClass_UResourceSystem_Statics::NewProp_Resource_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UResourceSystem_Statics::NewProp_Resource_MetaData)) }; // 3915769165
	const UECodeGen_Private::FStructPropertyParams Z_Construct_UClass_UResourceSystem_Statics::NewProp_CachedResources_ValueProp = { "CachedResources", nullptr, (EPropertyFlags)0x0000008000000000, UECodeGen_Private::EPropertyGenFlags::Struct, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, 1, Z_Construct_UScriptStruct_FBuildResource, METADATA_PARAMS(nullptr, 0) }; // 3915769165
	const UECodeGen_Private::FNamePropertyParams Z_Construct_UClass_UResourceSystem_Statics::NewProp_CachedResources_Key_KeyProp = { "CachedResources_Key", nullptr, (EPropertyFlags)0x0000008000000000, UECodeGen_Private::EPropertyGenFlags::Name, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, 0, METADATA_PARAMS(nullptr, 0) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UResourceSystem_Statics::NewProp_CachedResources_MetaData[] = {
		{ "ModuleRelativePath", "ResourceSystem.h" },
	};
#endif
	const UECodeGen_Private::FMapPropertyParams Z_Construct_UClass_UResourceSystem_Statics::NewProp_CachedResources = { "CachedResources", nullptr, (EPropertyFlags)0x0040008000000000, UECodeGen_Private::EPropertyGenFlags::Map, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UResourceSystem, CachedResources), EMapPropertyFlags::None, METADATA_PARAMS(Z_Construct_UClass_UResourceSystem_Statics::NewProp_CachedResources_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UResourceSystem_Statics::NewProp_CachedResources_MetaData)) }; // 3915769165
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UClass_UResourceSystem_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UResourceSystem_Statics::NewProp_OwnedResources_ValueProp,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UResourceSystem_Statics::NewProp_OwnedResources_Key_KeyProp,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UResourceSystem_Statics::NewProp_OwnedResources,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UResourceSystem_Statics::NewProp_SaveSlotName,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UResourceSystem_Statics::NewProp_UserIndex,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UResourceSystem_Statics::NewProp_ResourceDataTable,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UResourceSystem_Statics::NewProp_Resource,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UResourceSystem_Statics::NewProp_CachedResources_ValueProp,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UResourceSystem_Statics::NewProp_CachedResources_Key_KeyProp,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UResourceSystem_Statics::NewProp_CachedResources,
	};
	const FCppClassTypeInfoStatic Z_Construct_UClass_UResourceSystem_Statics::StaticCppClassTypeInfo = {
		TCppClassTypeTraits<UResourceSystem>::IsAbstract,
	};
	const UECodeGen_Private::FClassParams Z_Construct_UClass_UResourceSystem_Statics::ClassParams = {
		&UResourceSystem::StaticClass,
		"Engine",
		&StaticCppClassTypeInfo,
		DependentSingletons,
		FuncInfo,
		Z_Construct_UClass_UResourceSystem_Statics::PropPointers,
		nullptr,
		UE_ARRAY_COUNT(DependentSingletons),
		UE_ARRAY_COUNT(FuncInfo),
		UE_ARRAY_COUNT(Z_Construct_UClass_UResourceSystem_Statics::PropPointers),
		0,
		0x00B000A4u,
		METADATA_PARAMS(Z_Construct_UClass_UResourceSystem_Statics::Class_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UClass_UResourceSystem_Statics::Class_MetaDataParams))
	};
	UClass* Z_Construct_UClass_UResourceSystem()
	{
		if (!Z_Registration_Info_UClass_UResourceSystem.OuterSingleton)
		{
			UECodeGen_Private::ConstructUClass(Z_Registration_Info_UClass_UResourceSystem.OuterSingleton, Z_Construct_UClass_UResourceSystem_Statics::ClassParams);
		}
		return Z_Registration_Info_UClass_UResourceSystem.OuterSingleton;
	}
	template<> MYPROJECT3_API UClass* StaticClass<UResourceSystem>()
	{
		return UResourceSystem::StaticClass();
	}
	DEFINE_VTABLE_PTR_HELPER_CTOR(UResourceSystem);
	UResourceSystem::~UResourceSystem() {}
	struct Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_ResourceSystem_h_Statics
	{
		static const FClassRegisterCompiledInInfo ClassInfo[];
	};
	const FClassRegisterCompiledInInfo Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_ResourceSystem_h_Statics::ClassInfo[] = {
		{ Z_Construct_UClass_UResourceSystem, UResourceSystem::StaticClass, TEXT("UResourceSystem"), &Z_Registration_Info_UClass_UResourceSystem, CONSTRUCT_RELOAD_VERSION_INFO(FClassReloadVersionInfo, sizeof(UResourceSystem), 2977509299U) },
	};
	static FRegisterCompiledInInfo Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_ResourceSystem_h_2434131497(TEXT("/Script/MyProject3"),
		Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_ResourceSystem_h_Statics::ClassInfo, UE_ARRAY_COUNT(Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_ResourceSystem_h_Statics::ClassInfo),
		nullptr, 0,
		nullptr, 0);
PRAGMA_ENABLE_DEPRECATION_WARNINGS
