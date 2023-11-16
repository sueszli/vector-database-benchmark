// Copyright Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
#include "MyProject3/Public/ResourceSave.h"
PRAGMA_DISABLE_DEPRECATION_WARNINGS
void EmptyLinkFunctionForGeneratedCodeResourceSave() {}
// Cross Module References
	ENGINE_API UClass* Z_Construct_UClass_USaveGame();
	ENGINE_API UScriptStruct* Z_Construct_UScriptStruct_FTableRowBase();
	MYPROJECT3_API UClass* Z_Construct_UClass_UResourceSave();
	MYPROJECT3_API UClass* Z_Construct_UClass_UResourceSave_NoRegister();
	MYPROJECT3_API UScriptStruct* Z_Construct_UScriptStruct_FBuildResource();
	MYPROJECT3_API UScriptStruct* Z_Construct_UScriptStruct_FResourceQuantity();
	UMG_API UClass* Z_Construct_UClass_UImage_NoRegister();
	UPackage* Z_Construct_UPackage__Script_MyProject3();
// End Cross Module References

static_assert(std::is_polymorphic<FBuildResource>() == std::is_polymorphic<FTableRowBase>(), "USTRUCT FBuildResource cannot be polymorphic unless super FTableRowBase is polymorphic");

	static FStructRegistrationInfo Z_Registration_Info_UScriptStruct_BuildResource;
class UScriptStruct* FBuildResource::StaticStruct()
{
	if (!Z_Registration_Info_UScriptStruct_BuildResource.OuterSingleton)
	{
		Z_Registration_Info_UScriptStruct_BuildResource.OuterSingleton = GetStaticStruct(Z_Construct_UScriptStruct_FBuildResource, Z_Construct_UPackage__Script_MyProject3(), TEXT("BuildResource"));
	}
	return Z_Registration_Info_UScriptStruct_BuildResource.OuterSingleton;
}
template<> MYPROJECT3_API UScriptStruct* StaticStruct<FBuildResource>()
{
	return FBuildResource::StaticStruct();
}
	struct Z_Construct_UScriptStruct_FBuildResource_Statics
	{
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Struct_MetaDataParams[];
#endif
		static void* NewStructOps();
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_Name_MetaData[];
#endif
		static const UECodeGen_Private::FStrPropertyParams NewProp_Name;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_Icon_MetaData[];
#endif
		static const UECodeGen_Private::FObjectPtrPropertyParams NewProp_Icon;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
		static const UECodeGen_Private::FStructParams ReturnStructParams;
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UScriptStruct_FBuildResource_Statics::Struct_MetaDataParams[] = {
		{ "BlueprintType", "true" },
		{ "ModuleRelativePath", "Public/ResourceSave.h" },
	};
#endif
	void* Z_Construct_UScriptStruct_FBuildResource_Statics::NewStructOps()
	{
		return (UScriptStruct::ICppStructOps*)new UScriptStruct::TCppStructOps<FBuildResource>();
	}
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UScriptStruct_FBuildResource_Statics::NewProp_Name_MetaData[] = {
		{ "Category", "BuildResource" },
		{ "DisplayName", "Name" },
		{ "ModuleRelativePath", "Public/ResourceSave.h" },
	};
#endif
	const UECodeGen_Private::FStrPropertyParams Z_Construct_UScriptStruct_FBuildResource_Statics::NewProp_Name = { "Name", nullptr, (EPropertyFlags)0x0010000000000005, UECodeGen_Private::EPropertyGenFlags::Str, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(FBuildResource, Name), METADATA_PARAMS(Z_Construct_UScriptStruct_FBuildResource_Statics::NewProp_Name_MetaData, UE_ARRAY_COUNT(Z_Construct_UScriptStruct_FBuildResource_Statics::NewProp_Name_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UScriptStruct_FBuildResource_Statics::NewProp_Icon_MetaData[] = {
		{ "Category", "BuildResource" },
		{ "DisplayName", "Icon" },
		{ "EditInline", "true" },
		{ "MakeStructureDefaultValue", "None" },
		{ "ModuleRelativePath", "Public/ResourceSave.h" },
	};
#endif
	const UECodeGen_Private::FObjectPtrPropertyParams Z_Construct_UScriptStruct_FBuildResource_Statics::NewProp_Icon = { "Icon", nullptr, (EPropertyFlags)0x001400000008000d, UECodeGen_Private::EPropertyGenFlags::Object | UECodeGen_Private::EPropertyGenFlags::ObjectPtr, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(FBuildResource, Icon), Z_Construct_UClass_UImage_NoRegister, METADATA_PARAMS(Z_Construct_UScriptStruct_FBuildResource_Statics::NewProp_Icon_MetaData, UE_ARRAY_COUNT(Z_Construct_UScriptStruct_FBuildResource_Statics::NewProp_Icon_MetaData)) };
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UScriptStruct_FBuildResource_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UScriptStruct_FBuildResource_Statics::NewProp_Name,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UScriptStruct_FBuildResource_Statics::NewProp_Icon,
	};
	const UECodeGen_Private::FStructParams Z_Construct_UScriptStruct_FBuildResource_Statics::ReturnStructParams = {
		(UObject* (*)())Z_Construct_UPackage__Script_MyProject3,
		Z_Construct_UScriptStruct_FTableRowBase,
		&NewStructOps,
		"BuildResource",
		sizeof(FBuildResource),
		alignof(FBuildResource),
		Z_Construct_UScriptStruct_FBuildResource_Statics::PropPointers,
		UE_ARRAY_COUNT(Z_Construct_UScriptStruct_FBuildResource_Statics::PropPointers),
		RF_Public|RF_Transient|RF_MarkAsNative,
		EStructFlags(0x00000005),
		METADATA_PARAMS(Z_Construct_UScriptStruct_FBuildResource_Statics::Struct_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UScriptStruct_FBuildResource_Statics::Struct_MetaDataParams))
	};
	UScriptStruct* Z_Construct_UScriptStruct_FBuildResource()
	{
		if (!Z_Registration_Info_UScriptStruct_BuildResource.InnerSingleton)
		{
			UECodeGen_Private::ConstructUScriptStruct(Z_Registration_Info_UScriptStruct_BuildResource.InnerSingleton, Z_Construct_UScriptStruct_FBuildResource_Statics::ReturnStructParams);
		}
		return Z_Registration_Info_UScriptStruct_BuildResource.InnerSingleton;
	}
	static FStructRegistrationInfo Z_Registration_Info_UScriptStruct_ResourceQuantity;
class UScriptStruct* FResourceQuantity::StaticStruct()
{
	if (!Z_Registration_Info_UScriptStruct_ResourceQuantity.OuterSingleton)
	{
		Z_Registration_Info_UScriptStruct_ResourceQuantity.OuterSingleton = GetStaticStruct(Z_Construct_UScriptStruct_FResourceQuantity, Z_Construct_UPackage__Script_MyProject3(), TEXT("ResourceQuantity"));
	}
	return Z_Registration_Info_UScriptStruct_ResourceQuantity.OuterSingleton;
}
template<> MYPROJECT3_API UScriptStruct* StaticStruct<FResourceQuantity>()
{
	return FResourceQuantity::StaticStruct();
}
	struct Z_Construct_UScriptStruct_FResourceQuantity_Statics
	{
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Struct_MetaDataParams[];
#endif
		static void* NewStructOps();
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_ResourceData_MetaData[];
#endif
		static const UECodeGen_Private::FStructPropertyParams NewProp_ResourceData;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_Quantity_MetaData[];
#endif
		static const UECodeGen_Private::FUnsizedIntPropertyParams NewProp_Quantity;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
		static const UECodeGen_Private::FStructParams ReturnStructParams;
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UScriptStruct_FResourceQuantity_Statics::Struct_MetaDataParams[] = {
		{ "BlueprintType", "true" },
		{ "ModuleRelativePath", "Public/ResourceSave.h" },
	};
#endif
	void* Z_Construct_UScriptStruct_FResourceQuantity_Statics::NewStructOps()
	{
		return (UScriptStruct::ICppStructOps*)new UScriptStruct::TCppStructOps<FResourceQuantity>();
	}
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UScriptStruct_FResourceQuantity_Statics::NewProp_ResourceData_MetaData[] = {
		{ "Category", "ResourceQuantity" },
		{ "DisplayName", "Resource Name" },
		{ "ModuleRelativePath", "Public/ResourceSave.h" },
	};
#endif
	const UECodeGen_Private::FStructPropertyParams Z_Construct_UScriptStruct_FResourceQuantity_Statics::NewProp_ResourceData = { "ResourceData", nullptr, (EPropertyFlags)0x0010008000000005, UECodeGen_Private::EPropertyGenFlags::Struct, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(FResourceQuantity, ResourceData), Z_Construct_UScriptStruct_FBuildResource, METADATA_PARAMS(Z_Construct_UScriptStruct_FResourceQuantity_Statics::NewProp_ResourceData_MetaData, UE_ARRAY_COUNT(Z_Construct_UScriptStruct_FResourceQuantity_Statics::NewProp_ResourceData_MetaData)) }; // 3915769165
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UScriptStruct_FResourceQuantity_Statics::NewProp_Quantity_MetaData[] = {
		{ "Category", "ResourceQuantity" },
		{ "DisplayName", "Quantity Owned" },
		{ "ModuleRelativePath", "Public/ResourceSave.h" },
	};
#endif
	const UECodeGen_Private::FUnsizedIntPropertyParams Z_Construct_UScriptStruct_FResourceQuantity_Statics::NewProp_Quantity = { "Quantity", nullptr, (EPropertyFlags)0x0010000000000005, UECodeGen_Private::EPropertyGenFlags::Int, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(FResourceQuantity, Quantity), METADATA_PARAMS(Z_Construct_UScriptStruct_FResourceQuantity_Statics::NewProp_Quantity_MetaData, UE_ARRAY_COUNT(Z_Construct_UScriptStruct_FResourceQuantity_Statics::NewProp_Quantity_MetaData)) };
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UScriptStruct_FResourceQuantity_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UScriptStruct_FResourceQuantity_Statics::NewProp_ResourceData,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UScriptStruct_FResourceQuantity_Statics::NewProp_Quantity,
	};
	const UECodeGen_Private::FStructParams Z_Construct_UScriptStruct_FResourceQuantity_Statics::ReturnStructParams = {
		(UObject* (*)())Z_Construct_UPackage__Script_MyProject3,
		nullptr,
		&NewStructOps,
		"ResourceQuantity",
		sizeof(FResourceQuantity),
		alignof(FResourceQuantity),
		Z_Construct_UScriptStruct_FResourceQuantity_Statics::PropPointers,
		UE_ARRAY_COUNT(Z_Construct_UScriptStruct_FResourceQuantity_Statics::PropPointers),
		RF_Public|RF_Transient|RF_MarkAsNative,
		EStructFlags(0x00000005),
		METADATA_PARAMS(Z_Construct_UScriptStruct_FResourceQuantity_Statics::Struct_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UScriptStruct_FResourceQuantity_Statics::Struct_MetaDataParams))
	};
	UScriptStruct* Z_Construct_UScriptStruct_FResourceQuantity()
	{
		if (!Z_Registration_Info_UScriptStruct_ResourceQuantity.InnerSingleton)
		{
			UECodeGen_Private::ConstructUScriptStruct(Z_Registration_Info_UScriptStruct_ResourceQuantity.InnerSingleton, Z_Construct_UScriptStruct_FResourceQuantity_Statics::ReturnStructParams);
		}
		return Z_Registration_Info_UScriptStruct_ResourceQuantity.InnerSingleton;
	}
	void UResourceSave::StaticRegisterNativesUResourceSave()
	{
	}
	IMPLEMENT_CLASS_NO_AUTO_REGISTRATION(UResourceSave);
	UClass* Z_Construct_UClass_UResourceSave_NoRegister()
	{
		return UResourceSave::StaticClass();
	}
	struct Z_Construct_UClass_UResourceSave_Statics
	{
		static UObject* (*const DependentSingletons[])();
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Class_MetaDataParams[];
#endif
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_SaveSlotName_MetaData[];
#endif
		static const UECodeGen_Private::FStrPropertyParams NewProp_SaveSlotName;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_UserIndex_MetaData[];
#endif
		static const UECodeGen_Private::FIntPropertyParams NewProp_UserIndex;
		static const UECodeGen_Private::FStructPropertyParams NewProp_OwnedResources_ValueProp;
		static const UECodeGen_Private::FNamePropertyParams NewProp_OwnedResources_Key_KeyProp;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_OwnedResources_MetaData[];
#endif
		static const UECodeGen_Private::FMapPropertyParams NewProp_OwnedResources;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
		static const FCppClassTypeInfoStatic StaticCppClassTypeInfo;
		static const UECodeGen_Private::FClassParams ClassParams;
	};
	UObject* (*const Z_Construct_UClass_UResourceSave_Statics::DependentSingletons[])() = {
		(UObject* (*)())Z_Construct_UClass_USaveGame,
		(UObject* (*)())Z_Construct_UPackage__Script_MyProject3,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UResourceSave_Statics::Class_MetaDataParams[] = {
		{ "IncludePath", "ResourceSave.h" },
		{ "ModuleRelativePath", "Public/ResourceSave.h" },
	};
#endif
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UResourceSave_Statics::NewProp_SaveSlotName_MetaData[] = {
		{ "Category", "Basic" },
		{ "ModuleRelativePath", "Public/ResourceSave.h" },
	};
#endif
	const UECodeGen_Private::FStrPropertyParams Z_Construct_UClass_UResourceSave_Statics::NewProp_SaveSlotName = { "SaveSlotName", nullptr, (EPropertyFlags)0x0010000000020001, UECodeGen_Private::EPropertyGenFlags::Str, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UResourceSave, SaveSlotName), METADATA_PARAMS(Z_Construct_UClass_UResourceSave_Statics::NewProp_SaveSlotName_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UResourceSave_Statics::NewProp_SaveSlotName_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UResourceSave_Statics::NewProp_UserIndex_MetaData[] = {
		{ "Category", "Basic" },
		{ "ModuleRelativePath", "Public/ResourceSave.h" },
	};
#endif
	const UECodeGen_Private::FIntPropertyParams Z_Construct_UClass_UResourceSave_Statics::NewProp_UserIndex = { "UserIndex", nullptr, (EPropertyFlags)0x0010000000020001, UECodeGen_Private::EPropertyGenFlags::Int, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UResourceSave, UserIndex), METADATA_PARAMS(Z_Construct_UClass_UResourceSave_Statics::NewProp_UserIndex_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UResourceSave_Statics::NewProp_UserIndex_MetaData)) };
	const UECodeGen_Private::FStructPropertyParams Z_Construct_UClass_UResourceSave_Statics::NewProp_OwnedResources_ValueProp = { "OwnedResources", nullptr, (EPropertyFlags)0x0000008000020001, UECodeGen_Private::EPropertyGenFlags::Struct, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, 1, Z_Construct_UScriptStruct_FResourceQuantity, METADATA_PARAMS(nullptr, 0) }; // 2626433634
	const UECodeGen_Private::FNamePropertyParams Z_Construct_UClass_UResourceSave_Statics::NewProp_OwnedResources_Key_KeyProp = { "OwnedResources_Key", nullptr, (EPropertyFlags)0x0000008000020001, UECodeGen_Private::EPropertyGenFlags::Name, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, 0, METADATA_PARAMS(nullptr, 0) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UResourceSave_Statics::NewProp_OwnedResources_MetaData[] = {
		{ "Category", "Basic" },
		{ "ModuleRelativePath", "Public/ResourceSave.h" },
	};
#endif
	const UECodeGen_Private::FMapPropertyParams Z_Construct_UClass_UResourceSave_Statics::NewProp_OwnedResources = { "OwnedResources", nullptr, (EPropertyFlags)0x0010008000020001, UECodeGen_Private::EPropertyGenFlags::Map, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(UResourceSave, OwnedResources), EMapPropertyFlags::None, METADATA_PARAMS(Z_Construct_UClass_UResourceSave_Statics::NewProp_OwnedResources_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_UResourceSave_Statics::NewProp_OwnedResources_MetaData)) }; // 2626433634
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UClass_UResourceSave_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UResourceSave_Statics::NewProp_SaveSlotName,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UResourceSave_Statics::NewProp_UserIndex,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UResourceSave_Statics::NewProp_OwnedResources_ValueProp,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UResourceSave_Statics::NewProp_OwnedResources_Key_KeyProp,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_UResourceSave_Statics::NewProp_OwnedResources,
	};
	const FCppClassTypeInfoStatic Z_Construct_UClass_UResourceSave_Statics::StaticCppClassTypeInfo = {
		TCppClassTypeTraits<UResourceSave>::IsAbstract,
	};
	const UECodeGen_Private::FClassParams Z_Construct_UClass_UResourceSave_Statics::ClassParams = {
		&UResourceSave::StaticClass,
		nullptr,
		&StaticCppClassTypeInfo,
		DependentSingletons,
		nullptr,
		Z_Construct_UClass_UResourceSave_Statics::PropPointers,
		nullptr,
		UE_ARRAY_COUNT(DependentSingletons),
		0,
		UE_ARRAY_COUNT(Z_Construct_UClass_UResourceSave_Statics::PropPointers),
		0,
		0x009000A0u,
		METADATA_PARAMS(Z_Construct_UClass_UResourceSave_Statics::Class_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UClass_UResourceSave_Statics::Class_MetaDataParams))
	};
	UClass* Z_Construct_UClass_UResourceSave()
	{
		if (!Z_Registration_Info_UClass_UResourceSave.OuterSingleton)
		{
			UECodeGen_Private::ConstructUClass(Z_Registration_Info_UClass_UResourceSave.OuterSingleton, Z_Construct_UClass_UResourceSave_Statics::ClassParams);
		}
		return Z_Registration_Info_UClass_UResourceSave.OuterSingleton;
	}
	template<> MYPROJECT3_API UClass* StaticClass<UResourceSave>()
	{
		return UResourceSave::StaticClass();
	}
	DEFINE_VTABLE_PTR_HELPER_CTOR(UResourceSave);
	UResourceSave::~UResourceSave() {}
	struct Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_Public_ResourceSave_h_Statics
	{
		static const FStructRegisterCompiledInInfo ScriptStructInfo[];
		static const FClassRegisterCompiledInInfo ClassInfo[];
	};
	const FStructRegisterCompiledInInfo Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_Public_ResourceSave_h_Statics::ScriptStructInfo[] = {
		{ FBuildResource::StaticStruct, Z_Construct_UScriptStruct_FBuildResource_Statics::NewStructOps, TEXT("BuildResource"), &Z_Registration_Info_UScriptStruct_BuildResource, CONSTRUCT_RELOAD_VERSION_INFO(FStructReloadVersionInfo, sizeof(FBuildResource), 3915769165U) },
		{ FResourceQuantity::StaticStruct, Z_Construct_UScriptStruct_FResourceQuantity_Statics::NewStructOps, TEXT("ResourceQuantity"), &Z_Registration_Info_UScriptStruct_ResourceQuantity, CONSTRUCT_RELOAD_VERSION_INFO(FStructReloadVersionInfo, sizeof(FResourceQuantity), 2626433634U) },
	};
	const FClassRegisterCompiledInInfo Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_Public_ResourceSave_h_Statics::ClassInfo[] = {
		{ Z_Construct_UClass_UResourceSave, UResourceSave::StaticClass, TEXT("UResourceSave"), &Z_Registration_Info_UClass_UResourceSave, CONSTRUCT_RELOAD_VERSION_INFO(FClassReloadVersionInfo, sizeof(UResourceSave), 435398028U) },
	};
	static FRegisterCompiledInInfo Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_Public_ResourceSave_h_1204234642(TEXT("/Script/MyProject3"),
		Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_Public_ResourceSave_h_Statics::ClassInfo, UE_ARRAY_COUNT(Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_Public_ResourceSave_h_Statics::ClassInfo),
		Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_Public_ResourceSave_h_Statics::ScriptStructInfo, UE_ARRAY_COUNT(Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_Public_ResourceSave_h_Statics::ScriptStructInfo),
		nullptr, 0);
PRAGMA_ENABLE_DEPRECATION_WARNINGS
