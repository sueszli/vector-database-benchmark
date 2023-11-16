// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
#include "AbilitySystem2/Public/AbilityTypes.h"
#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable : 4883)
#endif
PRAGMA_DISABLE_DEPRECATION_WARNINGS
void EmptyLinkFunctionForGeneratedCodeAbilityTypes() {}
// Cross Module References
	ABILITYSYSTEM2_API UEnum* Z_Construct_UEnum_AbilitySystem2_EAbilityCostType();
	UPackage* Z_Construct_UPackage__Script_AbilitySystem2();
	ABILITYSYSTEM2_API UScriptStruct* Z_Construct_UScriptStruct_FGameplayAbilityInfo();
	COREUOBJECT_API UClass* Z_Construct_UClass_UClass();
	ABILITYSYSTEM2_API UClass* Z_Construct_UClass_UCppGameplayAbilityBase_NoRegister();
	ENGINE_API UClass* Z_Construct_UClass_UMaterialInterface_NoRegister();
// End Cross Module References
	static UEnum* EAbilityCostType_StaticEnum()
	{
		static UEnum* Singleton = nullptr;
		if (!Singleton)
		{
			Singleton = GetStaticEnum(Z_Construct_UEnum_AbilitySystem2_EAbilityCostType, Z_Construct_UPackage__Script_AbilitySystem2(), TEXT("EAbilityCostType"));
		}
		return Singleton;
	}
	template<> ABILITYSYSTEM2_API UEnum* StaticEnum<EAbilityCostType>()
	{
		return EAbilityCostType_StaticEnum();
	}
	static FCompiledInDeferEnum Z_CompiledInDeferEnum_UEnum_EAbilityCostType(EAbilityCostType_StaticEnum, TEXT("/Script/AbilitySystem2"), TEXT("EAbilityCostType"), false, nullptr, nullptr);
	uint32 Get_Z_Construct_UEnum_AbilitySystem2_EAbilityCostType_Hash() { return 945962455U; }
	UEnum* Z_Construct_UEnum_AbilitySystem2_EAbilityCostType()
	{
#if WITH_HOT_RELOAD
		UPackage* Outer = Z_Construct_UPackage__Script_AbilitySystem2();
		static UEnum* ReturnEnum = FindExistingEnumIfHotReloadOrDynamic(Outer, TEXT("EAbilityCostType"), 0, Get_Z_Construct_UEnum_AbilitySystem2_EAbilityCostType_Hash(), false);
#else
		static UEnum* ReturnEnum = nullptr;
#endif // WITH_HOT_RELOAD
		if (!ReturnEnum)
		{
			static const UE4CodeGen_Private::FEnumeratorParam Enumerators[] = {
				{ "EAbilityCostType::Health", (int64)EAbilityCostType::Health },
				{ "EAbilityCostType::Mana", (int64)EAbilityCostType::Mana },
				{ "EAbilityCostType::Strength", (int64)EAbilityCostType::Strength },
			};
#if WITH_METADATA
			const UE4CodeGen_Private::FMetaDataPairParam Enum_MetaDataParams[] = {
				{ "BlueprintType", "true" },
				{ "Health.Name", "EAbilityCostType::Health" },
				{ "Mana.Name", "EAbilityCostType::Mana" },
				{ "ModuleRelativePath", "Public/AbilityTypes.h" },
				{ "Strength.Name", "EAbilityCostType::Strength" },
			};
#endif
			static const UE4CodeGen_Private::FEnumParams EnumParams = {
				(UObject*(*)())Z_Construct_UPackage__Script_AbilitySystem2,
				nullptr,
				"EAbilityCostType",
				"EAbilityCostType",
				Enumerators,
				UE_ARRAY_COUNT(Enumerators),
				RF_Public|RF_Transient|RF_MarkAsNative,
				UE4CodeGen_Private::EDynamicType::NotDynamic,
				(uint8)UEnum::ECppForm::EnumClass,
				METADATA_PARAMS(Enum_MetaDataParams, UE_ARRAY_COUNT(Enum_MetaDataParams))
			};
			UE4CodeGen_Private::ConstructUEnum(ReturnEnum, EnumParams);
		}
		return ReturnEnum;
	}
class UScriptStruct* FGameplayAbilityInfo::StaticStruct()
{
	static class UScriptStruct* Singleton = NULL;
	if (!Singleton)
	{
		extern ABILITYSYSTEM2_API uint32 Get_Z_Construct_UScriptStruct_FGameplayAbilityInfo_Hash();
		Singleton = GetStaticStruct(Z_Construct_UScriptStruct_FGameplayAbilityInfo, Z_Construct_UPackage__Script_AbilitySystem2(), TEXT("GameplayAbilityInfo"), sizeof(FGameplayAbilityInfo), Get_Z_Construct_UScriptStruct_FGameplayAbilityInfo_Hash());
	}
	return Singleton;
}
template<> ABILITYSYSTEM2_API UScriptStruct* StaticStruct<FGameplayAbilityInfo>()
{
	return FGameplayAbilityInfo::StaticStruct();
}
static FCompiledInDeferStruct Z_CompiledInDeferStruct_UScriptStruct_FGameplayAbilityInfo(FGameplayAbilityInfo::StaticStruct, TEXT("/Script/AbilitySystem2"), TEXT("GameplayAbilityInfo"), false, nullptr, nullptr);
static struct FScriptStruct_AbilitySystem2_StaticRegisterNativesFGameplayAbilityInfo
{
	FScriptStruct_AbilitySystem2_StaticRegisterNativesFGameplayAbilityInfo()
	{
		UScriptStruct::DeferCppStructOps(FName(TEXT("GameplayAbilityInfo")),new UScriptStruct::TCppStructOps<FGameplayAbilityInfo>);
	}
} ScriptStruct_AbilitySystem2_StaticRegisterNativesFGameplayAbilityInfo;
	struct Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics
	{
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam Struct_MetaDataParams[];
#endif
		static void* NewStructOps();
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam NewProp_AbilityClass_MetaData[];
#endif
		static const UE4CodeGen_Private::FClassPropertyParams NewProp_AbilityClass;
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam NewProp_UIMat_MetaData[];
#endif
		static const UE4CodeGen_Private::FObjectPropertyParams NewProp_UIMat;
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam NewProp_CostType_MetaData[];
#endif
		static const UE4CodeGen_Private::FEnumPropertyParams NewProp_CostType;
		static const UE4CodeGen_Private::FBytePropertyParams NewProp_CostType_Underlying;
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam NewProp_Cost_MetaData[];
#endif
		static const UE4CodeGen_Private::FFloatPropertyParams NewProp_Cost;
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam NewProp_CoolDownDuration_MetaData[];
#endif
		static const UE4CodeGen_Private::FFloatPropertyParams NewProp_CoolDownDuration;
		static const UE4CodeGen_Private::FPropertyParamsBase* const PropPointers[];
		static const UE4CodeGen_Private::FStructParams ReturnStructParams;
	};
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::Struct_MetaDataParams[] = {
		{ "BlueprintType", "true" },
		{ "ModuleRelativePath", "Public/AbilityTypes.h" },
	};
#endif
	void* Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewStructOps()
	{
		return (UScriptStruct::ICppStructOps*)new UScriptStruct::TCppStructOps<FGameplayAbilityInfo>();
	}
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_AbilityClass_MetaData[] = {
		{ "Category", "AbilityInfo" },
		{ "ModuleRelativePath", "Public/AbilityTypes.h" },
	};
#endif
	const UE4CodeGen_Private::FClassPropertyParams Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_AbilityClass = { "AbilityClass", nullptr, (EPropertyFlags)0x0014000000000005, UE4CodeGen_Private::EPropertyGenFlags::Class, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(FGameplayAbilityInfo, AbilityClass), Z_Construct_UClass_UCppGameplayAbilityBase_NoRegister, Z_Construct_UClass_UClass, METADATA_PARAMS(Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_AbilityClass_MetaData, UE_ARRAY_COUNT(Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_AbilityClass_MetaData)) };
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_UIMat_MetaData[] = {
		{ "Category", "AbilityInfo" },
		{ "ModuleRelativePath", "Public/AbilityTypes.h" },
	};
#endif
	const UE4CodeGen_Private::FObjectPropertyParams Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_UIMat = { "UIMat", nullptr, (EPropertyFlags)0x0010000000000005, UE4CodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(FGameplayAbilityInfo, UIMat), Z_Construct_UClass_UMaterialInterface_NoRegister, METADATA_PARAMS(Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_UIMat_MetaData, UE_ARRAY_COUNT(Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_UIMat_MetaData)) };
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_CostType_MetaData[] = {
		{ "Category", "AbilityInfo" },
		{ "ModuleRelativePath", "Public/AbilityTypes.h" },
	};
#endif
	const UE4CodeGen_Private::FEnumPropertyParams Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_CostType = { "CostType", nullptr, (EPropertyFlags)0x0010000000000005, UE4CodeGen_Private::EPropertyGenFlags::Enum, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(FGameplayAbilityInfo, CostType), Z_Construct_UEnum_AbilitySystem2_EAbilityCostType, METADATA_PARAMS(Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_CostType_MetaData, UE_ARRAY_COUNT(Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_CostType_MetaData)) };
	const UE4CodeGen_Private::FBytePropertyParams Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_CostType_Underlying = { "UnderlyingType", nullptr, (EPropertyFlags)0x0000000000000000, UE4CodeGen_Private::EPropertyGenFlags::Byte, RF_Public|RF_Transient|RF_MarkAsNative, 1, 0, nullptr, METADATA_PARAMS(nullptr, 0) };
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_Cost_MetaData[] = {
		{ "Category", "AbilityInfo" },
		{ "ModuleRelativePath", "Public/AbilityTypes.h" },
	};
#endif
	const UE4CodeGen_Private::FFloatPropertyParams Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_Cost = { "Cost", nullptr, (EPropertyFlags)0x0010000000000005, UE4CodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(FGameplayAbilityInfo, Cost), METADATA_PARAMS(Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_Cost_MetaData, UE_ARRAY_COUNT(Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_Cost_MetaData)) };
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_CoolDownDuration_MetaData[] = {
		{ "Category", "AbilityInfo" },
		{ "ModuleRelativePath", "Public/AbilityTypes.h" },
	};
#endif
	const UE4CodeGen_Private::FFloatPropertyParams Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_CoolDownDuration = { "CoolDownDuration", nullptr, (EPropertyFlags)0x0010000000000005, UE4CodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(FGameplayAbilityInfo, CoolDownDuration), METADATA_PARAMS(Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_CoolDownDuration_MetaData, UE_ARRAY_COUNT(Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_CoolDownDuration_MetaData)) };
	const UE4CodeGen_Private::FPropertyParamsBase* const Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::PropPointers[] = {
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_AbilityClass,
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_UIMat,
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_CostType,
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_CostType_Underlying,
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_Cost,
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::NewProp_CoolDownDuration,
	};
	const UE4CodeGen_Private::FStructParams Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::ReturnStructParams = {
		(UObject* (*)())Z_Construct_UPackage__Script_AbilitySystem2,
		nullptr,
		&NewStructOps,
		"GameplayAbilityInfo",
		sizeof(FGameplayAbilityInfo),
		alignof(FGameplayAbilityInfo),
		Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::PropPointers,
		UE_ARRAY_COUNT(Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::PropPointers),
		RF_Public|RF_Transient|RF_MarkAsNative,
		EStructFlags(0x00000001),
		METADATA_PARAMS(Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::Struct_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::Struct_MetaDataParams))
	};
	UScriptStruct* Z_Construct_UScriptStruct_FGameplayAbilityInfo()
	{
#if WITH_HOT_RELOAD
		extern uint32 Get_Z_Construct_UScriptStruct_FGameplayAbilityInfo_Hash();
		UPackage* Outer = Z_Construct_UPackage__Script_AbilitySystem2();
		static UScriptStruct* ReturnStruct = FindExistingStructIfHotReloadOrDynamic(Outer, TEXT("GameplayAbilityInfo"), sizeof(FGameplayAbilityInfo), Get_Z_Construct_UScriptStruct_FGameplayAbilityInfo_Hash(), false);
#else
		static UScriptStruct* ReturnStruct = nullptr;
#endif
		if (!ReturnStruct)
		{
			UE4CodeGen_Private::ConstructUScriptStruct(ReturnStruct, Z_Construct_UScriptStruct_FGameplayAbilityInfo_Statics::ReturnStructParams);
		}
		return ReturnStruct;
	}
	uint32 Get_Z_Construct_UScriptStruct_FGameplayAbilityInfo_Hash() { return 2880675741U; }
PRAGMA_ENABLE_DEPRECATION_WARNINGS
#ifdef _MSC_VER
#pragma warning (pop)
#endif
