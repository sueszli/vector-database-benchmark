// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
#include "AbilitySystem2/AbilitySystem2GameMode.h"
#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable : 4883)
#endif
PRAGMA_DISABLE_DEPRECATION_WARNINGS
void EmptyLinkFunctionForGeneratedCodeAbilitySystem2GameMode() {}
// Cross Module References
	ABILITYSYSTEM2_API UClass* Z_Construct_UClass_AAbilitySystem2GameMode_NoRegister();
	ABILITYSYSTEM2_API UClass* Z_Construct_UClass_AAbilitySystem2GameMode();
	ENGINE_API UClass* Z_Construct_UClass_AGameModeBase();
	UPackage* Z_Construct_UPackage__Script_AbilitySystem2();
// End Cross Module References
	void AAbilitySystem2GameMode::StaticRegisterNativesAAbilitySystem2GameMode()
	{
	}
	UClass* Z_Construct_UClass_AAbilitySystem2GameMode_NoRegister()
	{
		return AAbilitySystem2GameMode::StaticClass();
	}
	struct Z_Construct_UClass_AAbilitySystem2GameMode_Statics
	{
		static UObject* (*const DependentSingletons[])();
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam Class_MetaDataParams[];
#endif
		static const FCppClassTypeInfoStatic StaticCppClassTypeInfo;
		static const UE4CodeGen_Private::FClassParams ClassParams;
	};
	UObject* (*const Z_Construct_UClass_AAbilitySystem2GameMode_Statics::DependentSingletons[])() = {
		(UObject* (*)())Z_Construct_UClass_AGameModeBase,
		(UObject* (*)())Z_Construct_UPackage__Script_AbilitySystem2,
	};
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UClass_AAbilitySystem2GameMode_Statics::Class_MetaDataParams[] = {
		{ "HideCategories", "Info Rendering MovementReplication Replication Actor Input Movement Collision Rendering Utilities|Transformation" },
		{ "IncludePath", "AbilitySystem2GameMode.h" },
		{ "ModuleRelativePath", "AbilitySystem2GameMode.h" },
		{ "ShowCategories", "Input|MouseInput Input|TouchInput" },
	};
#endif
	const FCppClassTypeInfoStatic Z_Construct_UClass_AAbilitySystem2GameMode_Statics::StaticCppClassTypeInfo = {
		TCppClassTypeTraits<AAbilitySystem2GameMode>::IsAbstract,
	};
	const UE4CodeGen_Private::FClassParams Z_Construct_UClass_AAbilitySystem2GameMode_Statics::ClassParams = {
		&AAbilitySystem2GameMode::StaticClass,
		"Game",
		&StaticCppClassTypeInfo,
		DependentSingletons,
		nullptr,
		nullptr,
		nullptr,
		UE_ARRAY_COUNT(DependentSingletons),
		0,
		0,
		0,
		0x008802ACu,
		METADATA_PARAMS(Z_Construct_UClass_AAbilitySystem2GameMode_Statics::Class_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UClass_AAbilitySystem2GameMode_Statics::Class_MetaDataParams))
	};
	UClass* Z_Construct_UClass_AAbilitySystem2GameMode()
	{
		static UClass* OuterClass = nullptr;
		if (!OuterClass)
		{
			UE4CodeGen_Private::ConstructUClass(OuterClass, Z_Construct_UClass_AAbilitySystem2GameMode_Statics::ClassParams);
		}
		return OuterClass;
	}
	IMPLEMENT_CLASS(AAbilitySystem2GameMode, 3716428205);
	template<> ABILITYSYSTEM2_API UClass* StaticClass<AAbilitySystem2GameMode>()
	{
		return AAbilitySystem2GameMode::StaticClass();
	}
	static FCompiledInDefer Z_CompiledInDefer_UClass_AAbilitySystem2GameMode(Z_Construct_UClass_AAbilitySystem2GameMode, &AAbilitySystem2GameMode::StaticClass, TEXT("/Script/AbilitySystem2"), TEXT("AAbilitySystem2GameMode"), false, nullptr, nullptr, nullptr);
	DEFINE_VTABLE_PTR_HELPER_CTOR(AAbilitySystem2GameMode);
PRAGMA_ENABLE_DEPRECATION_WARNINGS
#ifdef _MSC_VER
#pragma warning (pop)
#endif
