// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
#include "AbilitySystem2/AbilitySystem2HUD.h"
#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable : 4883)
#endif
PRAGMA_DISABLE_DEPRECATION_WARNINGS
void EmptyLinkFunctionForGeneratedCodeAbilitySystem2HUD() {}
// Cross Module References
	ABILITYSYSTEM2_API UClass* Z_Construct_UClass_AAbilitySystem2HUD_NoRegister();
	ABILITYSYSTEM2_API UClass* Z_Construct_UClass_AAbilitySystem2HUD();
	ENGINE_API UClass* Z_Construct_UClass_AHUD();
	UPackage* Z_Construct_UPackage__Script_AbilitySystem2();
// End Cross Module References
	void AAbilitySystem2HUD::StaticRegisterNativesAAbilitySystem2HUD()
	{
	}
	UClass* Z_Construct_UClass_AAbilitySystem2HUD_NoRegister()
	{
		return AAbilitySystem2HUD::StaticClass();
	}
	struct Z_Construct_UClass_AAbilitySystem2HUD_Statics
	{
		static UObject* (*const DependentSingletons[])();
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam Class_MetaDataParams[];
#endif
		static const FCppClassTypeInfoStatic StaticCppClassTypeInfo;
		static const UE4CodeGen_Private::FClassParams ClassParams;
	};
	UObject* (*const Z_Construct_UClass_AAbilitySystem2HUD_Statics::DependentSingletons[])() = {
		(UObject* (*)())Z_Construct_UClass_AHUD,
		(UObject* (*)())Z_Construct_UPackage__Script_AbilitySystem2,
	};
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UClass_AAbilitySystem2HUD_Statics::Class_MetaDataParams[] = {
		{ "HideCategories", "Rendering Actor Input Replication" },
		{ "IncludePath", "AbilitySystem2HUD.h" },
		{ "ModuleRelativePath", "AbilitySystem2HUD.h" },
		{ "ShowCategories", "Input|MouseInput Input|TouchInput" },
	};
#endif
	const FCppClassTypeInfoStatic Z_Construct_UClass_AAbilitySystem2HUD_Statics::StaticCppClassTypeInfo = {
		TCppClassTypeTraits<AAbilitySystem2HUD>::IsAbstract,
	};
	const UE4CodeGen_Private::FClassParams Z_Construct_UClass_AAbilitySystem2HUD_Statics::ClassParams = {
		&AAbilitySystem2HUD::StaticClass,
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
		0x008002ACu,
		METADATA_PARAMS(Z_Construct_UClass_AAbilitySystem2HUD_Statics::Class_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UClass_AAbilitySystem2HUD_Statics::Class_MetaDataParams))
	};
	UClass* Z_Construct_UClass_AAbilitySystem2HUD()
	{
		static UClass* OuterClass = nullptr;
		if (!OuterClass)
		{
			UE4CodeGen_Private::ConstructUClass(OuterClass, Z_Construct_UClass_AAbilitySystem2HUD_Statics::ClassParams);
		}
		return OuterClass;
	}
	IMPLEMENT_CLASS(AAbilitySystem2HUD, 3052689806);
	template<> ABILITYSYSTEM2_API UClass* StaticClass<AAbilitySystem2HUD>()
	{
		return AAbilitySystem2HUD::StaticClass();
	}
	static FCompiledInDefer Z_CompiledInDefer_UClass_AAbilitySystem2HUD(Z_Construct_UClass_AAbilitySystem2HUD, &AAbilitySystem2HUD::StaticClass, TEXT("/Script/AbilitySystem2"), TEXT("AAbilitySystem2HUD"), false, nullptr, nullptr, nullptr);
	DEFINE_VTABLE_PTR_HELPER_CTOR(AAbilitySystem2HUD);
PRAGMA_ENABLE_DEPRECATION_WARNINGS
#ifdef _MSC_VER
#pragma warning (pop)
#endif
