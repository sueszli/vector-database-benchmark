// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
#include "AbilitySystem2/Public/CppTargetActorGetAround.h"
#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable : 4883)
#endif
PRAGMA_DISABLE_DEPRECATION_WARNINGS
void EmptyLinkFunctionForGeneratedCodeCppTargetActorGetAround() {}
// Cross Module References
	ABILITYSYSTEM2_API UClass* Z_Construct_UClass_ACppTargetActorGetAround_NoRegister();
	ABILITYSYSTEM2_API UClass* Z_Construct_UClass_ACppTargetActorGetAround();
	GAMEPLAYABILITIES_API UClass* Z_Construct_UClass_AGameplayAbilityTargetActor();
	UPackage* Z_Construct_UPackage__Script_AbilitySystem2();
// End Cross Module References
	void ACppTargetActorGetAround::StaticRegisterNativesACppTargetActorGetAround()
	{
	}
	UClass* Z_Construct_UClass_ACppTargetActorGetAround_NoRegister()
	{
		return ACppTargetActorGetAround::StaticClass();
	}
	struct Z_Construct_UClass_ACppTargetActorGetAround_Statics
	{
		static UObject* (*const DependentSingletons[])();
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam Class_MetaDataParams[];
#endif
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam NewProp_Radius_MetaData[];
#endif
		static const UE4CodeGen_Private::FFloatPropertyParams NewProp_Radius;
		static const UE4CodeGen_Private::FPropertyParamsBase* const PropPointers[];
		static const FCppClassTypeInfoStatic StaticCppClassTypeInfo;
		static const UE4CodeGen_Private::FClassParams ClassParams;
	};
	UObject* (*const Z_Construct_UClass_ACppTargetActorGetAround_Statics::DependentSingletons[])() = {
		(UObject* (*)())Z_Construct_UClass_AGameplayAbilityTargetActor,
		(UObject* (*)())Z_Construct_UPackage__Script_AbilitySystem2,
	};
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UClass_ACppTargetActorGetAround_Statics::Class_MetaDataParams[] = {
		{ "Comment", "/**\n * \n */" },
		{ "IncludePath", "CppTargetActorGetAround.h" },
		{ "ModuleRelativePath", "Public/CppTargetActorGetAround.h" },
	};
#endif
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UClass_ACppTargetActorGetAround_Statics::NewProp_Radius_MetaData[] = {
		{ "Category", "FindRound" },
		{ "ExposeOnSpawn", "TRUE" },
		{ "ModuleRelativePath", "Public/CppTargetActorGetAround.h" },
	};
#endif
	const UE4CodeGen_Private::FFloatPropertyParams Z_Construct_UClass_ACppTargetActorGetAround_Statics::NewProp_Radius = { "Radius", nullptr, (EPropertyFlags)0x0011000000000005, UE4CodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(ACppTargetActorGetAround, Radius), METADATA_PARAMS(Z_Construct_UClass_ACppTargetActorGetAround_Statics::NewProp_Radius_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_ACppTargetActorGetAround_Statics::NewProp_Radius_MetaData)) };
	const UE4CodeGen_Private::FPropertyParamsBase* const Z_Construct_UClass_ACppTargetActorGetAround_Statics::PropPointers[] = {
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_ACppTargetActorGetAround_Statics::NewProp_Radius,
	};
	const FCppClassTypeInfoStatic Z_Construct_UClass_ACppTargetActorGetAround_Statics::StaticCppClassTypeInfo = {
		TCppClassTypeTraits<ACppTargetActorGetAround>::IsAbstract,
	};
	const UE4CodeGen_Private::FClassParams Z_Construct_UClass_ACppTargetActorGetAround_Statics::ClassParams = {
		&ACppTargetActorGetAround::StaticClass,
		"Engine",
		&StaticCppClassTypeInfo,
		DependentSingletons,
		nullptr,
		Z_Construct_UClass_ACppTargetActorGetAround_Statics::PropPointers,
		nullptr,
		UE_ARRAY_COUNT(DependentSingletons),
		0,
		UE_ARRAY_COUNT(Z_Construct_UClass_ACppTargetActorGetAround_Statics::PropPointers),
		0,
		0x009002A4u,
		METADATA_PARAMS(Z_Construct_UClass_ACppTargetActorGetAround_Statics::Class_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UClass_ACppTargetActorGetAround_Statics::Class_MetaDataParams))
	};
	UClass* Z_Construct_UClass_ACppTargetActorGetAround()
	{
		static UClass* OuterClass = nullptr;
		if (!OuterClass)
		{
			UE4CodeGen_Private::ConstructUClass(OuterClass, Z_Construct_UClass_ACppTargetActorGetAround_Statics::ClassParams);
		}
		return OuterClass;
	}
	IMPLEMENT_CLASS(ACppTargetActorGetAround, 4235701102);
	template<> ABILITYSYSTEM2_API UClass* StaticClass<ACppTargetActorGetAround>()
	{
		return ACppTargetActorGetAround::StaticClass();
	}
	static FCompiledInDefer Z_CompiledInDefer_UClass_ACppTargetActorGetAround(Z_Construct_UClass_ACppTargetActorGetAround, &ACppTargetActorGetAround::StaticClass, TEXT("/Script/AbilitySystem2"), TEXT("ACppTargetActorGetAround"), false, nullptr, nullptr, nullptr);
	DEFINE_VTABLE_PTR_HELPER_CTOR(ACppTargetActorGetAround);
PRAGMA_ENABLE_DEPRECATION_WARNINGS
#ifdef _MSC_VER
#pragma warning (pop)
#endif
