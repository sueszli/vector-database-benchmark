// Copyright Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
#include "MyProject3/MyProject3GameMode.h"
PRAGMA_DISABLE_DEPRECATION_WARNINGS
void EmptyLinkFunctionForGeneratedCodeMyProject3GameMode() {}
// Cross Module References
	ENGINE_API UClass* Z_Construct_UClass_AGameModeBase();
	MYPROJECT3_API UClass* Z_Construct_UClass_AMyProject3GameMode();
	MYPROJECT3_API UClass* Z_Construct_UClass_AMyProject3GameMode_NoRegister();
	UPackage* Z_Construct_UPackage__Script_MyProject3();
// End Cross Module References
	void AMyProject3GameMode::StaticRegisterNativesAMyProject3GameMode()
	{
	}
	IMPLEMENT_CLASS_NO_AUTO_REGISTRATION(AMyProject3GameMode);
	UClass* Z_Construct_UClass_AMyProject3GameMode_NoRegister()
	{
		return AMyProject3GameMode::StaticClass();
	}
	struct Z_Construct_UClass_AMyProject3GameMode_Statics
	{
		static UObject* (*const DependentSingletons[])();
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Class_MetaDataParams[];
#endif
		static const FCppClassTypeInfoStatic StaticCppClassTypeInfo;
		static const UECodeGen_Private::FClassParams ClassParams;
	};
	UObject* (*const Z_Construct_UClass_AMyProject3GameMode_Statics::DependentSingletons[])() = {
		(UObject* (*)())Z_Construct_UClass_AGameModeBase,
		(UObject* (*)())Z_Construct_UPackage__Script_MyProject3,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_AMyProject3GameMode_Statics::Class_MetaDataParams[] = {
		{ "HideCategories", "Info Rendering MovementReplication Replication Actor Input Movement Collision Rendering HLOD WorldPartition DataLayers Transformation" },
		{ "IncludePath", "MyProject3GameMode.h" },
		{ "ModuleRelativePath", "MyProject3GameMode.h" },
		{ "ShowCategories", "Input|MouseInput Input|TouchInput" },
	};
#endif
	const FCppClassTypeInfoStatic Z_Construct_UClass_AMyProject3GameMode_Statics::StaticCppClassTypeInfo = {
		TCppClassTypeTraits<AMyProject3GameMode>::IsAbstract,
	};
	const UECodeGen_Private::FClassParams Z_Construct_UClass_AMyProject3GameMode_Statics::ClassParams = {
		&AMyProject3GameMode::StaticClass,
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
		METADATA_PARAMS(Z_Construct_UClass_AMyProject3GameMode_Statics::Class_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UClass_AMyProject3GameMode_Statics::Class_MetaDataParams))
	};
	UClass* Z_Construct_UClass_AMyProject3GameMode()
	{
		if (!Z_Registration_Info_UClass_AMyProject3GameMode.OuterSingleton)
		{
			UECodeGen_Private::ConstructUClass(Z_Registration_Info_UClass_AMyProject3GameMode.OuterSingleton, Z_Construct_UClass_AMyProject3GameMode_Statics::ClassParams);
		}
		return Z_Registration_Info_UClass_AMyProject3GameMode.OuterSingleton;
	}
	template<> MYPROJECT3_API UClass* StaticClass<AMyProject3GameMode>()
	{
		return AMyProject3GameMode::StaticClass();
	}
	DEFINE_VTABLE_PTR_HELPER_CTOR(AMyProject3GameMode);
	AMyProject3GameMode::~AMyProject3GameMode() {}
	struct Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_MyProject3GameMode_h_Statics
	{
		static const FClassRegisterCompiledInInfo ClassInfo[];
	};
	const FClassRegisterCompiledInInfo Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_MyProject3GameMode_h_Statics::ClassInfo[] = {
		{ Z_Construct_UClass_AMyProject3GameMode, AMyProject3GameMode::StaticClass, TEXT("AMyProject3GameMode"), &Z_Registration_Info_UClass_AMyProject3GameMode, CONSTRUCT_RELOAD_VERSION_INFO(FClassReloadVersionInfo, sizeof(AMyProject3GameMode), 999656056U) },
	};
	static FRegisterCompiledInInfo Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_MyProject3GameMode_h_1911625142(TEXT("/Script/MyProject3"),
		Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_MyProject3GameMode_h_Statics::ClassInfo, UE_ARRAY_COUNT(Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_MyProject3GameMode_h_Statics::ClassInfo),
		nullptr, 0,
		nullptr, 0);
PRAGMA_ENABLE_DEPRECATION_WARNINGS
