// Copyright Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
#include "MultiplayerSession/Public/MultiplayerSessionSubsystem.h"
#include "Engine/Classes/Engine/GameInstance.h"
PRAGMA_DISABLE_DEPRECATION_WARNINGS
void EmptyLinkFunctionForGeneratedCodeMultiplayerSessionSubsystem() {}
// Cross Module References
	MULTIPLAYERSESSION_API UClass* Z_Construct_UClass_UMultiplayerSessionSubsystem_NoRegister();
	MULTIPLAYERSESSION_API UClass* Z_Construct_UClass_UMultiplayerSessionSubsystem();
	ENGINE_API UClass* Z_Construct_UClass_UGameInstanceSubsystem();
	UPackage* Z_Construct_UPackage__Script_MultiplayerSession();
// End Cross Module References
	void UMultiplayerSessionSubsystem::StaticRegisterNativesUMultiplayerSessionSubsystem()
	{
	}
	IMPLEMENT_CLASS_NO_AUTO_REGISTRATION(UMultiplayerSessionSubsystem);
	UClass* Z_Construct_UClass_UMultiplayerSessionSubsystem_NoRegister()
	{
		return UMultiplayerSessionSubsystem::StaticClass();
	}
	struct Z_Construct_UClass_UMultiplayerSessionSubsystem_Statics
	{
		static UObject* (*const DependentSingletons[])();
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Class_MetaDataParams[];
#endif
		static const FCppClassTypeInfoStatic StaticCppClassTypeInfo;
		static const UECodeGen_Private::FClassParams ClassParams;
	};
	UObject* (*const Z_Construct_UClass_UMultiplayerSessionSubsystem_Statics::DependentSingletons[])() = {
		(UObject* (*)())Z_Construct_UClass_UGameInstanceSubsystem,
		(UObject* (*)())Z_Construct_UPackage__Script_MultiplayerSession,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UMultiplayerSessionSubsystem_Statics::Class_MetaDataParams[] = {
		{ "Comment", "/**\n * \n */" },
		{ "IncludePath", "MultiplayerSessionSubsystem.h" },
		{ "ModuleRelativePath", "Public/MultiplayerSessionSubsystem.h" },
	};
#endif
	const FCppClassTypeInfoStatic Z_Construct_UClass_UMultiplayerSessionSubsystem_Statics::StaticCppClassTypeInfo = {
		TCppClassTypeTraits<UMultiplayerSessionSubsystem>::IsAbstract,
	};
	const UECodeGen_Private::FClassParams Z_Construct_UClass_UMultiplayerSessionSubsystem_Statics::ClassParams = {
		&UMultiplayerSessionSubsystem::StaticClass,
		nullptr,
		&StaticCppClassTypeInfo,
		DependentSingletons,
		nullptr,
		nullptr,
		nullptr,
		UE_ARRAY_COUNT(DependentSingletons),
		0,
		0,
		0,
		0x001000A0u,
		METADATA_PARAMS(Z_Construct_UClass_UMultiplayerSessionSubsystem_Statics::Class_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UClass_UMultiplayerSessionSubsystem_Statics::Class_MetaDataParams))
	};
	UClass* Z_Construct_UClass_UMultiplayerSessionSubsystem()
	{
		if (!Z_Registration_Info_UClass_UMultiplayerSessionSubsystem.OuterSingleton)
		{
			UECodeGen_Private::ConstructUClass(Z_Registration_Info_UClass_UMultiplayerSessionSubsystem.OuterSingleton, Z_Construct_UClass_UMultiplayerSessionSubsystem_Statics::ClassParams);
		}
		return Z_Registration_Info_UClass_UMultiplayerSessionSubsystem.OuterSingleton;
	}
	template<> MULTIPLAYERSESSION_API UClass* StaticClass<UMultiplayerSessionSubsystem>()
	{
		return UMultiplayerSessionSubsystem::StaticClass();
	}
	DEFINE_VTABLE_PTR_HELPER_CTOR(UMultiplayerSessionSubsystem);
	struct Z_CompiledInDeferFile_FID_HostProject_Plugins_MultiplayerSession_Source_MultiplayerSession_Public_MultiplayerSessionSubsystem_h_Statics
	{
		static const FClassRegisterCompiledInInfo ClassInfo[];
	};
	const FClassRegisterCompiledInInfo Z_CompiledInDeferFile_FID_HostProject_Plugins_MultiplayerSession_Source_MultiplayerSession_Public_MultiplayerSessionSubsystem_h_Statics::ClassInfo[] = {
		{ Z_Construct_UClass_UMultiplayerSessionSubsystem, UMultiplayerSessionSubsystem::StaticClass, TEXT("UMultiplayerSessionSubsystem"), &Z_Registration_Info_UClass_UMultiplayerSessionSubsystem, CONSTRUCT_RELOAD_VERSION_INFO(FClassReloadVersionInfo, sizeof(UMultiplayerSessionSubsystem), 2307567764U) },
	};
	static FRegisterCompiledInInfo Z_CompiledInDeferFile_FID_HostProject_Plugins_MultiplayerSession_Source_MultiplayerSession_Public_MultiplayerSessionSubsystem_h_2525912527(TEXT("/Script/MultiplayerSession"),
		Z_CompiledInDeferFile_FID_HostProject_Plugins_MultiplayerSession_Source_MultiplayerSession_Public_MultiplayerSessionSubsystem_h_Statics::ClassInfo, UE_ARRAY_COUNT(Z_CompiledInDeferFile_FID_HostProject_Plugins_MultiplayerSession_Source_MultiplayerSession_Public_MultiplayerSessionSubsystem_h_Statics::ClassInfo),
		nullptr, 0,
		nullptr, 0);
PRAGMA_ENABLE_DEPRECATION_WARNINGS
