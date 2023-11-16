// Copyright Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
#include "MyProject3/MyBuildComponent.h"
PRAGMA_DISABLE_DEPRECATION_WARNINGS
void EmptyLinkFunctionForGeneratedCodeMyBuildComponent() {}
// Cross Module References
	MYPROJECT3_API UClass* Z_Construct_UClass_UBuildComponent();
	MYPROJECT3_API UClass* Z_Construct_UClass_UMyBuildComponent();
	MYPROJECT3_API UClass* Z_Construct_UClass_UMyBuildComponent_NoRegister();
	UPackage* Z_Construct_UPackage__Script_MyProject3();
// End Cross Module References
	void UMyBuildComponent::StaticRegisterNativesUMyBuildComponent()
	{
	}
	IMPLEMENT_CLASS_NO_AUTO_REGISTRATION(UMyBuildComponent);
	UClass* Z_Construct_UClass_UMyBuildComponent_NoRegister()
	{
		return UMyBuildComponent::StaticClass();
	}
	struct Z_Construct_UClass_UMyBuildComponent_Statics
	{
		static UObject* (*const DependentSingletons[])();
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Class_MetaDataParams[];
#endif
		static const FCppClassTypeInfoStatic StaticCppClassTypeInfo;
		static const UECodeGen_Private::FClassParams ClassParams;
	};
	UObject* (*const Z_Construct_UClass_UMyBuildComponent_Statics::DependentSingletons[])() = {
		(UObject* (*)())Z_Construct_UClass_UBuildComponent,
		(UObject* (*)())Z_Construct_UPackage__Script_MyProject3,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UMyBuildComponent_Statics::Class_MetaDataParams[] = {
		{ "Comment", "/**\n * \n */" },
		{ "IncludePath", "MyBuildComponent.h" },
		{ "ModuleRelativePath", "MyBuildComponent.h" },
	};
#endif
	const FCppClassTypeInfoStatic Z_Construct_UClass_UMyBuildComponent_Statics::StaticCppClassTypeInfo = {
		TCppClassTypeTraits<UMyBuildComponent>::IsAbstract,
	};
	const UECodeGen_Private::FClassParams Z_Construct_UClass_UMyBuildComponent_Statics::ClassParams = {
		&UMyBuildComponent::StaticClass,
		"Engine",
		&StaticCppClassTypeInfo,
		DependentSingletons,
		nullptr,
		nullptr,
		nullptr,
		UE_ARRAY_COUNT(DependentSingletons),
		0,
		0,
		0,
		0x00B000A4u,
		METADATA_PARAMS(Z_Construct_UClass_UMyBuildComponent_Statics::Class_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UClass_UMyBuildComponent_Statics::Class_MetaDataParams))
	};
	UClass* Z_Construct_UClass_UMyBuildComponent()
	{
		if (!Z_Registration_Info_UClass_UMyBuildComponent.OuterSingleton)
		{
			UECodeGen_Private::ConstructUClass(Z_Registration_Info_UClass_UMyBuildComponent.OuterSingleton, Z_Construct_UClass_UMyBuildComponent_Statics::ClassParams);
		}
		return Z_Registration_Info_UClass_UMyBuildComponent.OuterSingleton;
	}
	template<> MYPROJECT3_API UClass* StaticClass<UMyBuildComponent>()
	{
		return UMyBuildComponent::StaticClass();
	}
	DEFINE_VTABLE_PTR_HELPER_CTOR(UMyBuildComponent);
	UMyBuildComponent::~UMyBuildComponent() {}
	struct Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_MyBuildComponent_h_Statics
	{
		static const FClassRegisterCompiledInInfo ClassInfo[];
	};
	const FClassRegisterCompiledInInfo Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_MyBuildComponent_h_Statics::ClassInfo[] = {
		{ Z_Construct_UClass_UMyBuildComponent, UMyBuildComponent::StaticClass, TEXT("UMyBuildComponent"), &Z_Registration_Info_UClass_UMyBuildComponent, CONSTRUCT_RELOAD_VERSION_INFO(FClassReloadVersionInfo, sizeof(UMyBuildComponent), 3105645457U) },
	};
	static FRegisterCompiledInInfo Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_MyBuildComponent_h_4033777681(TEXT("/Script/MyProject3"),
		Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_MyBuildComponent_h_Statics::ClassInfo, UE_ARRAY_COUNT(Z_CompiledInDeferFile_FID_BuildSystem_MyProject3_Source_MyProject3_MyBuildComponent_h_Statics::ClassInfo),
		nullptr, 0,
		nullptr, 0);
PRAGMA_ENABLE_DEPRECATION_WARNINGS
