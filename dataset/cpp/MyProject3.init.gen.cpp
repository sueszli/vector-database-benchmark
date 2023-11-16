// Copyright Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
PRAGMA_DISABLE_DEPRECATION_WARNINGS
void EmptyLinkFunctionForGeneratedCodeMyProject3_init() {}
	static FPackageRegistrationInfo Z_Registration_Info_UPackage__Script_MyProject3;
	FORCENOINLINE UPackage* Z_Construct_UPackage__Script_MyProject3()
	{
		if (!Z_Registration_Info_UPackage__Script_MyProject3.OuterSingleton)
		{
			static const UECodeGen_Private::FPackageParams PackageParams = {
				"/Script/MyProject3",
				nullptr,
				0,
				PKG_CompiledIn | 0x00000000,
				0xDFA9D117,
				0x502CEBB8,
				METADATA_PARAMS(nullptr, 0)
			};
			UECodeGen_Private::ConstructUPackage(Z_Registration_Info_UPackage__Script_MyProject3.OuterSingleton, PackageParams);
		}
		return Z_Registration_Info_UPackage__Script_MyProject3.OuterSingleton;
	}
	static FRegisterCompiledInInfo Z_CompiledInDeferPackage_UPackage__Script_MyProject3(Z_Construct_UPackage__Script_MyProject3, TEXT("/Script/MyProject3"), Z_Registration_Info_UPackage__Script_MyProject3, CONSTRUCT_RELOAD_VERSION_INFO(FPackageReloadVersionInfo, 0xDFA9D117, 0x502CEBB8));
PRAGMA_ENABLE_DEPRECATION_WARNINGS
