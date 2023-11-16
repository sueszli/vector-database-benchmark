// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
#include "AbilitySystem2/Public/CppPlayerControllerBase.h"
#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable : 4883)
#endif
PRAGMA_DISABLE_DEPRECATION_WARNINGS
void EmptyLinkFunctionForGeneratedCodeCppPlayerControllerBase() {}
// Cross Module References
	ABILITYSYSTEM2_API UClass* Z_Construct_UClass_ACppPlayerControllerBase_NoRegister();
	ABILITYSYSTEM2_API UClass* Z_Construct_UClass_ACppPlayerControllerBase();
	ENGINE_API UClass* Z_Construct_UClass_APlayerController();
	UPackage* Z_Construct_UPackage__Script_AbilitySystem2();
	ABILITYSYSTEM2_API UFunction* Z_Construct_UFunction_ACppPlayerControllerBase_AddAbilityToUI();
	ABILITYSYSTEM2_API UScriptStruct* Z_Construct_UScriptStruct_FGameplayAbilityInfo();
// End Cross Module References
	static FName NAME_ACppPlayerControllerBase_AddAbilityToUI = FName(TEXT("AddAbilityToUI"));
	void ACppPlayerControllerBase::AddAbilityToUI(FGameplayAbilityInfo AbilityInfo)
	{
		CppPlayerControllerBase_eventAddAbilityToUI_Parms Parms;
		Parms.AbilityInfo=AbilityInfo;
		ProcessEvent(FindFunctionChecked(NAME_ACppPlayerControllerBase_AddAbilityToUI),&Parms);
	}
	void ACppPlayerControllerBase::StaticRegisterNativesACppPlayerControllerBase()
	{
	}
	struct Z_Construct_UFunction_ACppPlayerControllerBase_AddAbilityToUI_Statics
	{
		static const UE4CodeGen_Private::FStructPropertyParams NewProp_AbilityInfo;
		static const UE4CodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UE4CodeGen_Private::FFunctionParams FuncParams;
	};
	const UE4CodeGen_Private::FStructPropertyParams Z_Construct_UFunction_ACppPlayerControllerBase_AddAbilityToUI_Statics::NewProp_AbilityInfo = { "AbilityInfo", nullptr, (EPropertyFlags)0x0010000000000080, UE4CodeGen_Private::EPropertyGenFlags::Struct, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(CppPlayerControllerBase_eventAddAbilityToUI_Parms, AbilityInfo), Z_Construct_UScriptStruct_FGameplayAbilityInfo, METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_ACppPlayerControllerBase_AddAbilityToUI_Statics::PropPointers[] = {
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppPlayerControllerBase_AddAbilityToUI_Statics::NewProp_AbilityInfo,
	};
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_ACppPlayerControllerBase_AddAbilityToUI_Statics::Function_MetaDataParams[] = {
		{ "Category", "PlayerControllerBase" },
		{ "ModuleRelativePath", "Public/CppPlayerControllerBase.h" },
	};
#endif
	const UE4CodeGen_Private::FFunctionParams Z_Construct_UFunction_ACppPlayerControllerBase_AddAbilityToUI_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_ACppPlayerControllerBase, nullptr, "AddAbilityToUI", nullptr, nullptr, sizeof(CppPlayerControllerBase_eventAddAbilityToUI_Parms), Z_Construct_UFunction_ACppPlayerControllerBase_AddAbilityToUI_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppPlayerControllerBase_AddAbilityToUI_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x08020800, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_ACppPlayerControllerBase_AddAbilityToUI_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppPlayerControllerBase_AddAbilityToUI_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_ACppPlayerControllerBase_AddAbilityToUI()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UE4CodeGen_Private::ConstructUFunction(ReturnFunction, Z_Construct_UFunction_ACppPlayerControllerBase_AddAbilityToUI_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	UClass* Z_Construct_UClass_ACppPlayerControllerBase_NoRegister()
	{
		return ACppPlayerControllerBase::StaticClass();
	}
	struct Z_Construct_UClass_ACppPlayerControllerBase_Statics
	{
		static UObject* (*const DependentSingletons[])();
		static const FClassFunctionLinkInfo FuncInfo[];
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam Class_MetaDataParams[];
#endif
		static const FCppClassTypeInfoStatic StaticCppClassTypeInfo;
		static const UE4CodeGen_Private::FClassParams ClassParams;
	};
	UObject* (*const Z_Construct_UClass_ACppPlayerControllerBase_Statics::DependentSingletons[])() = {
		(UObject* (*)())Z_Construct_UClass_APlayerController,
		(UObject* (*)())Z_Construct_UPackage__Script_AbilitySystem2,
	};
	const FClassFunctionLinkInfo Z_Construct_UClass_ACppPlayerControllerBase_Statics::FuncInfo[] = {
		{ &Z_Construct_UFunction_ACppPlayerControllerBase_AddAbilityToUI, "AddAbilityToUI" }, // 4132120013
	};
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UClass_ACppPlayerControllerBase_Statics::Class_MetaDataParams[] = {
		{ "Comment", "/**\n * \n */" },
		{ "HideCategories", "Collision Rendering Utilities|Transformation" },
		{ "IncludePath", "CppPlayerControllerBase.h" },
		{ "ModuleRelativePath", "Public/CppPlayerControllerBase.h" },
	};
#endif
	const FCppClassTypeInfoStatic Z_Construct_UClass_ACppPlayerControllerBase_Statics::StaticCppClassTypeInfo = {
		TCppClassTypeTraits<ACppPlayerControllerBase>::IsAbstract,
	};
	const UE4CodeGen_Private::FClassParams Z_Construct_UClass_ACppPlayerControllerBase_Statics::ClassParams = {
		&ACppPlayerControllerBase::StaticClass,
		"Game",
		&StaticCppClassTypeInfo,
		DependentSingletons,
		FuncInfo,
		nullptr,
		nullptr,
		UE_ARRAY_COUNT(DependentSingletons),
		UE_ARRAY_COUNT(FuncInfo),
		0,
		0,
		0x009002A4u,
		METADATA_PARAMS(Z_Construct_UClass_ACppPlayerControllerBase_Statics::Class_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UClass_ACppPlayerControllerBase_Statics::Class_MetaDataParams))
	};
	UClass* Z_Construct_UClass_ACppPlayerControllerBase()
	{
		static UClass* OuterClass = nullptr;
		if (!OuterClass)
		{
			UE4CodeGen_Private::ConstructUClass(OuterClass, Z_Construct_UClass_ACppPlayerControllerBase_Statics::ClassParams);
		}
		return OuterClass;
	}
	IMPLEMENT_CLASS(ACppPlayerControllerBase, 1143480705);
	template<> ABILITYSYSTEM2_API UClass* StaticClass<ACppPlayerControllerBase>()
	{
		return ACppPlayerControllerBase::StaticClass();
	}
	static FCompiledInDefer Z_CompiledInDefer_UClass_ACppPlayerControllerBase(Z_Construct_UClass_ACppPlayerControllerBase, &ACppPlayerControllerBase::StaticClass, TEXT("/Script/AbilitySystem2"), TEXT("ACppPlayerControllerBase"), false, nullptr, nullptr, nullptr);
	DEFINE_VTABLE_PTR_HELPER_CTOR(ACppPlayerControllerBase);
PRAGMA_ENABLE_DEPRECATION_WARNINGS
#ifdef _MSC_VER
#pragma warning (pop)
#endif
