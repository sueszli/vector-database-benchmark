// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
#include "AbilitySystem2/Public/CppGATargetActorGroundSelect.h"
#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable : 4883)
#endif
PRAGMA_DISABLE_DEPRECATION_WARNINGS
void EmptyLinkFunctionForGeneratedCodeCppGATargetActorGroundSelect() {}
// Cross Module References
	ABILITYSYSTEM2_API UClass* Z_Construct_UClass_ACppGATargetActorGroundSelect_NoRegister();
	ABILITYSYSTEM2_API UClass* Z_Construct_UClass_ACppGATargetActorGroundSelect();
	GAMEPLAYABILITIES_API UClass* Z_Construct_UClass_AGameplayAbilityTargetActor();
	UPackage* Z_Construct_UPackage__Script_AbilitySystem2();
	ABILITYSYSTEM2_API UFunction* Z_Construct_UFunction_ACppGATargetActorGroundSelect_GetPlayerLocatioinPoint();
	COREUOBJECT_API UScriptStruct* Z_Construct_UScriptStruct_FVector();
	ENGINE_API UClass* Z_Construct_UClass_UDecalComponent_NoRegister();
// End Cross Module References
	void ACppGATargetActorGroundSelect::StaticRegisterNativesACppGATargetActorGroundSelect()
	{
		UClass* Class = ACppGATargetActorGroundSelect::StaticClass();
		static const FNameNativePtrPair Funcs[] = {
			{ "GetPlayerLocatioinPoint", &ACppGATargetActorGroundSelect::execGetPlayerLocatioinPoint },
		};
		FNativeFunctionRegistrar::RegisterFunctions(Class, Funcs, UE_ARRAY_COUNT(Funcs));
	}
	struct Z_Construct_UFunction_ACppGATargetActorGroundSelect_GetPlayerLocatioinPoint_Statics
	{
		struct CppGATargetActorGroundSelect_eventGetPlayerLocatioinPoint_Parms
		{
			FVector OutViewPoint;
			bool ReturnValue;
		};
		static void NewProp_ReturnValue_SetBit(void* Obj);
		static const UE4CodeGen_Private::FBoolPropertyParams NewProp_ReturnValue;
		static const UE4CodeGen_Private::FStructPropertyParams NewProp_OutViewPoint;
		static const UE4CodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UE4CodeGen_Private::FFunctionParams FuncParams;
	};
	void Z_Construct_UFunction_ACppGATargetActorGroundSelect_GetPlayerLocatioinPoint_Statics::NewProp_ReturnValue_SetBit(void* Obj)
	{
		((CppGATargetActorGroundSelect_eventGetPlayerLocatioinPoint_Parms*)Obj)->ReturnValue = 1;
	}
	const UE4CodeGen_Private::FBoolPropertyParams Z_Construct_UFunction_ACppGATargetActorGroundSelect_GetPlayerLocatioinPoint_Statics::NewProp_ReturnValue = { "ReturnValue", nullptr, (EPropertyFlags)0x0010000000000580, UE4CodeGen_Private::EPropertyGenFlags::Bool | UE4CodeGen_Private::EPropertyGenFlags::NativeBool, RF_Public|RF_Transient|RF_MarkAsNative, 1, sizeof(bool), sizeof(CppGATargetActorGroundSelect_eventGetPlayerLocatioinPoint_Parms), &Z_Construct_UFunction_ACppGATargetActorGroundSelect_GetPlayerLocatioinPoint_Statics::NewProp_ReturnValue_SetBit, METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FStructPropertyParams Z_Construct_UFunction_ACppGATargetActorGroundSelect_GetPlayerLocatioinPoint_Statics::NewProp_OutViewPoint = { "OutViewPoint", nullptr, (EPropertyFlags)0x0010000000000180, UE4CodeGen_Private::EPropertyGenFlags::Struct, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(CppGATargetActorGroundSelect_eventGetPlayerLocatioinPoint_Parms, OutViewPoint), Z_Construct_UScriptStruct_FVector, METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_ACppGATargetActorGroundSelect_GetPlayerLocatioinPoint_Statics::PropPointers[] = {
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppGATargetActorGroundSelect_GetPlayerLocatioinPoint_Statics::NewProp_ReturnValue,
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppGATargetActorGroundSelect_GetPlayerLocatioinPoint_Statics::NewProp_OutViewPoint,
	};
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_ACppGATargetActorGroundSelect_GetPlayerLocatioinPoint_Statics::Function_MetaDataParams[] = {
		{ "Category", "GroundSelect" },
		{ "ModuleRelativePath", "Public/CppGATargetActorGroundSelect.h" },
	};
#endif
	const UE4CodeGen_Private::FFunctionParams Z_Construct_UFunction_ACppGATargetActorGroundSelect_GetPlayerLocatioinPoint_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_ACppGATargetActorGroundSelect, nullptr, "GetPlayerLocatioinPoint", nullptr, nullptr, sizeof(CppGATargetActorGroundSelect_eventGetPlayerLocatioinPoint_Parms), Z_Construct_UFunction_ACppGATargetActorGroundSelect_GetPlayerLocatioinPoint_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppGATargetActorGroundSelect_GetPlayerLocatioinPoint_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x04C20401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_ACppGATargetActorGroundSelect_GetPlayerLocatioinPoint_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppGATargetActorGroundSelect_GetPlayerLocatioinPoint_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_ACppGATargetActorGroundSelect_GetPlayerLocatioinPoint()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UE4CodeGen_Private::ConstructUFunction(ReturnFunction, Z_Construct_UFunction_ACppGATargetActorGroundSelect_GetPlayerLocatioinPoint_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	UClass* Z_Construct_UClass_ACppGATargetActorGroundSelect_NoRegister()
	{
		return ACppGATargetActorGroundSelect::StaticClass();
	}
	struct Z_Construct_UClass_ACppGATargetActorGroundSelect_Statics
	{
		static UObject* (*const DependentSingletons[])();
		static const FClassFunctionLinkInfo FuncInfo[];
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam Class_MetaDataParams[];
#endif
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam NewProp_Decal_MetaData[];
#endif
		static const UE4CodeGen_Private::FObjectPropertyParams NewProp_Decal;
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam NewProp_Radius_MetaData[];
#endif
		static const UE4CodeGen_Private::FFloatPropertyParams NewProp_Radius;
		static const UE4CodeGen_Private::FPropertyParamsBase* const PropPointers[];
		static const FCppClassTypeInfoStatic StaticCppClassTypeInfo;
		static const UE4CodeGen_Private::FClassParams ClassParams;
	};
	UObject* (*const Z_Construct_UClass_ACppGATargetActorGroundSelect_Statics::DependentSingletons[])() = {
		(UObject* (*)())Z_Construct_UClass_AGameplayAbilityTargetActor,
		(UObject* (*)())Z_Construct_UPackage__Script_AbilitySystem2,
	};
	const FClassFunctionLinkInfo Z_Construct_UClass_ACppGATargetActorGroundSelect_Statics::FuncInfo[] = {
		{ &Z_Construct_UFunction_ACppGATargetActorGroundSelect_GetPlayerLocatioinPoint, "GetPlayerLocatioinPoint" }, // 3850280365
	};
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UClass_ACppGATargetActorGroundSelect_Statics::Class_MetaDataParams[] = {
		{ "Comment", "/**\n * \n */" },
		{ "IncludePath", "CppGATargetActorGroundSelect.h" },
		{ "ModuleRelativePath", "Public/CppGATargetActorGroundSelect.h" },
	};
#endif
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UClass_ACppGATargetActorGroundSelect_Statics::NewProp_Decal_MetaData[] = {
		{ "Category", "GroundBlast" },
		{ "EditInline", "true" },
		{ "ModuleRelativePath", "Public/CppGATargetActorGroundSelect.h" },
	};
#endif
	const UE4CodeGen_Private::FObjectPropertyParams Z_Construct_UClass_ACppGATargetActorGroundSelect_Statics::NewProp_Decal = { "Decal", nullptr, (EPropertyFlags)0x00100000000a000d, UE4CodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(ACppGATargetActorGroundSelect, Decal), Z_Construct_UClass_UDecalComponent_NoRegister, METADATA_PARAMS(Z_Construct_UClass_ACppGATargetActorGroundSelect_Statics::NewProp_Decal_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_ACppGATargetActorGroundSelect_Statics::NewProp_Decal_MetaData)) };
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UClass_ACppGATargetActorGroundSelect_Statics::NewProp_Radius_MetaData[] = {
		{ "Category", "GroundSelect" },
		{ "ExposeOnSpawn", "TRUE" },
		{ "ModuleRelativePath", "Public/CppGATargetActorGroundSelect.h" },
	};
#endif
	const UE4CodeGen_Private::FFloatPropertyParams Z_Construct_UClass_ACppGATargetActorGroundSelect_Statics::NewProp_Radius = { "Radius", nullptr, (EPropertyFlags)0x0011000000000005, UE4CodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(ACppGATargetActorGroundSelect, Radius), METADATA_PARAMS(Z_Construct_UClass_ACppGATargetActorGroundSelect_Statics::NewProp_Radius_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_ACppGATargetActorGroundSelect_Statics::NewProp_Radius_MetaData)) };
	const UE4CodeGen_Private::FPropertyParamsBase* const Z_Construct_UClass_ACppGATargetActorGroundSelect_Statics::PropPointers[] = {
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_ACppGATargetActorGroundSelect_Statics::NewProp_Decal,
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_ACppGATargetActorGroundSelect_Statics::NewProp_Radius,
	};
	const FCppClassTypeInfoStatic Z_Construct_UClass_ACppGATargetActorGroundSelect_Statics::StaticCppClassTypeInfo = {
		TCppClassTypeTraits<ACppGATargetActorGroundSelect>::IsAbstract,
	};
	const UE4CodeGen_Private::FClassParams Z_Construct_UClass_ACppGATargetActorGroundSelect_Statics::ClassParams = {
		&ACppGATargetActorGroundSelect::StaticClass,
		"Engine",
		&StaticCppClassTypeInfo,
		DependentSingletons,
		FuncInfo,
		Z_Construct_UClass_ACppGATargetActorGroundSelect_Statics::PropPointers,
		nullptr,
		UE_ARRAY_COUNT(DependentSingletons),
		UE_ARRAY_COUNT(FuncInfo),
		UE_ARRAY_COUNT(Z_Construct_UClass_ACppGATargetActorGroundSelect_Statics::PropPointers),
		0,
		0x009002A4u,
		METADATA_PARAMS(Z_Construct_UClass_ACppGATargetActorGroundSelect_Statics::Class_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UClass_ACppGATargetActorGroundSelect_Statics::Class_MetaDataParams))
	};
	UClass* Z_Construct_UClass_ACppGATargetActorGroundSelect()
	{
		static UClass* OuterClass = nullptr;
		if (!OuterClass)
		{
			UE4CodeGen_Private::ConstructUClass(OuterClass, Z_Construct_UClass_ACppGATargetActorGroundSelect_Statics::ClassParams);
		}
		return OuterClass;
	}
	IMPLEMENT_CLASS(ACppGATargetActorGroundSelect, 1706445091);
	template<> ABILITYSYSTEM2_API UClass* StaticClass<ACppGATargetActorGroundSelect>()
	{
		return ACppGATargetActorGroundSelect::StaticClass();
	}
	static FCompiledInDefer Z_CompiledInDefer_UClass_ACppGATargetActorGroundSelect(Z_Construct_UClass_ACppGATargetActorGroundSelect, &ACppGATargetActorGroundSelect::StaticClass, TEXT("/Script/AbilitySystem2"), TEXT("ACppGATargetActorGroundSelect"), false, nullptr, nullptr, nullptr);
	DEFINE_VTABLE_PTR_HELPER_CTOR(ACppGATargetActorGroundSelect);
PRAGMA_ENABLE_DEPRECATION_WARNINGS
#ifdef _MSC_VER
#pragma warning (pop)
#endif
