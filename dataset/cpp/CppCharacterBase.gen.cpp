// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
#include "AbilitySystem2/Public/CppCharacterBase.h"
#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable : 4883)
#endif
PRAGMA_DISABLE_DEPRECATION_WARNINGS
void EmptyLinkFunctionForGeneratedCodeCppCharacterBase() {}
// Cross Module References
	ABILITYSYSTEM2_API UClass* Z_Construct_UClass_ACppCharacterBase_NoRegister();
	ABILITYSYSTEM2_API UClass* Z_Construct_UClass_ACppCharacterBase();
	ENGINE_API UClass* Z_Construct_UClass_ACharacter();
	UPackage* Z_Construct_UPackage__Script_AbilitySystem2();
	ABILITYSYSTEM2_API UFunction* Z_Construct_UFunction_ACppCharacterBase_AddGameplayTag();
	GAMEPLAYTAGS_API UScriptStruct* Z_Construct_UScriptStruct_FGameplayTag();
	ABILITYSYSTEM2_API UFunction* Z_Construct_UFunction_ACppCharacterBase_BP_Die();
	ABILITYSYSTEM2_API UFunction* Z_Construct_UFunction_ACppCharacterBase_BP_OnHelathChanged();
	ABILITYSYSTEM2_API UFunction* Z_Construct_UFunction_ACppCharacterBase_BP_OnManaChanged();
	ABILITYSYSTEM2_API UFunction* Z_Construct_UFunction_ACppCharacterBase_BP_OnStrengthChanged();
	ABILITYSYSTEM2_API UFunction* Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbility();
	COREUOBJECT_API UClass* Z_Construct_UClass_UClass();
	GAMEPLAYABILITIES_API UClass* Z_Construct_UClass_UGameplayAbility_NoRegister();
	ABILITYSYSTEM2_API UFunction* Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbilitys();
	ABILITYSYSTEM2_API UFunction* Z_Construct_UFunction_ACppCharacterBase_EvCpp_HitStun();
	ABILITYSYSTEM2_API UFunction* Z_Construct_UFunction_ACppCharacterBase_FCppIsOtherHosttile();
	ABILITYSYSTEM2_API UFunction* Z_Construct_UFunction_ACppCharacterBase_OnHealthChanged();
	ABILITYSYSTEM2_API UFunction* Z_Construct_UFunction_ACppCharacterBase_OnManaChanged();
	ABILITYSYSTEM2_API UFunction* Z_Construct_UFunction_ACppCharacterBase_OnStrengthChanged();
	ABILITYSYSTEM2_API UFunction* Z_Construct_UFunction_ACppCharacterBase_RemoveGameplayTag();
	ABILITYSYSTEM2_API UClass* Z_Construct_UClass_UCppAttributeSetBase_NoRegister();
	GAMEPLAYABILITIES_API UClass* Z_Construct_UClass_UAbilitySystemComponent_NoRegister();
	GAMEPLAYABILITIES_API UClass* Z_Construct_UClass_UAbilitySystemInterface_NoRegister();
// End Cross Module References
	static FName NAME_ACppCharacterBase_BP_Die = FName(TEXT("BP_Die"));
	void ACppCharacterBase::BP_Die()
	{
		ProcessEvent(FindFunctionChecked(NAME_ACppCharacterBase_BP_Die),NULL);
	}
	static FName NAME_ACppCharacterBase_BP_OnHelathChanged = FName(TEXT("BP_OnHelathChanged"));
	void ACppCharacterBase::BP_OnHelathChanged(float Health, float MaxHealth, float percentage)
	{
		CppCharacterBase_eventBP_OnHelathChanged_Parms Parms;
		Parms.Health=Health;
		Parms.MaxHealth=MaxHealth;
		Parms.percentage=percentage;
		ProcessEvent(FindFunctionChecked(NAME_ACppCharacterBase_BP_OnHelathChanged),&Parms);
	}
	static FName NAME_ACppCharacterBase_BP_OnManaChanged = FName(TEXT("BP_OnManaChanged"));
	void ACppCharacterBase::BP_OnManaChanged(float Mana, float MaxMana, float percentage)
	{
		CppCharacterBase_eventBP_OnManaChanged_Parms Parms;
		Parms.Mana=Mana;
		Parms.MaxMana=MaxMana;
		Parms.percentage=percentage;
		ProcessEvent(FindFunctionChecked(NAME_ACppCharacterBase_BP_OnManaChanged),&Parms);
	}
	static FName NAME_ACppCharacterBase_BP_OnStrengthChanged = FName(TEXT("BP_OnStrengthChanged"));
	void ACppCharacterBase::BP_OnStrengthChanged(float Strength, float MaxStrength, float percentage)
	{
		CppCharacterBase_eventBP_OnStrengthChanged_Parms Parms;
		Parms.Strength=Strength;
		Parms.MaxStrength=MaxStrength;
		Parms.percentage=percentage;
		ProcessEvent(FindFunctionChecked(NAME_ACppCharacterBase_BP_OnStrengthChanged),&Parms);
	}
	void ACppCharacterBase::StaticRegisterNativesACppCharacterBase()
	{
		UClass* Class = ACppCharacterBase::StaticClass();
		static const FNameNativePtrPair Funcs[] = {
			{ "AddGameplayTag", &ACppCharacterBase::execAddGameplayTag },
			{ "EvCpp_AquireAbility", &ACppCharacterBase::execEvCpp_AquireAbility },
			{ "EvCpp_AquireAbilitys", &ACppCharacterBase::execEvCpp_AquireAbilitys },
			{ "EvCpp_HitStun", &ACppCharacterBase::execEvCpp_HitStun },
			{ "FCppIsOtherHosttile", &ACppCharacterBase::execFCppIsOtherHosttile },
			{ "OnHealthChanged", &ACppCharacterBase::execOnHealthChanged },
			{ "OnManaChanged", &ACppCharacterBase::execOnManaChanged },
			{ "OnStrengthChanged", &ACppCharacterBase::execOnStrengthChanged },
			{ "RemoveGameplayTag", &ACppCharacterBase::execRemoveGameplayTag },
		};
		FNativeFunctionRegistrar::RegisterFunctions(Class, Funcs, UE_ARRAY_COUNT(Funcs));
	}
	struct Z_Construct_UFunction_ACppCharacterBase_AddGameplayTag_Statics
	{
		struct CppCharacterBase_eventAddGameplayTag_Parms
		{
			FGameplayTag TagToAdd;
		};
		static const UE4CodeGen_Private::FStructPropertyParams NewProp_TagToAdd;
		static const UE4CodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UE4CodeGen_Private::FFunctionParams FuncParams;
	};
	const UE4CodeGen_Private::FStructPropertyParams Z_Construct_UFunction_ACppCharacterBase_AddGameplayTag_Statics::NewProp_TagToAdd = { "TagToAdd", nullptr, (EPropertyFlags)0x0010000000000180, UE4CodeGen_Private::EPropertyGenFlags::Struct, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(CppCharacterBase_eventAddGameplayTag_Parms, TagToAdd), Z_Construct_UScriptStruct_FGameplayTag, METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_ACppCharacterBase_AddGameplayTag_Statics::PropPointers[] = {
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppCharacterBase_AddGameplayTag_Statics::NewProp_TagToAdd,
	};
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_ACppCharacterBase_AddGameplayTag_Statics::Function_MetaDataParams[] = {
		{ "Category", "Abilities" },
		{ "ModuleRelativePath", "Public/CppCharacterBase.h" },
	};
#endif
	const UE4CodeGen_Private::FFunctionParams Z_Construct_UFunction_ACppCharacterBase_AddGameplayTag_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_ACppCharacterBase, nullptr, "AddGameplayTag", nullptr, nullptr, sizeof(CppCharacterBase_eventAddGameplayTag_Parms), Z_Construct_UFunction_ACppCharacterBase_AddGameplayTag_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppCharacterBase_AddGameplayTag_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x04420401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_ACppCharacterBase_AddGameplayTag_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppCharacterBase_AddGameplayTag_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_ACppCharacterBase_AddGameplayTag()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UE4CodeGen_Private::ConstructUFunction(ReturnFunction, Z_Construct_UFunction_ACppCharacterBase_AddGameplayTag_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_ACppCharacterBase_BP_Die_Statics
	{
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UE4CodeGen_Private::FFunctionParams FuncParams;
	};
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_ACppCharacterBase_BP_Die_Statics::Function_MetaDataParams[] = {
		{ "Category", "Abilities" },
		{ "Comment", "/** END changes param */" },
		{ "DisplayName", "EvCppDie" },
		{ "ModuleRelativePath", "Public/CppCharacterBase.h" },
		{ "ToolTip", "END changes param" },
	};
#endif
	const UE4CodeGen_Private::FFunctionParams Z_Construct_UFunction_ACppCharacterBase_BP_Die_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_ACppCharacterBase, nullptr, "BP_Die", nullptr, nullptr, 0, nullptr, 0, RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x08020800, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_ACppCharacterBase_BP_Die_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppCharacterBase_BP_Die_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_ACppCharacterBase_BP_Die()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UE4CodeGen_Private::ConstructUFunction(ReturnFunction, Z_Construct_UFunction_ACppCharacterBase_BP_Die_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_ACppCharacterBase_BP_OnHelathChanged_Statics
	{
		static const UE4CodeGen_Private::FFloatPropertyParams NewProp_percentage;
		static const UE4CodeGen_Private::FFloatPropertyParams NewProp_MaxHealth;
		static const UE4CodeGen_Private::FFloatPropertyParams NewProp_Health;
		static const UE4CodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UE4CodeGen_Private::FFunctionParams FuncParams;
	};
	const UE4CodeGen_Private::FFloatPropertyParams Z_Construct_UFunction_ACppCharacterBase_BP_OnHelathChanged_Statics::NewProp_percentage = { "percentage", nullptr, (EPropertyFlags)0x0010000000000080, UE4CodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(CppCharacterBase_eventBP_OnHelathChanged_Parms, percentage), METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FFloatPropertyParams Z_Construct_UFunction_ACppCharacterBase_BP_OnHelathChanged_Statics::NewProp_MaxHealth = { "MaxHealth", nullptr, (EPropertyFlags)0x0010000000000080, UE4CodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(CppCharacterBase_eventBP_OnHelathChanged_Parms, MaxHealth), METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FFloatPropertyParams Z_Construct_UFunction_ACppCharacterBase_BP_OnHelathChanged_Statics::NewProp_Health = { "Health", nullptr, (EPropertyFlags)0x0010000000000080, UE4CodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(CppCharacterBase_eventBP_OnHelathChanged_Parms, Health), METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_ACppCharacterBase_BP_OnHelathChanged_Statics::PropPointers[] = {
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppCharacterBase_BP_OnHelathChanged_Statics::NewProp_percentage,
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppCharacterBase_BP_OnHelathChanged_Statics::NewProp_MaxHealth,
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppCharacterBase_BP_OnHelathChanged_Statics::NewProp_Health,
	};
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_ACppCharacterBase_BP_OnHelathChanged_Statics::Function_MetaDataParams[] = {
		{ "Category", "Abilities" },
		{ "DisplayName", "EvCppOnHealthChanged" },
		{ "ModuleRelativePath", "Public/CppCharacterBase.h" },
	};
#endif
	const UE4CodeGen_Private::FFunctionParams Z_Construct_UFunction_ACppCharacterBase_BP_OnHelathChanged_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_ACppCharacterBase, nullptr, "BP_OnHelathChanged", nullptr, nullptr, sizeof(CppCharacterBase_eventBP_OnHelathChanged_Parms), Z_Construct_UFunction_ACppCharacterBase_BP_OnHelathChanged_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppCharacterBase_BP_OnHelathChanged_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x08020800, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_ACppCharacterBase_BP_OnHelathChanged_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppCharacterBase_BP_OnHelathChanged_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_ACppCharacterBase_BP_OnHelathChanged()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UE4CodeGen_Private::ConstructUFunction(ReturnFunction, Z_Construct_UFunction_ACppCharacterBase_BP_OnHelathChanged_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_ACppCharacterBase_BP_OnManaChanged_Statics
	{
		static const UE4CodeGen_Private::FFloatPropertyParams NewProp_percentage;
		static const UE4CodeGen_Private::FFloatPropertyParams NewProp_MaxMana;
		static const UE4CodeGen_Private::FFloatPropertyParams NewProp_Mana;
		static const UE4CodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UE4CodeGen_Private::FFunctionParams FuncParams;
	};
	const UE4CodeGen_Private::FFloatPropertyParams Z_Construct_UFunction_ACppCharacterBase_BP_OnManaChanged_Statics::NewProp_percentage = { "percentage", nullptr, (EPropertyFlags)0x0010000000000080, UE4CodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(CppCharacterBase_eventBP_OnManaChanged_Parms, percentage), METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FFloatPropertyParams Z_Construct_UFunction_ACppCharacterBase_BP_OnManaChanged_Statics::NewProp_MaxMana = { "MaxMana", nullptr, (EPropertyFlags)0x0010000000000080, UE4CodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(CppCharacterBase_eventBP_OnManaChanged_Parms, MaxMana), METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FFloatPropertyParams Z_Construct_UFunction_ACppCharacterBase_BP_OnManaChanged_Statics::NewProp_Mana = { "Mana", nullptr, (EPropertyFlags)0x0010000000000080, UE4CodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(CppCharacterBase_eventBP_OnManaChanged_Parms, Mana), METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_ACppCharacterBase_BP_OnManaChanged_Statics::PropPointers[] = {
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppCharacterBase_BP_OnManaChanged_Statics::NewProp_percentage,
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppCharacterBase_BP_OnManaChanged_Statics::NewProp_MaxMana,
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppCharacterBase_BP_OnManaChanged_Statics::NewProp_Mana,
	};
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_ACppCharacterBase_BP_OnManaChanged_Statics::Function_MetaDataParams[] = {
		{ "Category", "Abilities" },
		{ "DisplayName", "EvCppOnManaChanged" },
		{ "ModuleRelativePath", "Public/CppCharacterBase.h" },
	};
#endif
	const UE4CodeGen_Private::FFunctionParams Z_Construct_UFunction_ACppCharacterBase_BP_OnManaChanged_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_ACppCharacterBase, nullptr, "BP_OnManaChanged", nullptr, nullptr, sizeof(CppCharacterBase_eventBP_OnManaChanged_Parms), Z_Construct_UFunction_ACppCharacterBase_BP_OnManaChanged_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppCharacterBase_BP_OnManaChanged_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x08020800, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_ACppCharacterBase_BP_OnManaChanged_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppCharacterBase_BP_OnManaChanged_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_ACppCharacterBase_BP_OnManaChanged()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UE4CodeGen_Private::ConstructUFunction(ReturnFunction, Z_Construct_UFunction_ACppCharacterBase_BP_OnManaChanged_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_ACppCharacterBase_BP_OnStrengthChanged_Statics
	{
		static const UE4CodeGen_Private::FFloatPropertyParams NewProp_percentage;
		static const UE4CodeGen_Private::FFloatPropertyParams NewProp_MaxStrength;
		static const UE4CodeGen_Private::FFloatPropertyParams NewProp_Strength;
		static const UE4CodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UE4CodeGen_Private::FFunctionParams FuncParams;
	};
	const UE4CodeGen_Private::FFloatPropertyParams Z_Construct_UFunction_ACppCharacterBase_BP_OnStrengthChanged_Statics::NewProp_percentage = { "percentage", nullptr, (EPropertyFlags)0x0010000000000080, UE4CodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(CppCharacterBase_eventBP_OnStrengthChanged_Parms, percentage), METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FFloatPropertyParams Z_Construct_UFunction_ACppCharacterBase_BP_OnStrengthChanged_Statics::NewProp_MaxStrength = { "MaxStrength", nullptr, (EPropertyFlags)0x0010000000000080, UE4CodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(CppCharacterBase_eventBP_OnStrengthChanged_Parms, MaxStrength), METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FFloatPropertyParams Z_Construct_UFunction_ACppCharacterBase_BP_OnStrengthChanged_Statics::NewProp_Strength = { "Strength", nullptr, (EPropertyFlags)0x0010000000000080, UE4CodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(CppCharacterBase_eventBP_OnStrengthChanged_Parms, Strength), METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_ACppCharacterBase_BP_OnStrengthChanged_Statics::PropPointers[] = {
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppCharacterBase_BP_OnStrengthChanged_Statics::NewProp_percentage,
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppCharacterBase_BP_OnStrengthChanged_Statics::NewProp_MaxStrength,
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppCharacterBase_BP_OnStrengthChanged_Statics::NewProp_Strength,
	};
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_ACppCharacterBase_BP_OnStrengthChanged_Statics::Function_MetaDataParams[] = {
		{ "Category", "Abilities" },
		{ "DisplayName", "EvCppOnStrengthChanged" },
		{ "ModuleRelativePath", "Public/CppCharacterBase.h" },
	};
#endif
	const UE4CodeGen_Private::FFunctionParams Z_Construct_UFunction_ACppCharacterBase_BP_OnStrengthChanged_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_ACppCharacterBase, nullptr, "BP_OnStrengthChanged", nullptr, nullptr, sizeof(CppCharacterBase_eventBP_OnStrengthChanged_Parms), Z_Construct_UFunction_ACppCharacterBase_BP_OnStrengthChanged_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppCharacterBase_BP_OnStrengthChanged_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x08020800, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_ACppCharacterBase_BP_OnStrengthChanged_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppCharacterBase_BP_OnStrengthChanged_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_ACppCharacterBase_BP_OnStrengthChanged()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UE4CodeGen_Private::ConstructUFunction(ReturnFunction, Z_Construct_UFunction_ACppCharacterBase_BP_OnStrengthChanged_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbility_Statics
	{
		struct CppCharacterBase_eventEvCpp_AquireAbility_Parms
		{
			TSubclassOf<UGameplayAbility>  AbilityToAquire;
		};
		static const UE4CodeGen_Private::FClassPropertyParams NewProp_AbilityToAquire;
		static const UE4CodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UE4CodeGen_Private::FFunctionParams FuncParams;
	};
	const UE4CodeGen_Private::FClassPropertyParams Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbility_Statics::NewProp_AbilityToAquire = { "AbilityToAquire", nullptr, (EPropertyFlags)0x0014000000000080, UE4CodeGen_Private::EPropertyGenFlags::Class, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(CppCharacterBase_eventEvCpp_AquireAbility_Parms, AbilityToAquire), Z_Construct_UClass_UGameplayAbility_NoRegister, Z_Construct_UClass_UClass, METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbility_Statics::PropPointers[] = {
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbility_Statics::NewProp_AbilityToAquire,
	};
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbility_Statics::Function_MetaDataParams[] = {
		{ "Category", "Abilities" },
		{ "Comment", "/** START AquireAbility */" },
		{ "ModuleRelativePath", "Public/CppCharacterBase.h" },
		{ "ToolTip", "START AquireAbility" },
	};
#endif
	const UE4CodeGen_Private::FFunctionParams Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbility_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_ACppCharacterBase, nullptr, "EvCpp_AquireAbility", nullptr, nullptr, sizeof(CppCharacterBase_eventEvCpp_AquireAbility_Parms), Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbility_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbility_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x04020401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbility_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbility_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbility()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UE4CodeGen_Private::ConstructUFunction(ReturnFunction, Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbility_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbilitys_Statics
	{
		struct CppCharacterBase_eventEvCpp_AquireAbilitys_Parms
		{
			TArray<TSubclassOf<UGameplayAbility> > AbilityToAquires;
		};
		static const UE4CodeGen_Private::FArrayPropertyParams NewProp_AbilityToAquires;
		static const UE4CodeGen_Private::FClassPropertyParams NewProp_AbilityToAquires_Inner;
		static const UE4CodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UE4CodeGen_Private::FFunctionParams FuncParams;
	};
	const UE4CodeGen_Private::FArrayPropertyParams Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbilitys_Statics::NewProp_AbilityToAquires = { "AbilityToAquires", nullptr, (EPropertyFlags)0x0014000000000080, UE4CodeGen_Private::EPropertyGenFlags::Array, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(CppCharacterBase_eventEvCpp_AquireAbilitys_Parms, AbilityToAquires), METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FClassPropertyParams Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbilitys_Statics::NewProp_AbilityToAquires_Inner = { "AbilityToAquires", nullptr, (EPropertyFlags)0x0004000000000000, UE4CodeGen_Private::EPropertyGenFlags::Class, RF_Public|RF_Transient|RF_MarkAsNative, 1, 0, Z_Construct_UClass_UGameplayAbility_NoRegister, Z_Construct_UClass_UClass, METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbilitys_Statics::PropPointers[] = {
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbilitys_Statics::NewProp_AbilityToAquires,
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbilitys_Statics::NewProp_AbilityToAquires_Inner,
	};
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbilitys_Statics::Function_MetaDataParams[] = {
		{ "Category", "Abilities" },
		{ "ModuleRelativePath", "Public/CppCharacterBase.h" },
	};
#endif
	const UE4CodeGen_Private::FFunctionParams Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbilitys_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_ACppCharacterBase, nullptr, "EvCpp_AquireAbilitys", nullptr, nullptr, sizeof(CppCharacterBase_eventEvCpp_AquireAbilitys_Parms), Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbilitys_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbilitys_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x04020401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbilitys_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbilitys_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbilitys()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UE4CodeGen_Private::ConstructUFunction(ReturnFunction, Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbilitys_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_ACppCharacterBase_EvCpp_HitStun_Statics
	{
		struct CppCharacterBase_eventEvCpp_HitStun_Parms
		{
			float StunDuration;
		};
		static const UE4CodeGen_Private::FFloatPropertyParams NewProp_StunDuration;
		static const UE4CodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UE4CodeGen_Private::FFunctionParams FuncParams;
	};
	const UE4CodeGen_Private::FFloatPropertyParams Z_Construct_UFunction_ACppCharacterBase_EvCpp_HitStun_Statics::NewProp_StunDuration = { "StunDuration", nullptr, (EPropertyFlags)0x0010000000000080, UE4CodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(CppCharacterBase_eventEvCpp_HitStun_Parms, StunDuration), METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_ACppCharacterBase_EvCpp_HitStun_Statics::PropPointers[] = {
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppCharacterBase_EvCpp_HitStun_Statics::NewProp_StunDuration,
	};
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_ACppCharacterBase_EvCpp_HitStun_Statics::Function_MetaDataParams[] = {
		{ "Category", "Abilities" },
		{ "ModuleRelativePath", "Public/CppCharacterBase.h" },
	};
#endif
	const UE4CodeGen_Private::FFunctionParams Z_Construct_UFunction_ACppCharacterBase_EvCpp_HitStun_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_ACppCharacterBase, nullptr, "EvCpp_HitStun", nullptr, nullptr, sizeof(CppCharacterBase_eventEvCpp_HitStun_Parms), Z_Construct_UFunction_ACppCharacterBase_EvCpp_HitStun_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppCharacterBase_EvCpp_HitStun_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x04020401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_ACppCharacterBase_EvCpp_HitStun_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppCharacterBase_EvCpp_HitStun_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_ACppCharacterBase_EvCpp_HitStun()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UE4CodeGen_Private::ConstructUFunction(ReturnFunction, Z_Construct_UFunction_ACppCharacterBase_EvCpp_HitStun_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_ACppCharacterBase_FCppIsOtherHosttile_Statics
	{
		struct CppCharacterBase_eventFCppIsOtherHosttile_Parms
		{
			ACppCharacterBase* Other;
			bool ReturnValue;
		};
		static void NewProp_ReturnValue_SetBit(void* Obj);
		static const UE4CodeGen_Private::FBoolPropertyParams NewProp_ReturnValue;
		static const UE4CodeGen_Private::FObjectPropertyParams NewProp_Other;
		static const UE4CodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UE4CodeGen_Private::FFunctionParams FuncParams;
	};
	void Z_Construct_UFunction_ACppCharacterBase_FCppIsOtherHosttile_Statics::NewProp_ReturnValue_SetBit(void* Obj)
	{
		((CppCharacterBase_eventFCppIsOtherHosttile_Parms*)Obj)->ReturnValue = 1;
	}
	const UE4CodeGen_Private::FBoolPropertyParams Z_Construct_UFunction_ACppCharacterBase_FCppIsOtherHosttile_Statics::NewProp_ReturnValue = { "ReturnValue", nullptr, (EPropertyFlags)0x0010000000000580, UE4CodeGen_Private::EPropertyGenFlags::Bool | UE4CodeGen_Private::EPropertyGenFlags::NativeBool, RF_Public|RF_Transient|RF_MarkAsNative, 1, sizeof(bool), sizeof(CppCharacterBase_eventFCppIsOtherHosttile_Parms), &Z_Construct_UFunction_ACppCharacterBase_FCppIsOtherHosttile_Statics::NewProp_ReturnValue_SetBit, METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FObjectPropertyParams Z_Construct_UFunction_ACppCharacterBase_FCppIsOtherHosttile_Statics::NewProp_Other = { "Other", nullptr, (EPropertyFlags)0x0010000000000080, UE4CodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(CppCharacterBase_eventFCppIsOtherHosttile_Parms, Other), Z_Construct_UClass_ACppCharacterBase_NoRegister, METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_ACppCharacterBase_FCppIsOtherHosttile_Statics::PropPointers[] = {
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppCharacterBase_FCppIsOtherHosttile_Statics::NewProp_ReturnValue,
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppCharacterBase_FCppIsOtherHosttile_Statics::NewProp_Other,
	};
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_ACppCharacterBase_FCppIsOtherHosttile_Statics::Function_MetaDataParams[] = {
		{ "Category", "Abilities" },
		{ "ModuleRelativePath", "Public/CppCharacterBase.h" },
	};
#endif
	const UE4CodeGen_Private::FFunctionParams Z_Construct_UFunction_ACppCharacterBase_FCppIsOtherHosttile_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_ACppCharacterBase, nullptr, "FCppIsOtherHosttile", nullptr, nullptr, sizeof(CppCharacterBase_eventFCppIsOtherHosttile_Parms), Z_Construct_UFunction_ACppCharacterBase_FCppIsOtherHosttile_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppCharacterBase_FCppIsOtherHosttile_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x04020401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_ACppCharacterBase_FCppIsOtherHosttile_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppCharacterBase_FCppIsOtherHosttile_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_ACppCharacterBase_FCppIsOtherHosttile()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UE4CodeGen_Private::ConstructUFunction(ReturnFunction, Z_Construct_UFunction_ACppCharacterBase_FCppIsOtherHosttile_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_ACppCharacterBase_OnHealthChanged_Statics
	{
		struct CppCharacterBase_eventOnHealthChanged_Parms
		{
			float Health;
			float MaxHealth;
		};
		static const UE4CodeGen_Private::FFloatPropertyParams NewProp_MaxHealth;
		static const UE4CodeGen_Private::FFloatPropertyParams NewProp_Health;
		static const UE4CodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UE4CodeGen_Private::FFunctionParams FuncParams;
	};
	const UE4CodeGen_Private::FFloatPropertyParams Z_Construct_UFunction_ACppCharacterBase_OnHealthChanged_Statics::NewProp_MaxHealth = { "MaxHealth", nullptr, (EPropertyFlags)0x0010000000000080, UE4CodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(CppCharacterBase_eventOnHealthChanged_Parms, MaxHealth), METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FFloatPropertyParams Z_Construct_UFunction_ACppCharacterBase_OnHealthChanged_Statics::NewProp_Health = { "Health", nullptr, (EPropertyFlags)0x0010000000000080, UE4CodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(CppCharacterBase_eventOnHealthChanged_Parms, Health), METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_ACppCharacterBase_OnHealthChanged_Statics::PropPointers[] = {
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppCharacterBase_OnHealthChanged_Statics::NewProp_MaxHealth,
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppCharacterBase_OnHealthChanged_Statics::NewProp_Health,
	};
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_ACppCharacterBase_OnHealthChanged_Statics::Function_MetaDataParams[] = {
		{ "Comment", "/** START changes param */" },
		{ "ModuleRelativePath", "Public/CppCharacterBase.h" },
		{ "ToolTip", "START changes param" },
	};
#endif
	const UE4CodeGen_Private::FFunctionParams Z_Construct_UFunction_ACppCharacterBase_OnHealthChanged_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_ACppCharacterBase, nullptr, "OnHealthChanged", nullptr, nullptr, sizeof(CppCharacterBase_eventOnHealthChanged_Parms), Z_Construct_UFunction_ACppCharacterBase_OnHealthChanged_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppCharacterBase_OnHealthChanged_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x00020401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_ACppCharacterBase_OnHealthChanged_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppCharacterBase_OnHealthChanged_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_ACppCharacterBase_OnHealthChanged()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UE4CodeGen_Private::ConstructUFunction(ReturnFunction, Z_Construct_UFunction_ACppCharacterBase_OnHealthChanged_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_ACppCharacterBase_OnManaChanged_Statics
	{
		struct CppCharacterBase_eventOnManaChanged_Parms
		{
			float Mana;
			float MaxMana;
		};
		static const UE4CodeGen_Private::FFloatPropertyParams NewProp_MaxMana;
		static const UE4CodeGen_Private::FFloatPropertyParams NewProp_Mana;
		static const UE4CodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UE4CodeGen_Private::FFunctionParams FuncParams;
	};
	const UE4CodeGen_Private::FFloatPropertyParams Z_Construct_UFunction_ACppCharacterBase_OnManaChanged_Statics::NewProp_MaxMana = { "MaxMana", nullptr, (EPropertyFlags)0x0010000000000080, UE4CodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(CppCharacterBase_eventOnManaChanged_Parms, MaxMana), METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FFloatPropertyParams Z_Construct_UFunction_ACppCharacterBase_OnManaChanged_Statics::NewProp_Mana = { "Mana", nullptr, (EPropertyFlags)0x0010000000000080, UE4CodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(CppCharacterBase_eventOnManaChanged_Parms, Mana), METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_ACppCharacterBase_OnManaChanged_Statics::PropPointers[] = {
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppCharacterBase_OnManaChanged_Statics::NewProp_MaxMana,
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppCharacterBase_OnManaChanged_Statics::NewProp_Mana,
	};
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_ACppCharacterBase_OnManaChanged_Statics::Function_MetaDataParams[] = {
		{ "ModuleRelativePath", "Public/CppCharacterBase.h" },
	};
#endif
	const UE4CodeGen_Private::FFunctionParams Z_Construct_UFunction_ACppCharacterBase_OnManaChanged_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_ACppCharacterBase, nullptr, "OnManaChanged", nullptr, nullptr, sizeof(CppCharacterBase_eventOnManaChanged_Parms), Z_Construct_UFunction_ACppCharacterBase_OnManaChanged_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppCharacterBase_OnManaChanged_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x00020401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_ACppCharacterBase_OnManaChanged_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppCharacterBase_OnManaChanged_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_ACppCharacterBase_OnManaChanged()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UE4CodeGen_Private::ConstructUFunction(ReturnFunction, Z_Construct_UFunction_ACppCharacterBase_OnManaChanged_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_ACppCharacterBase_OnStrengthChanged_Statics
	{
		struct CppCharacterBase_eventOnStrengthChanged_Parms
		{
			float Strength;
			float MaxStrength;
		};
		static const UE4CodeGen_Private::FFloatPropertyParams NewProp_MaxStrength;
		static const UE4CodeGen_Private::FFloatPropertyParams NewProp_Strength;
		static const UE4CodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UE4CodeGen_Private::FFunctionParams FuncParams;
	};
	const UE4CodeGen_Private::FFloatPropertyParams Z_Construct_UFunction_ACppCharacterBase_OnStrengthChanged_Statics::NewProp_MaxStrength = { "MaxStrength", nullptr, (EPropertyFlags)0x0010000000000080, UE4CodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(CppCharacterBase_eventOnStrengthChanged_Parms, MaxStrength), METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FFloatPropertyParams Z_Construct_UFunction_ACppCharacterBase_OnStrengthChanged_Statics::NewProp_Strength = { "Strength", nullptr, (EPropertyFlags)0x0010000000000080, UE4CodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(CppCharacterBase_eventOnStrengthChanged_Parms, Strength), METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_ACppCharacterBase_OnStrengthChanged_Statics::PropPointers[] = {
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppCharacterBase_OnStrengthChanged_Statics::NewProp_MaxStrength,
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppCharacterBase_OnStrengthChanged_Statics::NewProp_Strength,
	};
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_ACppCharacterBase_OnStrengthChanged_Statics::Function_MetaDataParams[] = {
		{ "ModuleRelativePath", "Public/CppCharacterBase.h" },
	};
#endif
	const UE4CodeGen_Private::FFunctionParams Z_Construct_UFunction_ACppCharacterBase_OnStrengthChanged_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_ACppCharacterBase, nullptr, "OnStrengthChanged", nullptr, nullptr, sizeof(CppCharacterBase_eventOnStrengthChanged_Parms), Z_Construct_UFunction_ACppCharacterBase_OnStrengthChanged_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppCharacterBase_OnStrengthChanged_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x00020401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_ACppCharacterBase_OnStrengthChanged_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppCharacterBase_OnStrengthChanged_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_ACppCharacterBase_OnStrengthChanged()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UE4CodeGen_Private::ConstructUFunction(ReturnFunction, Z_Construct_UFunction_ACppCharacterBase_OnStrengthChanged_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_ACppCharacterBase_RemoveGameplayTag_Statics
	{
		struct CppCharacterBase_eventRemoveGameplayTag_Parms
		{
			FGameplayTag TagToRemove;
		};
		static const UE4CodeGen_Private::FStructPropertyParams NewProp_TagToRemove;
		static const UE4CodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UE4CodeGen_Private::FFunctionParams FuncParams;
	};
	const UE4CodeGen_Private::FStructPropertyParams Z_Construct_UFunction_ACppCharacterBase_RemoveGameplayTag_Statics::NewProp_TagToRemove = { "TagToRemove", nullptr, (EPropertyFlags)0x0010000000000180, UE4CodeGen_Private::EPropertyGenFlags::Struct, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(CppCharacterBase_eventRemoveGameplayTag_Parms, TagToRemove), Z_Construct_UScriptStruct_FGameplayTag, METADATA_PARAMS(nullptr, 0) };
	const UE4CodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_ACppCharacterBase_RemoveGameplayTag_Statics::PropPointers[] = {
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_ACppCharacterBase_RemoveGameplayTag_Statics::NewProp_TagToRemove,
	};
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_ACppCharacterBase_RemoveGameplayTag_Statics::Function_MetaDataParams[] = {
		{ "Category", "Abilities" },
		{ "ModuleRelativePath", "Public/CppCharacterBase.h" },
	};
#endif
	const UE4CodeGen_Private::FFunctionParams Z_Construct_UFunction_ACppCharacterBase_RemoveGameplayTag_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_ACppCharacterBase, nullptr, "RemoveGameplayTag", nullptr, nullptr, sizeof(CppCharacterBase_eventRemoveGameplayTag_Parms), Z_Construct_UFunction_ACppCharacterBase_RemoveGameplayTag_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppCharacterBase_RemoveGameplayTag_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x04420401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_ACppCharacterBase_RemoveGameplayTag_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_ACppCharacterBase_RemoveGameplayTag_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_ACppCharacterBase_RemoveGameplayTag()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UE4CodeGen_Private::ConstructUFunction(ReturnFunction, Z_Construct_UFunction_ACppCharacterBase_RemoveGameplayTag_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	UClass* Z_Construct_UClass_ACppCharacterBase_NoRegister()
	{
		return ACppCharacterBase::StaticClass();
	}
	struct Z_Construct_UClass_ACppCharacterBase_Statics
	{
		static UObject* (*const DependentSingletons[])();
		static const FClassFunctionLinkInfo FuncInfo[];
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam Class_MetaDataParams[];
#endif
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam NewProp_FullHealthTag_MetaData[];
#endif
		static const UE4CodeGen_Private::FStructPropertyParams NewProp_FullHealthTag;
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam NewProp_AttributeBase_MetaData[];
#endif
		static const UE4CodeGen_Private::FObjectPropertyParams NewProp_AttributeBase;
#if WITH_METADATA
		static const UE4CodeGen_Private::FMetaDataPairParam NewProp_AbilitySystemComp_MetaData[];
#endif
		static const UE4CodeGen_Private::FObjectPropertyParams NewProp_AbilitySystemComp;
		static const UE4CodeGen_Private::FPropertyParamsBase* const PropPointers[];
		static const UE4CodeGen_Private::FImplementedInterfaceParams InterfaceParams[];
		static const FCppClassTypeInfoStatic StaticCppClassTypeInfo;
		static const UE4CodeGen_Private::FClassParams ClassParams;
	};
	UObject* (*const Z_Construct_UClass_ACppCharacterBase_Statics::DependentSingletons[])() = {
		(UObject* (*)())Z_Construct_UClass_ACharacter,
		(UObject* (*)())Z_Construct_UPackage__Script_AbilitySystem2,
	};
	const FClassFunctionLinkInfo Z_Construct_UClass_ACppCharacterBase_Statics::FuncInfo[] = {
		{ &Z_Construct_UFunction_ACppCharacterBase_AddGameplayTag, "AddGameplayTag" }, // 1835106227
		{ &Z_Construct_UFunction_ACppCharacterBase_BP_Die, "BP_Die" }, // 2096730044
		{ &Z_Construct_UFunction_ACppCharacterBase_BP_OnHelathChanged, "BP_OnHelathChanged" }, // 1852279884
		{ &Z_Construct_UFunction_ACppCharacterBase_BP_OnManaChanged, "BP_OnManaChanged" }, // 1014602804
		{ &Z_Construct_UFunction_ACppCharacterBase_BP_OnStrengthChanged, "BP_OnStrengthChanged" }, // 306994254
		{ &Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbility, "EvCpp_AquireAbility" }, // 66800762
		{ &Z_Construct_UFunction_ACppCharacterBase_EvCpp_AquireAbilitys, "EvCpp_AquireAbilitys" }, // 3696456716
		{ &Z_Construct_UFunction_ACppCharacterBase_EvCpp_HitStun, "EvCpp_HitStun" }, // 2780034690
		{ &Z_Construct_UFunction_ACppCharacterBase_FCppIsOtherHosttile, "FCppIsOtherHosttile" }, // 3017320488
		{ &Z_Construct_UFunction_ACppCharacterBase_OnHealthChanged, "OnHealthChanged" }, // 2375483249
		{ &Z_Construct_UFunction_ACppCharacterBase_OnManaChanged, "OnManaChanged" }, // 2806967758
		{ &Z_Construct_UFunction_ACppCharacterBase_OnStrengthChanged, "OnStrengthChanged" }, // 2642238787
		{ &Z_Construct_UFunction_ACppCharacterBase_RemoveGameplayTag, "RemoveGameplayTag" }, // 2121648246
	};
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UClass_ACppCharacterBase_Statics::Class_MetaDataParams[] = {
		{ "HideCategories", "Navigation" },
		{ "IncludePath", "CppCharacterBase.h" },
		{ "ModuleRelativePath", "Public/CppCharacterBase.h" },
	};
#endif
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UClass_ACppCharacterBase_Statics::NewProp_FullHealthTag_MetaData[] = {
		{ "Category", "Abilities" },
		{ "ModuleRelativePath", "Public/CppCharacterBase.h" },
	};
#endif
	const UE4CodeGen_Private::FStructPropertyParams Z_Construct_UClass_ACppCharacterBase_Statics::NewProp_FullHealthTag = { "FullHealthTag", nullptr, (EPropertyFlags)0x0010000000000005, UE4CodeGen_Private::EPropertyGenFlags::Struct, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(ACppCharacterBase, FullHealthTag), Z_Construct_UScriptStruct_FGameplayTag, METADATA_PARAMS(Z_Construct_UClass_ACppCharacterBase_Statics::NewProp_FullHealthTag_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_ACppCharacterBase_Statics::NewProp_FullHealthTag_MetaData)) };
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UClass_ACppCharacterBase_Statics::NewProp_AttributeBase_MetaData[] = {
		{ "Category", "Abilities" },
		{ "EditInline", "true" },
		{ "ModuleRelativePath", "Public/CppCharacterBase.h" },
	};
#endif
	const UE4CodeGen_Private::FObjectPropertyParams Z_Construct_UClass_ACppCharacterBase_Statics::NewProp_AttributeBase = { "AttributeBase", nullptr, (EPropertyFlags)0x00100000000a000d, UE4CodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(ACppCharacterBase, AttributeBase), Z_Construct_UClass_UCppAttributeSetBase_NoRegister, METADATA_PARAMS(Z_Construct_UClass_ACppCharacterBase_Statics::NewProp_AttributeBase_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_ACppCharacterBase_Statics::NewProp_AttributeBase_MetaData)) };
#if WITH_METADATA
	const UE4CodeGen_Private::FMetaDataPairParam Z_Construct_UClass_ACppCharacterBase_Statics::NewProp_AbilitySystemComp_MetaData[] = {
		{ "Category", "Abilities" },
		{ "Comment", "/** Our ability system */" },
		{ "EditInline", "true" },
		{ "ModuleRelativePath", "Public/CppCharacterBase.h" },
		{ "ToolTip", "Our ability system" },
	};
#endif
	const UE4CodeGen_Private::FObjectPropertyParams Z_Construct_UClass_ACppCharacterBase_Statics::NewProp_AbilitySystemComp = { "AbilitySystemComp", nullptr, (EPropertyFlags)0x00100000000a000d, UE4CodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, STRUCT_OFFSET(ACppCharacterBase, AbilitySystemComp), Z_Construct_UClass_UAbilitySystemComponent_NoRegister, METADATA_PARAMS(Z_Construct_UClass_ACppCharacterBase_Statics::NewProp_AbilitySystemComp_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_ACppCharacterBase_Statics::NewProp_AbilitySystemComp_MetaData)) };
	const UE4CodeGen_Private::FPropertyParamsBase* const Z_Construct_UClass_ACppCharacterBase_Statics::PropPointers[] = {
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_ACppCharacterBase_Statics::NewProp_FullHealthTag,
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_ACppCharacterBase_Statics::NewProp_AttributeBase,
		(const UE4CodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_ACppCharacterBase_Statics::NewProp_AbilitySystemComp,
	};
		const UE4CodeGen_Private::FImplementedInterfaceParams Z_Construct_UClass_ACppCharacterBase_Statics::InterfaceParams[] = {
			{ Z_Construct_UClass_UAbilitySystemInterface_NoRegister, (int32)VTABLE_OFFSET(ACppCharacterBase, IAbilitySystemInterface), false },
		};
	const FCppClassTypeInfoStatic Z_Construct_UClass_ACppCharacterBase_Statics::StaticCppClassTypeInfo = {
		TCppClassTypeTraits<ACppCharacterBase>::IsAbstract,
	};
	const UE4CodeGen_Private::FClassParams Z_Construct_UClass_ACppCharacterBase_Statics::ClassParams = {
		&ACppCharacterBase::StaticClass,
		"Game",
		&StaticCppClassTypeInfo,
		DependentSingletons,
		FuncInfo,
		Z_Construct_UClass_ACppCharacterBase_Statics::PropPointers,
		InterfaceParams,
		UE_ARRAY_COUNT(DependentSingletons),
		UE_ARRAY_COUNT(FuncInfo),
		UE_ARRAY_COUNT(Z_Construct_UClass_ACppCharacterBase_Statics::PropPointers),
		UE_ARRAY_COUNT(InterfaceParams),
		0x009000A4u,
		METADATA_PARAMS(Z_Construct_UClass_ACppCharacterBase_Statics::Class_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UClass_ACppCharacterBase_Statics::Class_MetaDataParams))
	};
	UClass* Z_Construct_UClass_ACppCharacterBase()
	{
		static UClass* OuterClass = nullptr;
		if (!OuterClass)
		{
			UE4CodeGen_Private::ConstructUClass(OuterClass, Z_Construct_UClass_ACppCharacterBase_Statics::ClassParams);
		}
		return OuterClass;
	}
	IMPLEMENT_CLASS(ACppCharacterBase, 3587901164);
	template<> ABILITYSYSTEM2_API UClass* StaticClass<ACppCharacterBase>()
	{
		return ACppCharacterBase::StaticClass();
	}
	static FCompiledInDefer Z_CompiledInDefer_UClass_ACppCharacterBase(Z_Construct_UClass_ACppCharacterBase, &ACppCharacterBase::StaticClass, TEXT("/Script/AbilitySystem2"), TEXT("ACppCharacterBase"), false, nullptr, nullptr, nullptr);
	DEFINE_VTABLE_PTR_HELPER_CTOR(ACppCharacterBase);
PRAGMA_ENABLE_DEPRECATION_WARNINGS
#ifdef _MSC_VER
#pragma warning (pop)
#endif
