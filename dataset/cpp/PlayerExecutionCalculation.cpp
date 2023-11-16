// Fill out your copyright notice in the Description page of Project Settings.


#include "PlayerExecutionCalculation.h"

UPlayerExecutionCalculation::UPlayerExecutionCalculation()
{
	// 将宏定义方法声明出来的值进行定义
	DEFINE_ATTRIBUTE_CAPTUREDEF(UBaseAttributeSet, Attack, Target, true);
	DEFINE_ATTRIBUTE_CAPTUREDEF(UBaseAttributeSet, Armor, Target, true);
	DEFINE_ATTRIBUTE_CAPTUREDEF(UBaseAttributeSet, Health, Target, true);

	// 添加进捕捉列表
	RelevantAttributesToCapture.Add(AttackDef);
	RelevantAttributesToCapture.Add(ArmorDef);
	RelevantAttributesToCapture.Add(HealthDef);
}

void UPlayerExecutionCalculation::Execute_Implementation(const FGameplayEffectCustomExecutionParameters& ExcutionParams,
	FGameplayEffectCustomExecutionOutput& OutExecutionOutput) const
{
	float AttackMagnitude, ArmorMagnitude = 0.0f;

	// 尝试计算捕获的属性大小
	ExcutionParams.AttemptCalculateCapturedAttributeMagnitude(AttackDef, FAggregatorEvaluateParameters(), AttackMagnitude);
	ExcutionParams.AttemptCalculateCapturedAttributeMagnitude(ArmorDef, FAggregatorEvaluateParameters(), ArmorMagnitude);

	// 计算最终伤害值（限制值的范围大小）
	float FinalDamage = FMath::Clamp(AttackMagnitude - ArmorMagnitude, 0.0f, AttackMagnitude - ArmorMagnitude);

	OutExecutionOutput.AddOutputModifier(
		FGameplayModifierEvaluatedData(HealthProperty, EGameplayModOp::Additive, -FinalDamage));
}