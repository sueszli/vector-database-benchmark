// Fill out your copyright notice in the Description page of Project Settings.


#include "PlayerExecutionCalculation.h"

UPlayerExecutionCalculation::UPlayerExecutionCalculation()
{
	// ���궨�巽������������ֵ���ж���
	DEFINE_ATTRIBUTE_CAPTUREDEF(UBaseAttributeSet, Attack, Target, true);
	DEFINE_ATTRIBUTE_CAPTUREDEF(UBaseAttributeSet, Armor, Target, true);
	DEFINE_ATTRIBUTE_CAPTUREDEF(UBaseAttributeSet, Health, Target, true);

	// ��ӽ���׽�б�
	RelevantAttributesToCapture.Add(AttackDef);
	RelevantAttributesToCapture.Add(ArmorDef);
	RelevantAttributesToCapture.Add(HealthDef);
}

void UPlayerExecutionCalculation::Execute_Implementation(const FGameplayEffectCustomExecutionParameters& ExcutionParams,
	FGameplayEffectCustomExecutionOutput& OutExecutionOutput) const
{
	float AttackMagnitude, ArmorMagnitude = 0.0f;

	// ���Լ��㲶������Դ�С
	ExcutionParams.AttemptCalculateCapturedAttributeMagnitude(AttackDef, FAggregatorEvaluateParameters(), AttackMagnitude);
	ExcutionParams.AttemptCalculateCapturedAttributeMagnitude(ArmorDef, FAggregatorEvaluateParameters(), ArmorMagnitude);

	// ���������˺�ֵ������ֵ�ķ�Χ��С��
	float FinalDamage = FMath::Clamp(AttackMagnitude - ArmorMagnitude, 0.0f, AttackMagnitude - ArmorMagnitude);

	OutExecutionOutput.AddOutputModifier(
		FGameplayModifierEvaluatedData(HealthProperty, EGameplayModOp::Additive, -FinalDamage));
}