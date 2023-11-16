/*
 * MIT License
 * Copyright (c) 2019 _VIFEXTech
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "exti.h"
#include "gpio.h"

/*外部中断回调函数指针数组*/
static EXTI_CallbackFunction_t EXTI_Function[16] = {0};

/**
  * @brief  外部中断初始化
  * @param  Pin: 引脚编号
  * @param  function: 回调函数
  * @param  Trigger_Mode: 触发方式
  * @param  PreemptionPriority: 抢占优先级
  * @param  SubPriority: 子优先级
  * @retval 无
  */
void EXTIx_Init(uint8_t Pin, EXTI_CallbackFunction_t function, EXTITrigger_TypeDef Trigger_Mode, uint8_t PreemptionPriority, uint8_t SubPriority)
{
    EXTI_InitTypeDef EXTI_InitStructure;
    NVIC_InitTypeDef NVIC_InitStructure;
    IRQn_Type EXTIx_IRQn;
    uint8_t Pinx;

    if(!IS_PIN(Pin))
        return;

    Pinx = Get_Pinx(Pin);

    if(Pinx > 15)
        return;
    
    EXTI_Function[Pinx] = function;

    //GPIO中断线以及中断初始化配置
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_SYSCFG, ENABLE);
    SYSCFG_EXTILineConfig(Get_EXTI_PortSourceGPIOx(Pin), Get_EXTI_PinSourcex(Pin));

    EXTI_InitStructure.EXTI_Line = 1 << Pinx;                       //设置中断线
    EXTI_InitStructure.EXTI_Mode = EXTI_Mode_Interrupt;             //设置触发模式，中断触发（事件触发）
    EXTI_InitStructure.EXTI_Trigger = Trigger_Mode;         //设置触发方式
    EXTI_InitStructure.EXTI_LineCmd = ENABLE;
    EXTI_Init(&EXTI_InitStructure);     //根据EXTI_InitStruct中指定的参数初始化外设EXTI寄存器

    if(Pinx <= 4)
    {
        switch(Pinx)
        {
        case 0:
            EXTIx_IRQn = EXTI0_IRQn;
            break;
        case 1:
            EXTIx_IRQn = EXTI1_IRQn;
            break;
        case 2:
            EXTIx_IRQn = EXTI2_TS_IRQn;
            break;
        case 3:
            EXTIx_IRQn = EXTI3_IRQn;
            break;
        case 4:
            EXTIx_IRQn = EXTI4_IRQn;
            break;
        }
    }
    else if(Pinx >= 5 && Pinx <= 9)
        EXTIx_IRQn = EXTI9_5_IRQn;
    else if(Pinx >= 10 && Pinx <= 15)
        EXTIx_IRQn = EXTI15_10_IRQn;

    NVIC_InitStructure.NVIC_IRQChannel = EXTIx_IRQn;                    //使能所在的外部中断通道
    NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = PreemptionPriority;      //抢占优先级
    NVIC_InitStructure.NVIC_IRQChannelSubPriority = SubPriority;                //子优先级
    NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;                     //使能外部中断通道
    NVIC_Init(&NVIC_InitStructure);
}


/**
  * @brief  外部中断初始化 (Arduino)
  * @param  Pin: 引脚编号
  * @param  function: 回调函数
  * @param  Trigger_Mode: 触发方式
  * @retval 无
  */
void attachInterrupt(uint8_t Pin, EXTI_CallbackFunction_t function, EXTITrigger_TypeDef Trigger_Mode)
{
    EXTIx_Init(Pin, function, Trigger_Mode, EXTI_PreemptionPriority_Default, EXTI_SubPriority_Default);
}

/**
  * @brief  关闭给定的中断 (Arduino)
  * @param  Pin: 引脚编号
  * @retval 无
  */
void detachInterrupt(uint8_t Pin)
{
    EXTI_InitTypeDef EXTI_InitStructure;
    uint8_t Pinx;

    if(!IS_PIN(Pin))
        return;

    Pinx = Get_Pinx(Pin);

    if(Pinx > 15)
        return;

    RCC_APB2PeriphClockCmd(RCC_APB2Periph_SYSCFG, ENABLE);
    SYSCFG_EXTILineConfig(Get_EXTI_PortSourceGPIOx(Pin), Get_EXTI_PinSourcex(Pin));//选择GPIO作为外部中断线路

    EXTI_InitStructure.EXTI_Line = 1 << Pinx;                       //设置中断线
    EXTI_InitStructure.EXTI_LineCmd = DISABLE;
    EXTI_Init(&EXTI_InitStructure);
}

/**
  * @brief  外部中断入口，通道0
  * @param  无
  * @retval 无
  */
void EXTI0_IRQHandler(void)
{
    if(EXTI_GetITStatus(EXTI_Line0) != RESET)
    {
        if(EXTI_Function[0]) EXTI_Function[0]();
        EXTI_ClearITPendingBit(EXTI_Line0);
    }
}

/**
  * @brief  外部中断入口，通道1
  * @param  无
  * @retval 无
  */
void EXTI1_IRQHandler(void)
{
    if(EXTI_GetITStatus(EXTI_Line1) != RESET)
    {
        if(EXTI_Function[1]) EXTI_Function[1]();
        EXTI_ClearITPendingBit(EXTI_Line1);
    }
}

/**
  * @brief  外部中断入口，通道2
  * @param  无
  * @retval 无
  */
void EXTI2_TS_IRQHandler(void)
{
    if(EXTI_GetITStatus(EXTI_Line2) != RESET)
    {
        if(EXTI_Function[2]) EXTI_Function[2]();
        EXTI_ClearITPendingBit(EXTI_Line2);
    }
}

/**
  * @brief  外部中断入口，通道3
  * @param  无
  * @retval 无
  */
void EXTI3_IRQHandler(void)
{
    if(EXTI_GetITStatus(EXTI_Line3) != RESET)
    {
        if(EXTI_Function[3]) EXTI_Function[3]();
        EXTI_ClearITPendingBit(EXTI_Line3);
    }
}

/**
  * @brief  外部中断入口，通道4
  * @param  无
  * @retval 无
  */
void EXTI4_IRQHandler(void)
{
    if(EXTI_GetITStatus(EXTI_Line4) != RESET)
    {
        if(EXTI_Function[4]) EXTI_Function[4]();
        EXTI_ClearITPendingBit(EXTI_Line4);
    }
}

/**
  * @brief  外部中断入口，通道9~5
  * @param  无
  * @retval 无
  */
void EXTI9_5_IRQHandler(void)
{
    if(EXTI_GetITStatus(EXTI_Line5) != RESET)
    {
        if(EXTI_Function[5]) EXTI_Function[5]();
        EXTI_ClearITPendingBit(EXTI_Line5);
    }
    if(EXTI_GetITStatus(EXTI_Line6) != RESET)
    {
        if(EXTI_Function[6]) EXTI_Function[6]();
        EXTI_ClearITPendingBit(EXTI_Line6);
    }
    if(EXTI_GetITStatus(EXTI_Line7) != RESET)
    {
        if(EXTI_Function[7]) EXTI_Function[7]();
        EXTI_ClearITPendingBit(EXTI_Line7);
    }
    if(EXTI_GetITStatus(EXTI_Line8) != RESET)
    {
        if(EXTI_Function[8]) EXTI_Function[8]();
        EXTI_ClearITPendingBit(EXTI_Line8);
    }
    if(EXTI_GetITStatus(EXTI_Line9) != RESET)
    {
        if(EXTI_Function[9]) EXTI_Function[9]();
        EXTI_ClearITPendingBit(EXTI_Line9);
    }
}

/**
  * @brief  外部中断入口，通道15~10
  * @param  无
  * @retval 无
  */
void EXTI15_10_IRQHandler(void)
{
    if(EXTI_GetITStatus(EXTI_Line10) != RESET)
    {
        if(EXTI_Function[10]) EXTI_Function[10]();
        EXTI_ClearITPendingBit(EXTI_Line10);
    }
    if(EXTI_GetITStatus(EXTI_Line11) != RESET)
    {
        if(EXTI_Function[11]) EXTI_Function[11]();
        EXTI_ClearITPendingBit(EXTI_Line11);
    }
    if(EXTI_GetITStatus(EXTI_Line12) != RESET)
    {
        if(EXTI_Function[12]) EXTI_Function[12]();
        EXTI_ClearITPendingBit(EXTI_Line12);
    }
    if(EXTI_GetITStatus(EXTI_Line13) != RESET)
    {
        if(EXTI_Function[13]) EXTI_Function[13]();
        EXTI_ClearITPendingBit(EXTI_Line13);
    }
    if(EXTI_GetITStatus(EXTI_Line14) != RESET)
    {
        if(EXTI_Function[14]) EXTI_Function[14]();
        EXTI_ClearITPendingBit(EXTI_Line14);
    }
    if(EXTI_GetITStatus(EXTI_Line15) != RESET)
    {
        if(EXTI_Function[15]) EXTI_Function[15]();
        EXTI_ClearITPendingBit(EXTI_Line15);
    }
}
