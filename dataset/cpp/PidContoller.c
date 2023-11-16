#include "stm32f10x.h"
#include "PidContoller.h"
//#include "UserConfiguration.h"

extern s32 CNT2,CNT3,CNT4,CNT5,V2,V3,V4,V5;

//����ʽPID�㷨
void PID_AbsoluteMode(PID_AbsoluteType* PID)
{
 if(PID->kp      < 0)    PID->kp      = -PID->kp;
 if(PID->ki      < 0)    PID->ki      = -PID->ki;
 if(PID->kd      < 0)    PID->kd      = -PID->kd;
 if(PID->errILim < 0)    PID->errILim = -PID->errILim;

 PID->errP = PID->errNow;  //��ȡ���ڵ�������kp����

 PID->errI += PID->errNow; //�����֣�����ki����

 if(PID->errILim != 0)	   //΢�����޺�����
 {
  if(     PID->errI >  PID->errILim)    PID->errI =  PID->errILim;
  else if(PID->errI < -PID->errILim)    PID->errI = -PID->errILim;
 }
 
 PID->errD = PID->errNow - PID->errOld;//���΢�֣�����kd����

 PID->errOld = PID->errNow;	//�������ڵ����
 
 PID->ctrOut = PID->kp * PID->errP + PID->ki * PID->errI + PID->kd * PID->errD;//�������ʽPID���

}


/*******************************************************************************************************/



//����ʽPID�㷨
void PID_IncrementMode(PID_IncrementType* PID)
{
 float dErrP, dErrI, dErrD;
 
 if(PID->kp < 0)    PID->kp = -PID->kp;
 if(PID->ki < 0)	PID->ki = -PID->ki;
 if(PID->kd < 0)    PID->kd = -PID->kd;

 dErrP = PID->errNow - PID->errOld1;

 dErrI = PID->errNow;

 dErrD = PID->errNow - 2 * PID->errOld1 + PID->errOld2;

 PID->errOld2 = PID->errOld1; //�������΢��
 PID->errOld1 = PID->errNow;  //һ�����΢��

 /*����ʽPID����*/
 PID->dCtrOut = PID->kp * dErrP + PID->ki * dErrI + PID->kd * dErrD;
 
 if(PID->kp == 0 && PID->ki == 0 && PID->kd == 0)   PID->ctrOut = 0;

 else PID->ctrOut += PID->dCtrOut;
}


/*****************************************����ٶȻ��ŷ�***********************************************/

s32 spdTag, spdNow, control;//����һ��Ŀ���ٶȣ������ٶȣ�������

PID_AbsoluteType PID_Control;//����PID�㷨�Ľṹ��

void User_PidSpeedControl(s32 SpeedTag)
{
   spdNow = V2; spdTag = SpeedTag;

   PID_Control.errNow = spdTag - spdNow; //���㲢д���ٶ����
   	
   PID_Control.kp      = 15;             //д�����ϵ��Ϊ15
   PID_Control.ki      = 5;              //д�����ϵ��Ϊ5
   PID_Control.kd      = 5;              //д��΢��ϵ��Ϊ5
   PID_Control.errILim = 1000;           //д������������Ϊ1000 ����Ϊ-1000

   PID_AbsoluteMode(&PID_Control);       //ִ�о���ʽPID�㷨
	
   control = PID_Control.ctrOut;         //��ȡ����ֵ

   //UserMotorSpeedSetOne(control);        //����PWM�����������ٶȵĿ�����

}










