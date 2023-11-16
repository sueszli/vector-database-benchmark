/*
 * USM_3.c
 *
 * Created: 2014-11-12 오후 1:25:56
 *  Author: Minyoung
 */ 


 /* Ultrasound Speed in AIR : 343 m/s
    fs = 1/(50us) = 20kHz
	Del_z = c / (2*fs) = 34300 / (2*20*1000) = 0.86 Cm/Sample
	Sample Number = Depth / Del_z
	Depth = 300cm : 300Cm / 0.86 Cm/Sample = 348 Samples
	Receiving Time : 348 * 50us = Minimum 17.472 ms
	Dead Zone Time : 2ms --> 2ms / 50us =  40 Samples
 */
#define F_CPU 1000000
#include <avr/io.h>
//#include <avr/iomx8.h>
#include <util/delay.h>
#include <avr/interrupt.h>
#include <avr/eeprom.h>
#include <avr/sleep.h>
#include "IGS_LGM_Define.h"
#include "IGS_LGM_Global.h"
#include "IGS_LGM_Functions.h"

int main(void)
{
//	USMF_InitialRegisterSetting();
	USMF_PortInit();
	USMF_ExternalInterruptInit();
	USMF_ADCInit();
	USMF_TimerZeroInit();
	
	mySerialNumber = USMF_ReadSerialNumber();
	ReadDefaultParmaFromEEPROM();
	
	sei();
	USMF_StatusInit();
	
		//TIFR0 = (1<<TOV0);
		//TCCR0B = (0<<CS02) | (0<<CS01) | (1<<CS00);
		//TCNT0 = TIMER_INIT_VALUE;

	struct TypeOnePacket rx_message1;
	struct TxPacket tx_message;
	
    while(1)
    {

		
		if(G_isSameValue == 1){
			//PORTC ^= (1<<PC3);
			G_plcRxBuffer[(G_plcRxBitCnt>>3)] |= G_preBitValue << (G_plcRxBitCnt - ((G_plcRxBitCnt>>3) << 3));
			G_plcRxBitCnt++;
			G_isSameValue = 0;
			
			
		}
		if(G_plcRxBitCnt>=PLC_RX_MAXBIT){
			TCCR0B = (0<<CS02) | (0<<CS01) | (0<<CS00);
	
			uint8_t packetType =0;
			struct TypeZeroPacket rx_message0;
		
			packetType = (G_plcRxBuffer[0] & 0x0F)>>1;
		
			switch(packetType)
			{
				
				// CCM Mode
				// USM Control
				case IDX_PCLTYPE_ADDR_SET_START :{
					if(SN_ReceiveSuccessFallingF != 1){
						PORTC = LED_GREEN_OFF;
						PORTC = LED_RED_OFF;
					}
					break;
				}
				case IDX_PCLTYPE_CCM_LGM_ADDR : {
					
					// Serial Number - Control ID Setting
					rxSerialNumber = ((uint32_t)(((G_plcRxBuffer[0]&0xF0) >> 4) | ((G_plcRxBuffer[1]&0x0F) << 4)) <<16)
									+((uint32_t)(((G_plcRxBuffer[1]&0xF0) >> 4) | ((G_plcRxBuffer[2]&0x0F) << 4)) <<8) 
									+((( G_plcRxBuffer[2]&0xF0) >> 4) | ((G_plcRxBuffer[3]&0x0F) << 4));
					rx_message0.controlID		 = ((G_plcRxBuffer[3]&0xF0) >> 4) | ((G_plcRxBuffer[4]&0x0F) << 4);

					if(rxSerialNumber == mySerialNumber)
					{
						PORTC = LED_GREEN_ON;
						EPR_ID = eeprom_read_byte((uint8_t *)7);
							
						if(EPR_ID != rx_message0.controlID)
						{
							
							MY_ID = rx_message0.controlID;
							eeprom_write_byte((uint8_t *)7, rx_message0.controlID);
						}
						else
						{
							MY_ID = EPR_ID;
						}
							
						SN_ReceiveSuccessFallingF = 1;
						//USMF_SetControlIDCompleteResponse();
					}
					break;
				}
				case IDX_PCLTYPE_CCM_LGM_CNTL : {
					// USM Control
					
					if( SN_ReceiveSuccessFallingF == 1)
					{
								
						rx_message1.controlID = ((G_plcRxBuffer[0]&0xF0) >> 4) | ((G_plcRxBuffer[1] & 0x0F) << 4);
						rx_message1.OperationMode =  (G_plcRxBuffer[1]&0xF0) >> 4;
						rx_message1.data = G_plcRxBuffer[2];
							
						tx_message.sensor_status	= 0;
						if(rx_message1.controlID == MY_ID)
						{
							switch(rx_message1.OperationMode)
							{
								
								case IDX_OPM_FORC_LED_ON_GREEN : {
									PORTC = LED_GREEN_ON;
									break;
								}
								case IDX_OPM_FORC_LED_ON_RED : {
									PORTC = LED_RED_ON;
									break;
								}
								case IDX_OPM_FORC_LED_BLINKING :{
									if(LEDStatus==0)
									{
										PORTC = LED_GREEN_ON;
										LEDStatus = 1;
									}
									else if(LEDStatus==1)
									{
										PORTC = LED_RED_ON;
										LEDStatus = 0;
									}
									break;
								}
								case IDX_OPM_FORC_LED_OFF : {
									PORTC = LED_GREEN_OFF;
									break;
								}
								default: break;

							}
						}
					}
					//else
					//{
						//_delay_ms(12);
					//}
					break;
				}
					
				//SCM Mode
				//case IDX_PCLTYPE_SCM_USM_CNTL : {
					////Sensor Control
					//rx_message1.controlID = ((G_plcRxBuffer[0]&0xF0) >> 4) | ((G_plcRxBuffer[1] & 0x0F) << 4);
					//rx_message1.OperationMode =  (G_plcRxBuffer[1]&0xF0) >> 4;
					//rx_message1.data = G_plcRxBuffer[2];
						//
					//tx_message.sensor_status	= 0;
//
					//if(rx_message1.controlID == EPR_ID)
					//{
						////PORTC = LED_GREEN_ON;
						//USMF_SensorOPMControl(rx_message1, &tx_message);
					//}
					//break;
				//}
			}
			USMF_StatusInit();
		}
    }
}

// 동일한 값이 오면 인터럽트 루틴을 타지 못하기 때문에 타이머로 체크
// SCM 에서 1bit time length 만큼의 시간으로 overflow interrupt를 발생시켜 주어야 한다. 
ISR(TIMER0_OVF_vect)
{
	//PORTC ^= (1<<PC2);
	G_isSameValue = 1;
	TCNT0 = TIMER_INIT_VALUE;
}

//Start : Falling Edge
ISR (INT0_vect)
{
	TCCR0B = (0<<CS02) | (0<<CS01) | (0<<CS00);
	TCNT0 = TIMER_INIT_VALUE;
	TIFR0 &= ~(1<<TOV0);
	//PORTC ^= (1<<PC2);
	_delay_us(30);
	if(!(MYPPNSR & (1<<PPS)))//if(G_RisingF == 0) 
	{
		MYPPNSR |= (1<<PNS);//G_FallingF = 1;
		G_plcRxBuffer[(G_plcRxBitCnt>>3)] |= 1 <<(G_plcRxBitCnt - ((G_plcRxBitCnt>>3) << 3)) ;
		G_preBitValue = 1;
		G_plcRxBitCnt++;
	}
	else
	{
		MYPPNSR &= ~(1<<PNS);//G_FallingF = 0;
		G_plcRxBuffer[(G_plcRxBitCnt>>3)] |= 0 << (G_plcRxBitCnt - ((G_plcRxBitCnt>>3) << 3));
		G_preBitValue = 0;
		G_plcRxBitCnt++;
	}
	
	//동일한 값이 오면 인터럽트 루틴을 타지 못하기 때문에 타이머로 체크
	
	TCCR0B = (0<<CS02) | (0<<CS01) | (1<<CS00);
	TCNT0 = TIMER_INIT_VALUE;
	TIFR0 = (1<<TOV0);
	

	EIFR  = (1<<INTF0) |(1<<INTF1);
}

//Start : Rising Edge
ISR (INT1_vect)
{
	TCCR0B = (0<<CS02) | (0<<CS01) | (0<<CS00);
	TCNT0 = TIMER_INIT_VALUE;
	TIFR0 &= ~(1<<TOV0);
	//PORTC ^= (1<<PC2);
	_delay_us(30);
	if(!(MYPPNSR & (1<<PNS)))//if(G_FallingF == 0)
	{
		MYPPNSR |= (1<<PPS);//G_RisingF = 1;
		G_plcRxBuffer[(G_plcRxBitCnt>>3)] |= 1 << (G_plcRxBitCnt - ((G_plcRxBitCnt>>3) << 3));
		G_preBitValue = 1;
		G_plcRxBitCnt++;
	}
	else
	{
		MYPPNSR &= ~(1<<PPS);//G_RisingF = 0;
		G_plcRxBuffer[(G_plcRxBitCnt>>3)] |= 0 << (G_plcRxBitCnt - ((G_plcRxBitCnt>>3) << 3));
		G_preBitValue = 0;
		G_plcRxBitCnt++;
	}
	
	TCCR0B = (0<<CS02) | (0<<CS01) | (1<<CS00);
	TCNT0 = TIMER_INIT_VALUE;
	TIFR0 = (1<<TOV0);
	

	EIFR  = (1<<INTF0) |(1<<INTF1);	
}
