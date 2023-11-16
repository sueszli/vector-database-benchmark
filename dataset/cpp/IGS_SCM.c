/*
 * SCM_2.c
 *
 * Created: 2015-01-30 오후 12:15:06
 *  Author: Minyoung
 */ 


#define F_CPU							8000000

#include <avr/io.h>
#include <avr/interrupt.h>
//#include <avr/wdt.h>

#include <util/delay.h>

#include "IGS_SCM_CRC16.h"
#include "IGS_SCM_Define.h"
#include "IGS_SCM_Global.h"
#include "IGS_SCM_Function_Task.h"
#include "IGS_SCM_Function_485.h"
#include "IGS_SCM_Function_PLC.h"
#include "IGS_SCM_Function_ExtMem.h"

int main(void)
{

	uint8_t longCommandH, longCommandL;
	uint16_t ResLength;
	//SCMTF_SetClock();
	SCMTF_PortInit();
	SCM485F_ExtMemInit();
	
	
	G_idDipValue = SCMTF_ReadDipSwitchValueMYID();
	G_cntlmodeDipValue = SCMTF_ReadDipSwitchValueMode();
	G_testmodeDipValue = SCMTF_ReadDipSwitchValueTest();

	G_SULCR = 0;
	
	
	if(G_cntlmodeDipValue == OFF) // CCM MODE
	{
		SCM485F_USART0Init();
		SCM485F_RxCompleteTimerInit();
		SCM485F_BufferInit(&G_rxBuf485);
		SCM485F_BufferInit(&G_txBuf485);
		
		SCMPLCF_StatusUSMBuffInit();
		
		SCMPLCF_PLCTxTimerInit();
	}
	else // Only SCM Mode
	{

		G_totalUSMCnt = 100;
		G_SULCR = (UICF0 | LICF1);
		
		G_usmSensingF = ON;
		G_txCntlBufPLC.type = 0x5;

		
		SCMEMF_WriteExtMemUSMCntlID(G_totalUSMCnt);
		SCMPLCF_PLCTxTimerInit();
		
		TCCR2 = (0<<CS22)|(1<<CS21)|(1<<CS20);
	}

	//SCMPLCF_PLCTxTimerInit();
	

	sei();
	
	while(1)
	{
		if(LOW == SCMTF_CheckOverCurrent())
		{
			_delay_us(100);
			if(LOW == SCMTF_CheckOverCurrent())
			{
				PORTE = (1<<PE4);
				PORTB &= ~(1<<PB4);
				break;
			}

		}

		// 485 Command Task 
		longCommandH = G_RSCOMR>>8;
		longCommandL = G_RSCOMR & 0x00FF;
		if(longCommandH){
			if(longCommandH&RSCOMKH0){
				
				G_SULCR &= ~UICF0;
				SCMPLCF_StatusUSMBuffInit();
				G_totalUSMCnt = G_485Packet.data_length >> 2;
				G_StatusBufUSMtoSCM[0] = G_totalUSMCnt;
				SCMEMF_WriteExtMemUSMAddr(G_485Packet.data, G_totalUSMCnt);
				if(G_totalUSMCnt ==0)
					SCM485F_makeTxPacket(IDX_DT_INIT, 0, (uint8_t *)0);
				else{
					SCM485F_makeTxPacket(IDX_DT_ACK, 0, (uint8_t *)0);
				}
				G_plcUSMCurrentIndex = 0;
				G_usmSensingF = OFF;
				G_LedOffFlag = 0;
				G_RSCOMR = 0x0000;	
			}
			if(longCommandH&RSCOMKH1){
				G_SULCR &= ~LICF1;
				SCMPLCF_StatusLGMBuffInit();
				G_totalLGMCnt = G_485Packet.data_length >> 2;
				G_StatusBufLGMtoSCM[0] = G_totalLGMCnt;
				SCMEMF_WriteExtMemLGMAddr(G_485Packet.data, G_totalLGMCnt);
				if(G_totalUSMCnt ==0)
					SCM485F_makeTxPacket(IDX_DT_INIT, 0, (uint8_t *)0);
				else
					SCM485F_makeTxPacket(IDX_DT_ACK, 0, (uint8_t *)0);	
				G_plcLGMCurrentIndex = 0;
				G_RSCOMR = 0x0000;					
			}
			if(longCommandH&RSCOMKH2){
				SCMEMF_WriteExtMemUSMtoLGM(G_485Packet.data, G_totalUSMCnt);
				if(G_totalUSMCnt ==0)
					SCM485F_makeTxPacket(IDX_DT_INIT, 0, (uint8_t *)0);
				else
					SCM485F_makeTxPacket(IDX_DT_ACK, 0, (uint8_t *)0);	
				G_RSCOMR = 0x0000;						
			}
			if(longCommandH&RSCOMKH3){
				G_usmOPModeOnF = ON;
				SCMEMF_WriteExtMemUSMOPMode(G_485Packet.data, G_totalUSMCnt);
				SCMPLCF_StatusUSMBuffInit();
				if(G_totalUSMCnt ==0)
					SCM485F_makeTxPacket(IDX_DT_INIT, 0, (uint8_t *)0);
				else
					SCM485F_makeTxPacket(IDX_DT_ACK, 0, (uint8_t *)0);
				G_RSCOMR = 0x0000;	
			}
			if(longCommandH&RSCOMKH4){
				G_lgmOPModeOnF = ON;
				SCMEMF_WriteExtMemLGMOPMode(G_485Packet.data, G_totalUSMCnt);
				SCMPLCF_StatusLGMBuffInit();
				if(G_totalUSMCnt ==0)
					SCM485F_makeTxPacket(IDX_DT_INIT, 0, (uint8_t *)0);
				else
					SCM485F_makeTxPacket(IDX_DT_ACK, 0, (uint8_t *)0);
				G_RSCOMR = 0x0000;	
			}
			if(longCommandH&RSCOMKH5){
				EIMSK = (0<<INT6);
				G_usmSensingF = OFF;
				SCMEMF_WriteExtMemUSMParam(G_485Packet.data,G_totalUSMCnt);
				ParamCnt = 0;
				G_plcUSMCurrentIndex = 0;
				SCMPLCF_StatusUSMBuffInit();
				if(G_totalUSMCnt ==0)
					SCM485F_makeTxPacket(IDX_DT_INIT, 0, (uint8_t *)0);
				else
					SCM485F_makeTxPacket(IDX_DT_ACK, 0, (uint8_t *)0);
				G_RSCOMR = 0x0000;	
			}
			if(longCommandH&RSCOMKH6){
				G_usmOPModeOnF = OFF;
				SCMPLCF_StatusUSMBuffInit();
				if(G_totalUSMCnt ==0)
					SCM485F_makeTxPacket(IDX_DT_INIT, 0, (uint8_t *)0);
				else
					SCM485F_makeTxPacket(IDX_DT_ACK, 0, (uint8_t *)0);
				
				G_RSCOMR = 0x0000;
				
			}
			if(longCommandH&RSCOMKH7){
				G_lgmOPModeOnF = OFF;
				SCMPLCF_StatusLGMBuffInit();
				if(G_totalUSMCnt ==0)
					SCM485F_makeTxPacket(IDX_DT_INIT, 0, (uint8_t *)0);
				else
					SCM485F_makeTxPacket(IDX_DT_ACK, 0, (uint8_t *)0);
					
				G_RSCOMR = 0x0000;
				
			}			
		}
		else if(longCommandL){
			
			if(longCommandL&RSCOMKL0){
				ResLength = 1 + ((G_totalUSMCnt +1) >> 1);
				
				if(G_totalUSMCnt ==0){
					SCM485F_makeTxPacket(IDX_DT_INIT, 0, (uint8_t *)0);
				}
				else{
					SCM485F_makeTxPacket(IDX_DT_RES_USM_STAT, ResLength, G_StatusBufUSMtoSCM);
				}
				G_RSCOMR = 0x0000;
			}
			if(longCommandL&RSCOMKL1){
				ResLength = 1 + ((G_totalLGMCnt +1) >> 1);
				if(G_totalUSMCnt ==0){
					SCM485F_makeTxPacket(IDX_DT_INIT, 0, (uint8_t *)0);
				}					
				else
					SCM485F_makeTxPacket(IDX_DT_RES_LGM_STAT, ResLength, G_StatusBufLGMtoSCM);
				G_RSCOMR = 0x0000;
			}
			if(longCommandL&RSCOMKL2){
				G_usmSensingF = ON;
				if(G_totalUSMCnt ==0)
					SCM485F_makeTxPacket(IDX_DT_INIT, 0, (uint8_t *)0);
				else
					SCM485F_makeTxPacket(IDX_DT_ACK, 0, (uint8_t *)0);
				G_RSCOMR = 0x0000;	
			}
			if(longCommandL&RSCOMKL3){
				G_usmSensingF = OFF;
				if(G_totalUSMCnt ==0)
					SCM485F_makeTxPacket(IDX_DT_INIT, 0, (uint8_t *)0);
				else
					SCM485F_makeTxPacket(IDX_DT_ACK, 0, (uint8_t *)0);				
				G_RSCOMR = 0x0000;	
			}
			if(longCommandL&RSCOMKL4){
				
			}
			if(longCommandL&RSCOMKL5){
				
			}
			if(longCommandL&RSCOMKL6){
				
			}
			if(longCommandL&RSCOMKL7){
				if(G_totalUSMCnt ==0)
					SCM485F_makeTxPacket(IDX_DT_INIT, 0, (uint8_t *)0);
				else
					SCM485F_makeTxPacket(IDX_DT_NACK, 0, (uint8_t *)0);				
				G_RSCOMR = 0x0000;	
			}
		}
	}
}

ISR(USART0_RX_vect)
{
	u8data = UDR0;
	if(WRITE_FAIL == SCM485F_OneByteWrite_485(&G_rxBuf485, u8data))
	{
		// ERROR :: 485 Rx buffer overFlow
		SCM485F_BufferInit(&G_rxBuf485);
	}
	
	//485 Rx Data Complete Timer Setting
	TCNT0 = 0;
	TCCR0 = (1<<CS02)|(0<<CS01)|(0<<CS00);//TIMSK |= (1<<TOIE0);
	//G_485TimerOverFlowCnt =0;
	
	PORTB |= (1<<PB4);
	PORTB |= (1<<PB5);
	PORTB |= (1<<PB6);
}

ISR(TIMER0_OVF_vect)
{
	TCNT0 = 0;
	
	G_485TimerOverFlowCnt++;
	if(G_485TimerOverFlowCnt == 1 ) // 10ms 동안 RS485 Rx 가 없으면 Rx Complete
	{
		
		TCCR0 = (0<<CS02)|(0<<CS01)|(0<<CS00);//TIMSK &= ~(1<<TOIE0); //Timer OverFlow Interrupt Disable
		SCM485F_BufferInit(&G_rxBuf485);
		SCM485F_BufferInit(&G_txBuf485);
		
		G_485TimerOverFlowCnt = 0;
		switch(SCM485F_getRxPacket()){
			case RS485_RX_STX_ERROR :{
				PORTB &= ~(1<<PB6);
				break;
			}
			case RS485_RX_CRC_ERROR :{
				PORTB &= ~(1<<PB6);
				break;
			}
			case RS485_RX_INVALID_ID :{
				break;
			}
			case RS485_RX_OK :{
				switch(G_485Packet.data_type){
					case IDX_DT_SET_USM_ADDRESS:{
						G_RSCOMR |= (RSCOMKH0<<8);
						break;
					}
					case IDX_DT_SET_LGM_ADDRESS:{
						G_RSCOMR |= (RSCOMKH1<<8);
						break;
					}
					case IDX_DT_SET_USM_TO_LGM :{
						G_RSCOMR |= (RSCOMKH2<<8);
						break;
					}
					case IDX_DT_SET_USM_OP_MODE:{
						G_RSCOMR |= (RSCOMKH3<<8);
						break;
					}
					case IDX_DT_SET_LGM_OP_MODE:{
						G_RSCOMR |= (RSCOMKH4<<8);
						break;
					}
					case IDX_DT_SET_USM_PARAM:{
						G_RSCOMR |= (RSCOMKH5<<8);
						break;
					}
					case IDX_DT_SET_USM_OP_MODE_OFF :{
						G_RSCOMR |= (RSCOMKH6<<8);
						break;
					}
					case IDX_DT_SET_LGM_OP_MODE_OFF :{
						G_RSCOMR |= (RSCOMKH7<<8);
						break;
					}
					case IDX_DT_REQ_USM_STAT:{
						G_RSCOMR |= (RSCOMKL0);
						break;
					}
					case IDX_DT_REQ_LGM_STAT:{
						G_RSCOMR |= (RSCOMKL1);
						break;
					}
					case IDX_DT_SENSING_ON:{
						G_RSCOMR |= (RSCOMKL2);
						break;
					}
					case IDX_DT_SENSING_OFF:{
						G_RSCOMR |= (RSCOMKL3);
						break;
					}
					//case ReservedL1:{
						//G_RSCOMR |= (RSCOMKL4);
						//break;
					//}
					//case ReservedL2:{
						//G_RSCOMR |= (RSCOMKL5);
						//break;
					//}
					//case ReservedL3:{
						//G_RSCOMR |= (RSCOMKL6);
						//break;
					//}									
					default:{
						G_RSCOMR |= (RSCOMKL7);
						break;
					}
				}
				break;
			}
		}
		
	}
}

// RS485 : UDR0 Empty interrupt service routine
ISR(USART0_UDRE_vect)
{
	
	if(SCM485F_OneByteRead_485(&G_txBuf485, &UDR0) == 1)
	{
		UCSR0B &= ~(1<<UDRIE0);
		UCSR0B |= (1<<TXCIE0);
	}
}
//
ISR(USART0_TX_vect)
{
	UCSR0B &= ~(1<<TXCIE0);
	UCSR0B |= (1<<RXCIE0);
	PORTE |= (1<<PE2); // Tx : PE2 low / Rx : PE2 high
	TCCR2 = (0<<CS22)|(1<<CS21)|(1<<CS20);//TIMSK |= (1<<TOIE2);
	TCNT2 = 0;
	
}


//PLC Tx : 8bit Timer : 2ms / 1loop
ISR(TIMER2_OVF_vect)
{
	//uint32_t tmp = 0;
	TCNT2 = 0;

	G_plcTimerOverFlowCnt++;
	
	//[S] OverFlow Function
	if(G_plcTimerOverFlowCnt > 30) // 60ms 마다 한번씩 Tx
	{
		
		G_plcTimerOverFlowCnt=0;
		
		if(G_plcUSMCurrentIndex >= G_totalUSMCnt)
		{
			G_plcUSMCurrentIndex = 0;
		}
		if(G_plcLGMCurrentIndex >= G_totalLGMCnt)
		{
			G_plcLGMCurrentIndex = 0;
		}

		//[S] Start Address Setting
		initialUSMF = G_SULCR & (1<<SULCR0);
		initialLGMF = G_SULCR & (1<<SULCR1);
		paramCF = G_SULCR & (1<<SULCR4);

		if(UICF0 != initialUSMF || LICF1 != initialLGMF)
		{
			//[S] Start USM Address Setting
			uaddCF = G_SULCR & (1<<SULCR2);
			laddCF = G_SULCR & (1<<SULCR3);
			
			
			if(UACF2 == uaddCF)
			{
				if(G_LedOffFlag == 0){
					G_txAddBufPLC.type = IDX_PCLTYPE_ADDR_SET_START;
					G_txAddBufPLC.SerialNumber[0] = 0;
					G_txAddBufPLC.SerialNumber[1] = 0;
					G_txAddBufPLC.SerialNumber[2] = 0;
					G_txAddBufPLC.controlID =       0;
					SCMPLCF_SendMessageAddPacket(G_txAddBufPLC);
					G_LedOffFlag = 1;
					
					_delay_ms(10);
				}else{
					G_txAddBufPLC.type = IDX_PCLTYPE_CCM_USM_ADDR;
					G_txAddBufPLC.SerialNumber[0] = (G_usmAdd[G_plcUSMCurrentIndex] >> 16) & 0xFF;
					G_txAddBufPLC.SerialNumber[1] = (G_usmAdd[G_plcUSMCurrentIndex] >>  8) & 0xFF;
					G_txAddBufPLC.SerialNumber[2] = (G_usmAdd[G_plcUSMCurrentIndex]      ) & 0xFF;
					G_txAddBufPLC.controlID =       G_plcUSMCurrentIndex+1;
					PORTB &= ~(1<<PB5);
				
						
					if(G_plcUSMCurrentIndex == G_totalUSMCnt-1)
					{
						G_SULCR |= UICF0;
						G_SULCR &= ~UACF2;
						G_plcUSMCurrentIndex = 0;
						PORTB |= (1<<PB5);
						PORTB |= (1<<PB4);
					}
					else
					{
						G_plcUSMCurrentIndex++;
					}
				
					SCMPLCF_SendMessageAddPacket(G_txAddBufPLC);
				
					//_delay_ms(10);
				}
			}
			//[E] End USM Address Setting
			//[S] Start LGM Address SEtting
			else{
				if(LACF3 == laddCF){
					G_txAddBufPLC.type = IDX_PCLTYPE_CCM_LGM_ADDR;
					G_txAddBufPLC.SerialNumber[0] = (G_lgmAdd[G_plcLGMCurrentIndex] >> 16) & 0xFF;
					G_txAddBufPLC.SerialNumber[1] = (G_lgmAdd[G_plcLGMCurrentIndex] >>  8) & 0xFF;
					G_txAddBufPLC.SerialNumber[2] = (G_lgmAdd[G_plcLGMCurrentIndex]      ) & 0xFF;
					G_txAddBufPLC.controlID =       G_plcLGMCurrentIndex+1;
					PORTB &= ~(1<<PB5);

					if(G_plcLGMCurrentIndex == G_totalLGMCnt-1){
						G_SULCR |= LICF1;
						G_SULCR &= ~LACF3;
						G_plcLGMCurrentIndex = 0;
						PORTB |= (1<<PB5);
					}
					else{
						G_plcLGMCurrentIndex++;
					}
					
					SCMPLCF_SendMessageAddPacket(G_txAddBufPLC);
				}
			}
			//[E]End LGM Address SEtting
		}
		//[E] End Address Setting
		else if(UICF0 == initialUSMF && LICF1 == initialLGMF && paramCF == UPCF4){
			PORTB &= ~(1<<PB5);
			G_txCntlBufPLC.type = IDX_PCLTYPE_CCM_USM_CNTL;
			G_txCntlBufPLC.controlID = G_plcUSMCurrentIndex+1;

			if(ParamCnt == 0){
				G_txCntlBufPLC.OperationMode = IDX_OPM_PARAM_SETTING_PARAM1;
				G_txCntlBufPLC.data = G_usmParam[G_plcUSMCurrentIndex*4+0];
			}
			else if(ParamCnt == 1){
				G_txCntlBufPLC.OperationMode = IDX_OPM_PARAM_SETTING_PARAM2;
				G_txCntlBufPLC.data = G_usmParam[G_plcUSMCurrentIndex*4+1];;
			}
			else if(ParamCnt == 2){
				G_txCntlBufPLC.OperationMode = IDX_OPM_PARAM_SETTING_PARAM3;
				G_txCntlBufPLC.data = G_usmParam[G_plcUSMCurrentIndex*4+2];
			}
			else if(ParamCnt == 3){
				G_txCntlBufPLC.OperationMode = IDX_OPM_PARAM_SETTING_PARAM4;
				G_txCntlBufPLC.data = G_usmParam[G_plcUSMCurrentIndex*4+3];
			}
			
			if(ParamCnt == 3){
				ParamCnt = 0;
				G_plcUSMCurrentIndex+=1;
			}
			else{
				ParamCnt+=1;
			}

			if(G_plcUSMCurrentIndex == G_totalUSMCnt-1 ){
				PORTB |= (1<<PB5);
				PORTB |= (1<<PB4);
				G_SULCR &= ~UPCF4;
				ParamLoopCnt = 0;
				G_plcUSMCurrentIndex = 0;
				//G_usmSensingF = ON;
			}
			
			SCMPLCF_SendMessageCntlPacket(G_txCntlBufPLC);
			
			//_delay_ms(10);
		}
		//[S] Start Mode Control in Sensing mode
		else{
			if(G_usmSensingF == ON){
				uint8_t lop_mode, uop_mode;
				
				//[S] Start Update LGM
				if(G_cntlmodeDipValue == OFF){ // Only CCM Mode
					
					G_txCntlBufPLC.type = IDX_PCLTYPE_CCM_LGM_CNTL;
					G_txCntlBufPLC.controlID = G_plcLGMCurrentIndex+1;
					

					//lopmodeCF = G_SULCR & (1<<SULCR6);
					lop_mode = IDX_OPM_FORC_LED_ON_GREEN;
					//if(lopmodeCF == LOCF6){
					if(G_lgmOPModeOnF == ON){
						lop_mode = G_lgmOPMode[G_plcLGMCurrentIndex];
						if(lop_mode == 0){
							lop_mode = G_lgmStatusBuf[G_plcLGMCurrentIndex];
						}
					}
					else{
							lop_mode = G_lgmStatusBuf[G_txCntlBufPLC.controlID-1];
					}
					if(G_testmodeDipValue == ON){
						lop_mode = IDX_OPM_FORC_LED_BLINKING;
					}
					
					G_txCntlBufPLC.OperationMode = lop_mode;
					G_txCntlBufPLC.data = 0x00;
					
					
					
					if(G_plcLGMCurrentIndex == G_totalLGMCnt-1){
						G_plcLGMCurrentIndex = 0;
					}
					else{
						G_plcLGMCurrentIndex++;
					}
					SCMPLCF_SendMessageCntlPacket(G_txCntlBufPLC);
					_delay_ms(10);
				}
				//[E] End Update LGM

				
				//UMS Control
				G_txCntlBufPLC.type = (G_cntlmodeDipValue == OFF) ? IDX_PCLTYPE_CCM_USM_CNTL : IDX_PCLTYPE_SCM_USM_CNTL;
				G_txCntlBufPLC.controlID = G_plcUSMCurrentIndex+1;
				uop_mode = (G_cntlmodeDipValue == OFF) ? G_usmOPMode[G_plcUSMCurrentIndex] : 0;
				G_usmCurrentCntlID = G_txCntlBufPLC.controlID;
				
				if(G_cntlmodeDipValue != OFF) PORTB &= ~(1<<PB5);
						

				//uopmodeCF = G_SULCR & (1<<SULCR5);
				//if(uopmodeCF == UOCF5){
				if(G_usmOPModeOnF == ON){
					uop_mode = G_usmOPMode[G_plcUSMCurrentIndex];
				}
				else{
					uop_mode = 0;
				}
				if(G_testmodeDipValue == ON){
					uop_mode = IDX_OPM_FORC_LED_BLINKING;
				}

				switch(uop_mode)
				{
					case IDX_OPM_FORC_LED_ON_GREEN : {
						G_txCntlBufPLC.OperationMode = IDX_OPM_FORC_LED_ON_GREEN;
						G_txCntlBufPLC.data = 0x00;
						break;
					}
					case IDX_OPM_FORC_LED_ON_RED :{
						G_txCntlBufPLC.OperationMode = IDX_OPM_FORC_LED_ON_RED;
						G_txCntlBufPLC.data = 0x00;
						break;
					}
					case IDX_OPM_FORC_LED_BLINKING : {
						G_txCntlBufPLC.OperationMode = IDX_OPM_FORC_LED_BLINKING;
						G_txCntlBufPLC.data = 0x00;
						break;
					}
					case IDX_OPM_FORC_LED_OFF : {
						G_txCntlBufPLC.OperationMode = IDX_OPM_FORC_LED_OFF;
						G_txCntlBufPLC.data = 0x00;
						break;
					}
					case IDX_OPM_SENS_LED_ON_GREEN : {
						G_txCntlBufPLC.OperationMode = IDX_OPM_SENSING_ON;
						G_txCntlBufPLC.data = 0x00;
						break;
					}
					case IDX_OPM_SENS_LED_ON_RED : {
						G_txCntlBufPLC.OperationMode = IDX_OPM_SENSING_ON;
						G_txCntlBufPLC.data = 0x00;
						break;
					}
					default:{
						G_txCntlBufPLC.OperationMode = IDX_OPM_SENSING_ON;
						G_txCntlBufPLC.data = 0x00;
						break;
					}
				}

				if(G_plcUSMCurrentIndex == G_totalUSMCnt-1){
					G_plcUSMCurrentIndex = 0;
					
					//Make USM LGM Status Stream Buffer
					uint8_t usmIdx,lgmIdx;
					uint8_t lgmID, lgmStatusFirst, lgmStatusSecond,lgmStatus, usmStatus;
					uint8_t bufIdx;
					SCMPLCF_lgmStatusBuffInit();
					for(usmIdx=1; usmIdx<G_totalUSMCnt; usmIdx += 2 ){
						usmStatus = ((G_usmStatusBuffer[usmIdx-1]<<4)&0xF0) | (G_usmStatusBuffer[usmIdx]&0x0F);
						bufIdx = usmIdx>>1;	
						G_StatusBufUSMtoSCM[1+bufIdx] = usmStatus;
			
						lgmID = G_usmTolgm[usmIdx-1];
						if(G_usmStatusBuffer[usmIdx-1] != 1)
						{
							G_lgmStatusCnt[lgmID]+=1;
						}
						lgmID = G_usmTolgm[usmIdx];
						if(G_usmStatusBuffer[usmIdx] != 1)
						{
							G_lgmStatusCnt[lgmID]+=1;
						}
					}
		
					for(lgmIdx=1; lgmIdx<G_totalLGMCnt; lgmIdx += 2 ){
			
						if(G_lgmStatusCnt[lgmIdx] < G_lgmTotalCnt[lgmIdx])
							lgmStatusFirst = IDX_OPM_FORC_LED_ON_GREEN;
						else
							lgmStatusFirst = IDX_OPM_FORC_LED_ON_RED;
				
						if(G_lgmStatusCnt[lgmIdx+1] < G_lgmTotalCnt[lgmIdx+1])
							lgmStatusSecond = IDX_OPM_FORC_LED_ON_GREEN;
						else
							lgmStatusSecond = IDX_OPM_FORC_LED_ON_RED;
				
						if(G_usmPLCInterruptF[lgmIdx] == 0)
							lgmStatusFirst = 0;
						if(G_usmPLCInterruptF[lgmIdx+1] == 0)
							lgmStatusSecond = 0;
			
						//lopmodeCF = G_SULCR & (1<<SULCR6);
						//if(lopmodeCF == LOCF6){
						if(G_lgmOPModeOnF == ON){
							if(G_lgmOPMode[lgmIdx-1] != 0)
								lgmStatusFirst = G_lgmOPMode[lgmIdx-1];
							if(G_lgmOPMode[lgmIdx] != 0)
								lgmStatusSecond = G_lgmOPMode[lgmIdx];
						}
			
						lgmStatus = ((lgmStatusFirst<<4)&0xF0) | (lgmStatusSecond&0x0F);
						bufIdx = 1+(lgmIdx>>1);		
						G_StatusBufLGMtoSCM[bufIdx] = 	lgmStatus;	
						G_lgmStatusBuf[lgmIdx-1] = lgmStatusFirst ;
						G_lgmStatusBuf[lgmIdx] = lgmStatusSecond ;
					}
				}
				else{
					G_plcUSMCurrentIndex++;
				}
				////PLC TEST [S]
//				G_txCntlBufPLC.controlID = 0xAA;
//				G_txCntlBufPLC.data = 0xAA;
//				G_txCntlBufPLC.OperationMode = 0xAA;
//				G_txCntlBufPLC.type = 0xAA;
				////PLC TEST [E]
				
				SCMPLCF_SendMessageCntlPacket(G_txCntlBufPLC);
				EIMSK = (1<<INT6);
			}//[E] Start Mode Control in Sensing mode
		}
	}		//[E] OverFlow Function
}


//PLC RX Interrupt Service Routine
ISR(INT6_vect)
{
	uint8_t mask = 0x00;
	uint8_t bitvalue;
	uint8_t lgmID;
	
	G_rxBufPLC.sensor_status = 0xF8;
	PLC_RX_ONE_BIT(0,PLC_RX_DELAY_INIT,mask,bitvalue);

	PLC_RX_ONE_BIT(0,PLC_RX_DELAY,G_rxBufPLC.sensor_status,bitvalue);
	PLC_RX_ONE_BIT(1,PLC_RX_DELAY,G_rxBufPLC.sensor_status,bitvalue);
	PLC_RX_ONE_BIT(2,PLC_RX_DELAY,G_rxBufPLC.sensor_status,bitvalue);

	G_usmStatusBuffer[G_usmCurrentCntlID-1] = ~G_rxBufPLC.sensor_status;
	
	lgmID = G_usmTolgm[G_usmCurrentCntlID-1];
	G_usmPLCInterruptF[lgmID] = 1;
	EIMSK = (0<<INT6);
}