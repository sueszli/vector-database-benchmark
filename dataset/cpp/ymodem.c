/*******************************************************************************
** �ļ���: 		ymodem.c
** �汾��  		1.0
** ��������: 	RealView MDK-ARM 4.14
** ����: 		wuguoyana
** ��������: 	2011-04-29
** ����:		��Ymodem.c����ص�Э���ļ�
                ����ӳ����ն˽�������(ʹ��YmodemЭ��)���������ݼ��ص��ڲ�RAM�С�
                ����������������������ݱ�̵�Flash�У����������������ʾ����
** ����ļ�:	stm32f10x.h
** �޸���־��	2011-04-29   �����ĵ�
*******************************************************************************/

/* ����ͷ�ļ� *****************************************************************/

#include "common.h"
#include "stm32f10x_flash.h"

/* �������� -----------------------------------------------------------------*/
uint8_t file_name[FILE_NAME_LENGTH];//���յĶ������ļ��ļ����ļ���
//�û�����Flashƫ��
uint32_t FlashDestination = ApplicationAddress;//�û�������ʼ���ش洢��ַ
uint16_t PageSize = PAGE_SIZE;//����С
uint32_t EraseCounter = 0x0;
uint32_t NbrOfPage = 0;       //flash�ǰ�ҳ���洢�� ��ʼ��ַ���ļ���С��Ҫ�����flashҳ��
FLASH_Status FLASHStatus = FLASH_COMPLETE;
uint32_t RamSource;
extern uint8_t tab_1024[1024];

/*******************************************************************************
  * @��������	Receive_Byte
  * @����˵��   �ӷ��Ͷ˽���һ���ֽ�
  * @�������   c: �����ַ�
                timeout: ��ʱʱ��
  * @�������   ��
  * @���ز���   ���յĽ��
                0���ɹ�����
                1��ʱ�䳬ʱ
*******************************************************************************/
static  int32_t Receive_Byte (uint8_t *c, uint32_t timeout)
{
    while (timeout-- > 0)
    {
        if (SerialKeyPressed(c) == 1)//���ڽ���һ���ֽ�����
        {
            return 0;
        }
    }
    return -1;
}

/*******************************************************************************
  * @��������	Send_Byte
  * @����˵��   ����һ���ַ�
  * @�������   c: ���͵��ַ�
  * @�������   ��
  * @���ز���   ���͵Ľ��
                0���ɹ�����
*******************************************************************************/
static uint32_t Send_Byte (uint8_t c)
{
    SerialPutChar(c);// ���ڷ���һ���ֽ�
    return 0;
}

/*******************************************************************************
  * @��������	Receive_Packet
  * @����˵��   �ӷ��Ͷ˽���һ�����ݰ�
  * @�������   data ���洢����ָ��           ���ֽڶ������ݰ����� + ���ݰ�ͷ��Ϣ(5�ֽڣ�) + 1024/128�ֽ�����
                length���������ݳ��� 128/1024
                timeout ����ʱʱ��
  * @�������   �������ݰ��Ĵ�С 
  * @���ز���   ���յĽ��
                0: ��������
                -1: ��ʱ�������ݰ�����
                1: �û�ȡ��
*******************************************************************************/
static int32_t Receive_Packet (uint8_t *data, int32_t *length, uint32_t timeout)
{
    uint16_t i, packet_size;
    uint8_t c; //���յ�һ���ֽ�����
    *length = 0;

///////�жϵ�һ�ֽ�����////////////	
	//ÿ�����ݰ��� ��һ���ֽ����� Ϊ��־ ����c��
    if (Receive_Byte(&c, timeout) != 0)// ���յ�һ���ֽ����� �û������С ״����  ���ڽ���һ������  RS485   CAN ��Ҳ����
    {
        return -1;
    }
    switch (c)
    {
    case SOH:  // #define SOH   (0x01)  //128�ֽ����ݰ���ʼ
        packet_size = PACKET_SIZE;// ���ݴ����  �������ݰ��Ĵ�С 128
        break;
    case STX:  // #define STX   (0x02)  //1024�ֽڵ����ݰ���ʼ
        packet_size = PACKET_1K_SIZE;// ���ݰ���СΪ  1024
        break;
    case EOT:  // #define EOT                     (0x04)  //��������  end of transmit
        return 0;
    case CA:   // #define CA                      (0x18)  //�����������ֹת�� cancel
        if ((Receive_Byte(&c, timeout) == 0) && (c == CA))
        {
            *length = -1;   //���Ͷ���ֹ    ����  ��������  CA=(0x18)  ��ʾ����ȡ������
            return 0;     
        }
        else
        {
            return -1; 
        }
    case ABORT1:  // #define ABORT1   0x41)  //'A' == 0x41, �û���ֹ 
    case ABORT2:  // #define ABORT2   (0x61)  //'a' == 0x61, �û���ֹ
        return 1;
    default:
        return -1;
    }
		
//////���պ��������/////////////////
    *data = c;// ��ŵ�һ���ֽ�����
		// �������ݰ���һ���ֽڶ���� ���ݰ���С��������  ע�⻹��һ�����ݰ�ͷ��Ϣ
    for (i = 1; i < (packet_size + PACKET_OVERHEAD); i++)
    {
        if (Receive_Byte(data + i, timeout) != 0)//���պ��������  ���뵽 data��
        {
            return -1;
        }
    }
    if (data[PACKET_SEQNO_INDEX] != ((data[PACKET_SEQNO_COMP_INDEX] ^ 0xff) & 0xff))
    {
        return -1;
    }
    *length = packet_size;//���ؽ��ܵ����ݰ�����  128���� 1024
    return 0;
}

/*******************************************************************************
  * @��������	Ymodem_Receive
  * @����˵��   ͨ�� ymodemЭ�����һ���ļ�
  * @�������   �ļ��洢���� buf: �׵�ַָ��
  * @�������   ��
  * @���ز���   �ļ�����

���ͷ�:
�ȴ����յ� C  �ȴ������ļ���־   
��0�����ݰ�Ϊ 3�ֽ� ���ݰ���С��־ + ���00+��ŷ���FF  + 128�ֽ�/1024�ֽ� �ļ��������ݰ�(�ļ�����  +  �ļ���С ���㲹��) + ���ֽ� CRCУ����
              �ȴ���ӦACK
���Ϊ���ݰ�  3�ֽ� ���ݰ���С��־ + ���01+��ŷ���FE  + 128�ֽ�/1024�ֽ� ����(���㲹�� ���з�) + ���ֽ� CRCУ����
              �ȴ���ӦACK
����� EOT�������ͱ�־
              �ȴ���ӦACK
�ٷ���һ��ȫΪ������ݰ� 3�ֽ� ���ݰ���С��־ + ���00+��ŷ���FF + 128�ֽ� 0����  + CRCУ����
              �ȴ���ӦACK
�����һ�η��� EOT�������ͱ�־����������
              �ȴ���ӦACK

���շ���
����'C',�ȴ����ͷ��ͷ���Ӧ
�������ݰ�   ��һ����Ϊ Ϊ���ݰ���С��־  ���ݱ�־�����ַ�����Ӧ


*******************************************************************************/
int32_t Ymodem_Receive (uint8_t *buf)
{
	  // ����һ�����ݰ���С  �ļ���С
    uint8_t packet_data[PACKET_1K_SIZE + PACKET_OVERHEAD], file_size[FILE_SIZE_LENGTH], *file_ptr, *buf_ptr;
	// ���ݰ�����
    int32_t i, j, packet_length, session_done, file_done, packets_received, errors, session_begin, size = 0;

    //��ʼ��Flash��ַ����
    FlashDestination = ApplicationAddress;//�û��������صĳ�ʼ��ַ

    for (session_done = 0, errors = 0, session_begin = 0; ;)
    {
        for (packets_received = 0, file_done = 0, buf_ptr = buf; ;)
        {
					  // ����һ�����ݰ� ���ݴ���� packet_data��  ���ݰ���Ч���ݴ�С 128/1024 �����packet_length��
            switch (Receive_Packet(packet_data, &packet_length, NAK_TIMEOUT))//����һ�����ݰ� ���ݰ��׵�ַ ��ʱ
            {
            case 0://������������
                errors = 0;
                switch (packet_length)//���ݰ�����
                {
                    //���Ͷ���ֹ
                case - 1:
                    Send_Byte(ACK); // #define ACK                     (0x06)  //��Ӧ��λ��
                    return 0;
                    //��������
                case 0:
                    Send_Byte(ACK); // #define ACK                     (0x06)  //��Ӧ��λ��
                    file_done = 1;  // ���ݽ������
                    break;
                    //���������ݰ�
                default:
                    if ((packet_data[PACKET_SEQNO_INDEX] & 0xff) != (packets_received & 0xff))
                    {
                        Send_Byte(NAK); // #define NAK    (0x15)  //û��Ӧ
                    }
                    else
                    {
///////////////////////��0�����ݰ�   Ϊ�ļ���+�ļ���С���ݰ�/////////////////////////
                        if (packets_received == 0)
                        {
                            // �ļ������ݰ�
													  // ����FLASH�ռ�  packet_data[PACKET_HEADER]�������ݿ�ʼ
                            if (packet_data[PACKET_HEADER] != 0)
                            {
                                //�ļ������ݰ���Ч��������  ��PACKET_HEADER�±꿪ʼΪ����
															  //ע�ⷢ���ļ����� �����Ż���һ��0 �Ի��� �ļ�������� �ļ���С����
                                for (i = 0, file_ptr = packet_data + PACKET_HEADER; (*file_ptr != 0) && (i < FILE_NAME_LENGTH);)
                                {
                                    file_name[i++] = *file_ptr++; // �ļ������� �� *file_ptr = 0 �ĵط�
                                }
                                file_name[i++] = '\0';//��ӽ�����
                                for (i = 0, file_ptr ++; (*file_ptr != ' ') && (i < FILE_SIZE_LENGTH);)
                                {
                                    file_size[i++] = *file_ptr++;// �ļ���С���� �� file_ptr = ' '�ĵط�
                                }
                                file_size[i++] = '\0';//��ӽ�����
                                Str2Int(file_size, &size);// �ַ�����Сת����   ���ִ�С size

                                //�������ݰ��Ƿ����
                                if (size > (FLASH_SIZE - ApplicationAddress_offset - 1)) 
																	// �е�����FLASH_SIZE = (0x20000)  /* 128 KBytes */ Ϊȫ����С Ӧ��Ҫ��ȥ IAP�����С ApplicationAddress_offset
                                {
                                    //����
                                    Send_Byte(CA);
                                    Send_Byte(CA);
                                    return -1; //�ļ�����FLASH �û������С
                                }

                                //������Ҫ����Flash��ҳ
                                NbrOfPage = FLASH_PagesMask(size);
      /////////// ֱ�� ����Flash�ǲ���������  Ӧ�����ж�һ�� ���ܵ��ļ��Ƿ���Ч �ٲ�����������
								//�ж� crcУ�����Ƿ���ȷ
                 ////////////////////////����Flash//////////////////////////
                                for (EraseCounter = 0; (EraseCounter < NbrOfPage) && (FLASHStatus == FLASH_COMPLETE); EraseCounter++)
                                {
                                    FLASHStatus = FLASH_ErasePage(FlashDestination + (PageSize * EraseCounter));
                                }
                                Send_Byte(ACK);//��Ӧ
                                Send_Byte(CRC16);//�ٷ��� 'C'
                            }
                            //�ļ������ݰ��գ���������
                            else  // ��������Ϊ 0 �Ѿ��������
                            {
                                Send_Byte(ACK);
                                file_done = 1;
                                session_done = 1;
                                break;
                            }
                        }
												
//////////////////////����ʵ���ļ����� ��
                        else
                        {
                            memcpy(buf_ptr, packet_data + PACKET_HEADER, packet_length);//ֱ�ӿ��� ���ݰ��� ��Ч����
                            RamSource = (uint32_t)buf;
                            for (j = 0; (j < packet_length) && (FlashDestination <  ApplicationAddress + size); j += 4)
                            {
                                //�ѽ��յ������ݱ�д��Flash��
                                FLASH_ProgramWord(FlashDestination, *(uint32_t*)RamSource);

                                if (*(uint32_t*)FlashDestination != *(uint32_t*)RamSource)
                                {
                                    //����
                                    Send_Byte(CA); // ����
                                    Send_Byte(CA);
                                    return -2;
                                }
                                FlashDestination += 4;// flash��ַ��4�ֽ����� ��Ϊż��
                                RamSource += 4;       // ��Ҫд��Ķ������ļ���ַҲͬ����
                            }
                            Send_Byte(ACK);//��Ӧ
                        }
                        packets_received ++;//���ݰ����+1
                        session_begin = 1;
                    }
                }
                break;
            case 1://����������ֹ
                Send_Byte(CA);
                Send_Byte(CA);
                return -3;
            default:
                if (session_begin > 0)
                {
                    errors ++;
                }
                if (errors > MAX_ERRORS)
                {
                    Send_Byte(CA);
                    Send_Byte(CA);
                    return 0;
                }
                Send_Byte(CRC16);
                break;
            }
            if (file_done != 0)
            {
                break;
            }
        }
        if (session_done != 0)
        {
            break;
        }
    }
    return (int32_t)size;
}

/*******************************************************************************
  * @��������	Ymodem_CheckResponse
  * @����˵��   ͨ�� ymodemЭ������Ӧ
  * @�������   c
  * @�������   ��
  * @���ز���   0
*******************************************************************************/
int32_t Ymodem_CheckResponse(uint8_t c)
{
    return 0;
}

/*******************************************************************************
  * @��������	Ymodem_PrepareIntialPacket
  * @����˵��   ׼����һ�����ݰ�     ��һ�����ݰ� ����������Ϊ   �ļ��� ���ļ���С ����������
  * @�������   data�����ݰ���ʼ��ַ
                fileName ���ļ���
                length ���ļ���С
  * @�������   ��
  * @���ز���   ��
*******************************************************************************/
void Ymodem_PrepareIntialPacket(uint8_t *data, const uint8_t* fileName, uint32_t *length)
{
    uint16_t i, j;
    uint8_t file_ptr[10];

    //����ͷ3�����ݰ�
    data[0] = SOH;//��ʶ���ݰ�����128
    data[1] = 0x00;
    data[2] = 0xff;
    //�ļ������ݰ���Ч����
	  //�ӵ����ֽڿ�ʼ�洢�ļ���
    for (i = 0; (fileName[i] != '\0') && (i < FILE_NAME_LENGTH); i++)
    {
        data[i + PACKET_HEADER] = fileName[i];//�ļ��� �Ϊ 256�ֽ�
    }
		//��ʱiΪ�ļ����洢���ƫ����
    //�����ļ����洢��ĵ�ַ��ʼ �洢 �ļ���С
    data[i + PACKET_HEADER] = 0x00;//  ���ļ�������� ��0  ���ڽ���ʱ����

    Int2Str (file_ptr, *length);//�ļ���С ת�� �ַ�����ŵ� file_ptr��
    for (j =0, i = i + PACKET_HEADER + 1; file_ptr[j] != '\0' ; )
    {
        data[i++] = file_ptr[j++];//����ļ���С
    }
		// �����ǲ���ȱ��  ' ' �ո���ţ���
///////////////////////////////////////////////////////////////
		//������ʱ�ĸ�ʽ ��Ӧ��ȱ��һ�� �ո�� �������� ����Ϊ�������
		data[i + PACKET_HEADER] = ' ';//  ���ļ�������� ��0  ���ڽ���ʱ���
    for (j = i+1; j < PACKET_SIZE + PACKET_HEADER; j++)
    {
        data[j] = 0;//��һ�����ݰ� ����λ �� ��� �ļ������ļ���С������������Ͳ� 0
    }
////////////////////////////////////////////////////////////
/*		
    for (j = i; j < PACKET_SIZE + PACKET_HEADER; j++)
    {
        data[j] = 0;//��һ�����ݰ� ����λ �� ��� �ļ������ļ���С������������Ͳ� 0
    }
*/		 
}

/*******************************************************************************
  * @��������	Ymodem_PreparePacket
  * @����˵��   ׼�����ݰ�
  * @�������   SourceBuf������Դ����
                data�����ݰ�
                pktNo �����ݰ����
                sizeBlk �����ݰ�����
  * @�������   ��
  * @���ز���   ��
*******************************************************************************/
void Ymodem_PreparePacket(uint8_t *SourceBuf, uint8_t *data, uint8_t pktNo, uint32_t sizeBlk)
{
    uint16_t i, size, packetSize;
    uint8_t* file_ptr;

    //����ͷ3���ֽڵ� ���ݰ�ͷ
    packetSize = sizeBlk >= PACKET_1K_SIZE ? PACKET_1K_SIZE : PACKET_SIZE;
    size = sizeBlk < packetSize ? sizeBlk :packetSize;
	// ��һ���ֽ� Ϊ���������С ��ʾ 128/1024
    if (packetSize == PACKET_1K_SIZE)
    {
        data[0] = STX;
    }
    else
    {
        data[0] = SOH;
    }
// �ڶ����ֽ�
    data[1] = pktNo;   //���ݰ� ���� ��¼���ʹ���
// �������ֽ�
    data[2] = (~pktNo);//����ķ���
    file_ptr = SourceBuf;//�ļ�����

//////���ݰ��ڵ���Ч���� �ӵ������ֽڿ�ʼ ��������
    for (i = PACKET_HEADER; i < size + PACKET_HEADER; i++)
    {
        data[i] = *file_ptr++;//д���ļ�����
    }
		// Ӧ�����һ�����ݰ� �������ݴ�СС�� һ�������ݴ�С
    if ( size  <= packetSize) //���ݲ����Ļ�   ������
    {
        for (i = size + PACKET_HEADER; i < packetSize + PACKET_HEADER; i++)
        {
            data[i] = 0x1A; //����   
        }
    }
}

/*******************************************************************************
  * @��������	UpdateCRC16
  * @����˵��   �����������ݵģãң�У��
  * @�������   crcIn
                byte
  * @�������   ��
  * @���ز���   �ãң�У��ֵ
*******************************************************************************/
uint16_t UpdateCRC16(uint16_t crcIn, uint8_t byte)
{
    uint32_t crc = crcIn;
    uint32_t in = byte|0x100;
    do
    {
        crc <<= 1;
        in <<= 1;
        if (in&0x100)
            ++crc;
        if (crc&0x10000)
            crc ^= 0x1021;
    }
    while (!(in&0x10000));
    return crc&0xffffu;
}

/*******************************************************************************
  * @��������	  Cal_CRC16
  * @����˵��  �����������ݵ�CRCУ����
  * @�������   data ������
                size ������
  * @�������   ��
  * @���ز���   CRC У��ֵ
*******************************************************************************/
uint16_t Cal_CRC16(const uint8_t* data, uint32_t size)
{
    uint32_t crc = 0;
    const uint8_t* dataEnd = data+size;
    while (data<dataEnd)
        crc = UpdateCRC16(crc,*data++);

    crc = UpdateCRC16(crc,0);
    crc = UpdateCRC16(crc,0);
    return crc&0xffffu;
}


/*******************************************************************************
  * @��������	CalChecksum
  * @����˵��   ����YModem���ݰ����ܴ�С
  * @�������   data ������
                size ������
  * @�������   ��
  * @���ز���   ���ݰ����ܴ�С
*******************************************************************************/
uint8_t CalChecksum(const uint8_t* data, uint32_t size)
{
    uint32_t sum = 0;
    const uint8_t* dataEnd = data+size;
    while (data < dataEnd )
        sum += *data++;
    return sum&0xffu;
}

/*******************************************************************************
  * @��������	Ymodem_SendPacket
  * @����˵��   ͨ��ymodemЭ�鴫��һ�����ݰ�
#  ���ڴ��� ����
  * @�������   data �����ݵ�ַָ��
                length������
  * @�������   ��
  * @���ز���   ��
*******************************************************************************/
void Ymodem_SendPacket(uint8_t *data, uint16_t length)
{
    uint16_t i;
    i = 0;
    while (i < length)
    {
        Send_Byte(data[i]);
        i++;
    }
}

/*******************************************************************************
  * @��������	Ymodem_Transmit
  * @����˵��   ͨ��ymodemЭ�鴫��һ���ļ�
  * @�������   buf �����ݵ�ַָ��
                sendFileName ���ļ���
                sizeFile���ļ�����
  * @�������   ��
  * @���ز���   �Ƿ�ɹ�
                0���ɹ�
*******************************************************************************/
uint8_t Ymodem_Transmit (uint8_t *buf, const uint8_t* sendFileName, uint32_t sizeFile)
{

    uint8_t packet_data[PACKET_1K_SIZE + PACKET_OVERHEAD];//�������ݰ�����
    uint8_t FileName[FILE_NAME_LENGTH];//�ļ���
    uint8_t *buf_ptr, tempCheckSum ;
    uint16_t tempCRC, blkNumber;
    uint8_t receivedC[2], CRC16_F = 0, i;
    uint32_t errors, ackReceived, size = 0, pktSize;

    errors = 0;
    ackReceived = 0;
    for (i = 0; i < (FILE_NAME_LENGTH - 1); i++)
    {
        FileName[i] = sendFileName[i];//�����ļ���
    }
    CRC16_F = 1;// ��ҪУ����

    //׼����һ�����ݰ�           ���ݰ���ʼ��ַ   �ļ���   �ļ���С
    Ymodem_PrepareIntialPacket(&packet_data[0], FileName, &sizeFile);

    do
    {
///////���͵�һ�����ݰ� �ļ���   �ļ���С/////////////////////////////////////////////////////////////////////
        Ymodem_SendPacket(packet_data, PACKET_SIZE + PACKET_HEADER);//128 ��С+3
        //���������ֽڵ� CRCУ����  �߰�λ��ǰ  �Ͱ�λ�ں�
			  // ���ݰ����ٷ���  �����ֽڵ�  CRCУ����  ���� ������ ��������ĺ͵ĺ��λ
        if (CRC16_F)
        {
            tempCRC = Cal_CRC16(&packet_data[3], PACKET_SIZE);//����������ʼ���� �����ֽڵ� CRCУ����
            Send_Byte(tempCRC >> 8);//�ȷ��͸߰�λ
            Send_Byte(tempCRC & 0xFF);//�ٷ��͵Ͱ�λ
        }
        else //���� ��������ĺ͵ĺ��λ
        {
            tempCheckSum = CalChecksum (&packet_data[3], PACKET_SIZE);
            Send_Byte(tempCheckSum);
        }

        //�ȴ����ݽ��ն� ��Ӧ  ACK
        if (Receive_Byte(&receivedC[0], 10000) == 0)
        {
            if (receivedC[0] == ACK)
            {
                //��һ�� ���ݰ���ȷ����
                ackReceived = 1;
            }
        }
        else
        {
            errors++;
        }
    } while (!ackReceived && (errors < 0x0A)); //��һ�����ݰ�������ȷ �� �Լ� ������������ �Ͳ����͵�һ�����ݰ�

		// ����� ��һ�����ݰ����ʹ����������  ��ȡ�������ļ�  �����ش������
    if (errors >=  0x0A)
    {
        return errors;
    }
		
///////��һ�����ݰ��������֮��  ��ʼ�������� �ļ����ݰ�
    buf_ptr = buf; //�ļ���ŵ� �׵�ַ
    size = sizeFile;//�ļ���С
    blkNumber = 0x01;//���ݰ��ı��
		
////////////////////////////////////////////////////////////////////////////////
////���ͺ�������� 1024�ֽڵ����ݰ�����///////////////////
    while (size) 
    {
        //ÿ�η���ǰ ����׼������
	//׼����һ�����ݰ�   �վ��׵�ַ    ���ɵ�һ������   ���ݲ����Ļ� �������򲹳� ���з�
        Ymodem_PreparePacket(buf_ptr, &packet_data[0], blkNumber, size);
        ackReceived = 0;//���ݻ�Ӧ��־ ����
        receivedC[0]= 0;//��Ӧ��������
        errors = 0;     //�����������
        do
        {
        //������һ�����ݰ�
		//ȷ�����ݰ���С
            if (size >= PACKET_1K_SIZE)//���ݳ��� 1024��С �Ͱ� 1024�����ݰ�����
            {
                pktSize = PACKET_1K_SIZE;

            }
            else
            {
                pktSize = PACKET_SIZE;//���߰�128��С����
            }
		//ͨ�����ݽӿ�(���ڵ�)����׼���õ�һ������
            Ymodem_SendPacket(packet_data, pktSize + PACKET_HEADER);
            //����CRCУ��
            if (CRC16_F)
            {
                tempCRC = Cal_CRC16(&packet_data[3], pktSize);
                Send_Byte(tempCRC >> 8);   //�߰�λ
                Send_Byte(tempCRC & 0xFF); //�Ͱ�λ
            }
            else
            {
                tempCheckSum = CalChecksum (&packet_data[3], pktSize);
                Send_Byte(tempCheckSum);//���ݺ͵Ͱ�λ
            }

            //�ȴ����ն˻�Ӧ
            if ((Receive_Byte(&receivedC[0], 100000) == 0)  && (receivedC[0] == ACK))
            {
                ackReceived = 1; //�ѻ�Ӧ
                if (size > pktSize) //ʣ��δ�������� ����һ�����ݰ���С��
                {
                    buf_ptr += pktSize;// ���ݵ�ַƫ��
                    size -= pktSize;   // δ�������ݴ�С��Ӧ��С
                    if (blkNumber == (FLASH_IMAGE_SIZE/1024))//�����������flash��ַ��С���ش���
                    {
                        return 0xFF; //����
                    }
                    else
                    {
                        blkNumber++;//���ݰ� ���+1
                    }
                }
                else//�����Ѿ���������  ��Ϊ�������ݰ���ǰ  ���ݰ���С��С�ں�
                {
                    buf_ptr += pktSize;
                    size = 0; // ���ݴ�С��0 �������ݵķ���
                }
            }
            else
            {
                errors++;//δ��Ӧ  �������+1
            }
        } while (!ackReceived && (errors < 0x0A));
				
        //���û��Ӧ10�ξͷ��ش���
        if (errors >=  0x0A)
        {
            return errors;
        }

    }
//////////////////////////////////////////		
    ackReceived = 0;//��Ӧ��־����
    receivedC[0] = 0x00;//��Ӧ��������
    errors = 0;         //δ��Ӧ�����������
    do
    {
        Send_Byte(EOT);//���ͽ������ͱ�־
        //���� (EOT);
        //�ȴ���Ӧ
        if ((Receive_Byte(&receivedC[0], 10000) == 0)  && receivedC[0] == ACK)
        {
            ackReceived = 1;//�ѻ�Ӧ��־
        }
        else
        {
            errors++;//δ��Ӧ�������+1
        }
    } while (!ackReceived && (errors < 0x0A));

    if (errors >=  0x0A)
    {
        return errors;
    }
		
////////////////////////////////////////////////////////////////////////
////  ���� ��������֮�� ��Ҫ����һ��ȫΪ0�����ݰ�
    ackReceived = 0;
    receivedC[0] = 0x00;
    errors = 0;
////���ݰ�ǰ�����ֽ�
    packet_data[0] = SOH;//128�ֽڴ�С
    packet_data[1] = 0;
    packet_data [2] = 0xFF;
//׼���������� 
    for (i = PACKET_HEADER; i < (PACKET_SIZE + PACKET_HEADER); i++)
    {
        packet_data [i] = 0x00;//ȫΪ 0
    }

    do
    {
      	//ͨ�����ݽӿ�(���ڵ�)����׼���õ�һ������
        Ymodem_SendPacket(packet_data, PACKET_SIZE + PACKET_HEADER);
			
        //����CRCУ��
        tempCRC = Cal_CRC16(&packet_data[3], PACKET_SIZE);
        Send_Byte(tempCRC >> 8);
        Send_Byte(tempCRC & 0xFF);

        //�ȴ���Ӧ
        if (Receive_Byte(&receivedC[0], 10000) == 0)
        {
            if (receivedC[0] == ACK)
            {
                //��������ȷ
                ackReceived = 1;
            }
        }
        else
        {
            errors++;
        }

    } while (!ackReceived && (errors < 0x0A));
    //���û��Ӧ10�ξͷ��ش���
    if (errors >=  0x0A)
    {
        return errors;
    }

		ackReceived = 0;
    receivedC[0] = 0x00;
    errors = 0;
		//�����һ�η��ͽ������ͱ�־
    do
    {
        Send_Byte(EOT);
        //���� (EOT);
        //�ȴ���Ӧ
        if ((Receive_Byte(&receivedC[0], 10000) == 0)  && receivedC[0] == ACK)
        {
            ackReceived = 1;
        }
        else
        {
            errors++;
        }
    } while (!ackReceived && (errors < 0x0A));

    if (errors >=  0x0A)
    {
        return errors;
    }
    return 0;//�ļ�����ɹ�
}

/*******************************�ļ�����***************************************/
