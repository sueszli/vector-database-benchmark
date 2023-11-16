
#include "DS1302.h"
#include "ets_sys.h"
#include "osapi.h"
#include "gpio.h"

void ICACHE_FLASH_ATTR
DS1302_master_gpio_init(void) {
	ETS_GPIO_INTR_DISABLE();
//    ETS_INTR_LOCK();

	PIN_FUNC_SELECT(DS1302_MASTER_IO_MUX, DS1302_MASTER_IO_FUNC);
	PIN_FUNC_SELECT(DS1302_MASTER_SCLK_MUX, DS1302_MASTER_SCLK_FUNC);
	PIN_FUNC_SELECT(DS1302_MASTER_RST_MUX, DS1302_MASTER_RST_FUNC);

	GPIO_REG_WRITE(GPIO_PIN_ADDR(GPIO_ID_PIN(DS1302_MASTER_IO_GPIO)),
			GPIO_REG_READ(GPIO_PIN_ADDR(GPIO_ID_PIN(DS1302_MASTER_IO_GPIO))) | GPIO_PIN_PAD_DRIVER_SET(GPIO_PAD_DRIVER_ENABLE)); //open drain;
	GPIO_REG_WRITE(GPIO_ENABLE_ADDRESS,
			GPIO_REG_READ(GPIO_ENABLE_ADDRESS) | (1 << DS1302_MASTER_IO_GPIO));

	GPIO_REG_WRITE(GPIO_PIN_ADDR(GPIO_ID_PIN(DS1302_MASTER_SCLK_GPIO)),
			GPIO_REG_READ(GPIO_PIN_ADDR(GPIO_ID_PIN(DS1302_MASTER_SCLK_GPIO))) | GPIO_PIN_PAD_DRIVER_SET(GPIO_PAD_DRIVER_ENABLE)); //open drain;
	GPIO_REG_WRITE(GPIO_ENABLE_ADDRESS,
			GPIO_REG_READ(GPIO_ENABLE_ADDRESS) | (1 << DS1302_MASTER_SCLK_GPIO));

	GPIO_REG_WRITE(GPIO_PIN_ADDR(GPIO_ID_PIN(DS1302_MASTER_RST_GPIO)),
			GPIO_REG_READ(GPIO_PIN_ADDR(GPIO_ID_PIN(DS1302_MASTER_RST_GPIO))) | GPIO_PIN_PAD_DRIVER_SET(GPIO_PAD_DRIVER_ENABLE)); //open drain;
	GPIO_REG_WRITE(GPIO_ENABLE_ADDRESS,
			GPIO_REG_READ(GPIO_ENABLE_ADDRESS) | (1 << DS1302_MASTER_RST_GPIO));

	DS1302_MASTER_SCLK_LOW_RST_LOW(); //��ʼ��RST��SCLK

	ETS_GPIO_INTR_ENABLE();

}
void ICACHE_FLASH_ATTR
DS1302_master_writeByte(uint8 addr, uint8 wrdata) //��DS1302д��һ�ֽ�����
{
	uint8 i;
	GPIO_OUTPUT_SET(GPIO_ID_PIN(DS1302_MASTER_RST_GPIO), 1);        //����DS1302����
	os_delay_us(1);
	//д��Ŀ���ַ��addr
	addr = addr & 0xFE;   //���λ���㣬�Ĵ���0λΪ0ʱд��Ϊ1ʱ��
	for (i = 0; i < 8; i++) {
		if (addr & 0x01) {
			GPIO_OUTPUT_SET(GPIO_ID_PIN(DS1302_MASTER_IO_GPIO), 1);
		} else {
			GPIO_OUTPUT_SET(GPIO_ID_PIN(DS1302_MASTER_IO_GPIO), 0);
		}
		ets_delay_us(1);
		GPIO_OUTPUT_SET(GPIO_ID_PIN(DS1302_MASTER_SCLK_GPIO), 1);   //����������ʱ��
		ets_delay_us(1);
		GPIO_OUTPUT_SET(GPIO_ID_PIN(DS1302_MASTER_SCLK_GPIO), 0);
		ets_delay_us(1);
		;
		addr = addr >> 1;
	}
	//д�����ݣ�wrdata
	for (i = 0; i < 8; i++) {
		if (wrdata & 0x01) {
			GPIO_OUTPUT_SET(GPIO_ID_PIN(DS1302_MASTER_IO_GPIO), 1);
		} else {
			GPIO_OUTPUT_SET(GPIO_ID_PIN(DS1302_MASTER_IO_GPIO), 0);
		}
		ets_delay_us(1);
		GPIO_OUTPUT_SET(GPIO_ID_PIN(DS1302_MASTER_SCLK_GPIO), 1);   //����������ʱ��
		ets_delay_us(1);
		GPIO_OUTPUT_SET(GPIO_ID_PIN(DS1302_MASTER_SCLK_GPIO), 0);
		ets_delay_us(1);
		;
		wrdata = wrdata >> 1;
	}
	GPIO_OUTPUT_SET(GPIO_ID_PIN(DS1302_MASTER_RST_GPIO), 0);		//ֹͣDS1302����
	os_delay_us(1);
}

uint8 ICACHE_FLASH_ATTR
DS1302_master_readByte(uint8 addr) //��DS1302����һ�ֽ�����
{
	uint8 i, temp = 0; //ע�⣺temp��������ʼ������Ȼ�������������Ǵ���ģ������˼���ŷ������bug
	GPIO_OUTPUT_SET(GPIO_ID_PIN(DS1302_MASTER_RST_GPIO), 1);        //����DS1302����
	os_delay_us(1);
	//д��Ŀ���ַ��addr
	addr = addr | 0x01;    //���λ�øߣ��Ĵ���0λΪ0ʱд��Ϊ1ʱ��
	for (i = 8; i > 0; i--) {
		if (addr & 0x01) {
			GPIO_OUTPUT_SET(GPIO_ID_PIN(DS1302_MASTER_IO_GPIO), 1);
		} else {
			GPIO_OUTPUT_SET(GPIO_ID_PIN(DS1302_MASTER_IO_GPIO), 0);
		}
		ets_delay_us(1);
		GPIO_OUTPUT_SET(GPIO_ID_PIN(DS1302_MASTER_SCLK_GPIO), 1);   //����������ʱ��
		ets_delay_us(1);
		GPIO_OUTPUT_SET(GPIO_ID_PIN(DS1302_MASTER_SCLK_GPIO), 0);
		ets_delay_us(1);
		addr = addr >> 1;
	}
	//��DS1302���ݣ�temp
	for (i = 8; i > 0; i--) {
		temp = temp >> 1;
		os_delay_us(1);
		if (GPIO_INPUT_GET(GPIO_ID_PIN(DS1302_MASTER_IO_GPIO))) {
			temp |= 0x80;
		} else {
			temp &= 0x7F;
		}
		ets_delay_us(1);
		GPIO_OUTPUT_SET(GPIO_ID_PIN(DS1302_MASTER_SCLK_GPIO), 1);   //����������ʱ��
		ets_delay_us(1);
		GPIO_OUTPUT_SET(GPIO_ID_PIN(DS1302_MASTER_SCLK_GPIO), 0);
		ets_delay_us(1);
	}
	GPIO_OUTPUT_SET(GPIO_ID_PIN(DS1302_MASTER_RST_GPIO), 0);		//ֹͣDS1302����
	os_delay_us(1);
	return temp;
}
//��DS302д��ʱ������
void ICACHE_FLASH_ATTR DS1302_Clock_init(unsigned char *pDate)		//��ʼ��ʱ������
{

	int i = 0;
	for (i = 0; i < 7; i++) {
		os_printf("DS1302_Clock_init : %d \n", pDate[i]);
	}
	DS1302_master_writeByte(ds1302_control_add,0x00); //�ر�д����
	DS1302_master_writeByte(ds1302_sec_add,0x80); //��ͣ
	DS1302_master_writeByte(ds1302_charger_add,0xa9); //������
	DS1302_master_writeByte(ds1302_year_add,pDate[1]); //��
	DS1302_master_writeByte(ds1302_month_add,pDate[2]); //��
	DS1302_master_writeByte(ds1302_date_add,pDate[3]); //��
	DS1302_master_writeByte(ds1302_day_add,pDate[7]); //��
	DS1302_master_writeByte(ds1302_hr_add,pDate[4]); //ʱ
	DS1302_master_writeByte(ds1302_min_add,pDate[5]); //��
	DS1302_master_writeByte(ds1302_sec_add,pDate[6]); //��
	DS1302_master_writeByte(ds1302_day_add,pDate[7]); //��
	DS1302_master_writeByte(ds1302_control_add,0x80); //��д����
}

