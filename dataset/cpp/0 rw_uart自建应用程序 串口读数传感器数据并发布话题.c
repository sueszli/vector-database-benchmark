/****************************************************************************
 *
�����ǲ��ô��ڵķ�ʽ���һ���Զ��崫����������������

��֪���ǳ�����ģ����ͨ�����ڷ�ʽ���ͣ�Tx�������ݣ�ʹ�õ�ģ�����ݷ�������Ϊ100ms�����ݸ�ʽΪ��

R0034 R0122 R0122 R0046 R0127 R0044 R0044 R0125 R0034 R0037 R0041 R0122 R0122 .....
1
1
�����ͨ��Pixhawk���ϵĴ��������գ�Rx�����ݣ�������������Tx�ӿ����ӵ�Pixhawk���ϵ�Rx�ӿڡ� ����뷨��Ŀǰ��˵�Ƿǳ��ò���ʵʩ�����򵥵ģ��Ͼ�PX4Դ���ܴܺ�ʮ�ָ��ӣ�����ԭ�еĻ����ϸĴ�����Ҫ������ϵͳ���ź������⣬���г�ѧ��ȷʵ���Բ������֡�Эͬ���ķ�ʽ���뵽Pixhawk��������һ���ⲿ��������ǿ�Ĵ����������̬��λ�û��Ӿ����ݾ������ڷ��͵�Pixhawk�У���ô����Щ�����ںϽ�ԭ���㷨�Ǿ���˵�ˡ�

OK���������⣬Pixhawk���ϴ���˵�����£�

NuttX UART	Pixhawk UART
/dev/ttyS0	IO DEBUG(RX ONLY)
/dev/ttyS1	TELEM1(USART2)
/dev/ttyS2	TELEM2(USART3)
/dev/ttyS3	GPS(UART4)
/dev/ttyS4	N/A(UART5, IO link)
/dev/ttyS5	SERIAL5(UART7,NSH Console Only)
/dev/ttyS6	SERIAL4(UART8)
����ʹ��Pixhawk����TELEM2�ӿڵ�USART2����Ӧ��Nuttx UART�豸�ļ�β/dev/ttyS2��

�ӿ�����  http://www.pixhawk.com/modules/pixhawk

�����ʵ�ֹ�����鿴irmware/src/drivers/hott/comms.cpp��
�����ǹ��ڴ��ڵ����ã���Ƶ�NuttX�����Unix����ϵͳ���ļ�ϵͳ�ˣ�С���е��ѿС�


2. ��rw_uart�ļ����д���CMakeLists.txt�ļ����������������ݣ�

set(MODULE_CFLAGS)
px4_add_module(
    MODULE modules__rw_uart
    MAIN rw_uart
    COMPILE_FLAGS
        -Os
    SRCS
        rw_uart.c
    DEPENDS
        platforms__common
    )
# vim: set noet ft=cmake fenc=utf-8 ff=unix : 

3. ע������ӵ�Ӧ�õ�NuttShell�С�Firmware/cmake/configs/nuttx_px4fmu-v2_default.cmake

�ļ�������������ݣ�
	#
	# General system control
	#
	modules/commander  #������ָ�ӹ�
	modules/events     #�¼��жϴ���
	modules/load_mon
	modules/navigator  #����
	modules/mavlink    #����λ��ͨ��
	modules/gpio_led   #ͨ��IO��
	#modules/uavcan
	modules/land_detector#��½���
	
# ����� ģ����  ���ڶ�ȡ ��������
	modules/rw_uart
	
4. ���벢�ϴ��̼�
make px4fmu-v2_default upload



5. ��ȡ����
�鿴app 
��NSH�ն�������help����Builtin Apps�г���rw_uartӦ�á� 

����rw_uartӦ�ã�ǰ����ģ����Pixhawk���Ӻã� 

���ú÷���װ�� 

������ʹ�õ��Ǳ���STM32�������USART1��Pixhawk��TELEM2���Ͷ�Ӧ���ַ�����
���ݴ��룬���ݵĸ�ʽҪ����ϸ񣬲�Ȼ�ܿ��ܵò������ݣ�������ʼ�����ô��ڷ��ķ�Rxxxx��
����ٶ�̫���ǲ��еģ�����������һ����ʱ����������Ƶ������Ϊ20Hz����ʱ���ǻ���������֡�
��λ�������õ�ʱ����Ҫע�Ⲩ���ʶ�Ӧ�����ݸ�ʽ��Ҫ������	

int main(void)
{   

  // USARTx config 9600 8-N-1 //
    USARTx_Config();    

   //����SysTickΪÿ10us�ж�һ��//
    SysTick_Init();

    printf("R1000 ");

  for(;;){
        int i = 1000;
        for(i=1001;i<2000;i++)
        {
            printf("R%d ",i);
            Delay_us(5000); // 5000*10us = 50ms
        }
    }
}


�����ͳн�ǰһƪ����FreeApe�Ĵ�����ӳ��������������ĺ�벿�ֽ���ѧϰ��


Ϊʲô��ǰ���أ���Ϊ����α���������ѵ�Ƭ���ô��ڷ��͵��й��ɵ����ݵ����˴�������ȡ�������ݡ������ް��ġ�

����������Ubuntu  Firmware 1.4.1

�����˻�����ʱ��������Ҫ��Ӧ����ϵͳ����ʱ�����������ģ��ҽ���õĳ��������ݲ��ϵķ�����ȥ��

�Ӷ�������Ӧ�õ��Զ���ʹ�á�����Ҳʹ��Pixhawk�����ͨ��ģʽ�������̣߳����app�������룬����һ���߳������ϵķ������ݡ�
 
 ��������ͷ�������

��Firmware/msgĿ¼���½�read_uart_sensor.msg�ļ���������
char[4] datastr
int16 data

# TOPICS read_uart_sensor   
ע�� 
# TOPICS # �ź�TOPICS�м���һ���ո�

����ӵ�CMakeLists.txt��( Firmware/msg�е�CMakeLists.txt����ӣ�read_uart_sensor.msg)��
������Զ�����uORB/topics/read_uart_sensor.hͷ�ļ�
 
 
���ڶ�ȡ����

��Firmware/src/modulesĿ¼���½��ļ���read_uart_sensor

����ļ�read_uart_sensor.c��������

 * read_uart_sensor.c
 * 
 * read sensor through uart


#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#include <stdbool.h>
#include <errno.h>
#include <drivers/drv_hrt.h>
#include <string.h>
#include <systemlib/err.h>
#include <systemlib/systemlib.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <uORB/topics/read_uart_sensor.h>

__EXPORT int read_uart_sensor_main(int argc, char *argv[]);


static bool thread_should_exit = false;//Ddemon exit flag// 
static bool thread_running = false; ///Daemon status flag// 
static int daemon_task;

// 
//Main loop
// 
int read_uart_thread_main(int argc, char *argv[]);

static int uart_init(const char * uart_name);
static int set_uart_baudrate(const int fd, unsigned int baud);
static void usage(const char *reason);

int set_uart_baudrate(const int fd, unsigned int baud)
{
    int speed;

    switch (baud) {
        case 9600:   speed = B9600;   break;
        case 19200:  speed = B19200;  break;
        case 38400:  speed = B38400;  break;
        case 57600:  speed = B57600;  break;
        case 115200: speed = B115200; break;
        default:
            warnx("ERR: baudrate: %d\n", baud);
            return -EINVAL;
    }

    struct termios uart_config;

    int termios_state;

  //fill the struct for the new configuration // 
    tcgetattr(fd, &uart_config);
	 
   //clear ONLCR flag (which appends a CR for every LF)// 
    uart_config.c_oflag &= ~ONLCR;
    
   // no parity, one stop bit// 
    uart_config.c_cflag &= ~(CSTOPB | PARENB);
   
   
   //set baud rate // 
    if ((termios_state = cfsetispeed(&uart_config, speed)) < 0) {
        warnx("ERR: %d (cfsetispeed)\n", termios_state);
        return false;
    }

    if ((termios_state = cfsetospeed(&uart_config, speed)) < 0) {
        warnx("ERR: %d (cfsetospeed)\n", termios_state);
        return false;
    }

    if ((termios_state = tcsetattr(fd, TCSANOW, &uart_config)) < 0) {
        warnx("ERR: %d (tcsetattr)\n", termios_state);
        return false;
    }

    return true;
}


int uart_init(const char * uart_name)
{
    int serial_fd = open(uart_name, O_RDWR | O_NOCTTY);

    if (serial_fd < 0) {
        err(1, "failed to open port: %s", uart_name);
        return false;
    }
    return serial_fd;
}

static void usage(const char *reason)
{
    if (reason) {
        fprintf(stderr, "%s\n", reason);
    }

    fprintf(stderr, "usage: read_uart_sensor {start|stop|status} [param]\n\n");
    exit(1);
}

int read_uart_sensor_main(int argc, char *argv[])
{
    if (argc < 2) {
        usage("[Fantasy]missing command");
    }

    if (!strcmp(argv[1], "start")) {
        if (thread_running) {
            warnx("[Fantasy]already running\n");
            return 0;
        }

        thread_should_exit = false;
        daemon_task = px4_task_spawn_cmd("read_uart_sensor",
                         SCHED_DEFAULT,
                         SCHED_PRIORITY_MAX - 5,
                         2000,
                         read_uart_thread_main,
                         (argv) ? (char * const *)&argv[2] : (char * const *)NULL);
        return 0;
    }

    if (!strcmp(argv[1], "stop")) {
        thread_should_exit = true;
        return 0;
    }

    if (!strcmp(argv[1], "status")) {
        if (thread_running) {
            warnx("[Fantasy]running");
            return 0;

        } else {
            warnx("[Fantasy]stopped");
            return 1;
        }

        return 0;
    }

    usage("unrecognized command");
    return 1;
}

int read_uart_thread_main(int argc, char *argv[])
{

    if (argc < 2) {
        errx(1, "[Fantasy]need a serial port name as argument");
        usage("eg:");
    }

    const char *uart_name = argv[1];

    warnx("[Fantasy]opening port %s", uart_name);
    char data = '0';
    char buffer[4] = "";
 // 
     * TELEM1 : /dev/ttyS1
     * TELEM2 : /dev/ttyS2
     * GPS    : /dev/ttyS3
     * NSH    : /dev/ttyS5
     * SERIAL4: /dev/ttyS6
     * N/A    : /dev/ttyS4
     * IO DEBUG (RX only):/dev/ttyS0
   // 
    int uart_read = uart_init(uart_name);
    if(false == uart_read)return -1;
    if(false == set_uart_baudrate(uart_read,9600)){
        printf("[Fantasy]set_uart_baudrate is failed\n");
        return -1;
    }
    printf("[Fantasy]uart init is successful\n");

    thread_running = true;

   //��ʼ�����ݽṹ�� // 
    struct read_uart_sensor_s sensordata;
    memset(&sensordata, 0, sizeof(sensordata));
   //�������� // 
    orb_advert_t read_uart_sensor_pub = orb_advertise(ORB_ID(read_uart_sensor), &sensordata);

    while(!thread_should_exit){
        read(uart_read,&data,1);
        if(data == 'R'){
            for(int i = 0;i <4;++i){
                read(uart_read,&data,1);
                buffer[i] = data;
                data = '0';
            }
            strncpy(sensordata.datastr,buffer,4);// �����ַ���Buffer��ǰ�������ֵ�Datastr��
            sensordata.data = atoi(sensordata.datastr);//���ַ���ת������������
            //printf("[Fantasy]sensor.data=%d\n",sensordata.data);
            orb_publish(ORB_ID(read_uart_sensor), read_uart_sensor_pub, &sensordata);
        }
    }

    warnx("[Fantasy]exiting");
    thread_running = false;
    close(uart_read);

    fflush(stdout);
    return 0;
}


���CMakeLists.txt�ļ�
set(MODULE_CFLAGS)
px4_add_module(
        MODULE modules__read_uart_sensor
        MAIN read_uart_sensor
    COMPILE_FLAGS
        -Os
    SRCS
                read_uart_sensor.c
    DEPENDS
        platforms__common
    )
# vim: set noet ft=cmake fenc=utf-8 ff=unix : 



��Firmware/cmake/configs/nuttx/nuttx_px4fmu-v2_default.cmake��ע���ģ��


���Է���������

���Կ������һ��������app�н������ⶩ�ģ�Ȼ�󽫶��ĵ����ݴ�ӡ���������Ƿ��ǳ����������ݡ������½�һ��Ӧ��px4_test���в��ԡ�


 * px4_test.c
 *  
 * test the uart sensor app

#include <px4_config.h>
#include <px4_tasks.h>
#include <px4_posix.h>
#include <unistd.h>
#include <stdio.h>
#include <poll.h>
#include <string.h>
#include <math.h>

#include <uORB/uORB.h>
#include <uORB/topics/read_uart_sensor.h>

__EXPORT int px4_test_main(int argc, char *argv[]);

int px4_test_main(int argc, char *argv[])
{
    PX4_INFO("Hello Sky!\n");

  
  //subscribe to rw_uart_sensor topic/// 
    int sensor_sub_fd = orb_subscribe(ORB_ID(read_uart_sensor));

    //limit the update rate to 20 Hz // 
    orb_set_interval(sensor_sub_fd, 50);


    //one could wait for multiple topics with this technique, just using one here // 
    px4_pollfd_struct_t fds[] = {
        { .fd = sensor_sub_fd,   .events = POLLIN },
       //there could be more file descriptors here, in the form like:
        //{ .fd = other_sub_fd,   .events = POLLIN },
        // 
    };

    int error_counter = 0;

    for (int i = 0; ; i++) { // infinite loop
      // wait for sensor update of 1 file descriptor for 1000 ms (1 second) // 
        int poll_ret = poll(fds, 1, 1000);

       // handle the poll result // 
        if (poll_ret == 0) {
        	
           //this means none of our providers is giving us data // 
            printf("[px4_test] Got no data within a second\n");

        } else if (poll_ret < 0) {
           // this is seriously bad - should be an emergency // 
            if (error_counter < 10 || error_counter % 50 == 0) {
                //use a counter to prevent flooding (and slowing us down)/ 
                printf("[px4_test] ERROR return value from poll(): %d\n", poll_ret);
            }

            error_counter++;

        } else {

            if (fds[0].revents & POLLIN) {
            	
                / obtained data for the first file descriptor / 
                
                struct read_uart_sensor_s raw;
                
                /copy sensors raw data into local buffer / 
                orb_copy(ORB_ID(read_uart_sensor), sensor_sub_fd, &raw);
                PX4_INFO("[px4_test] sensor data:\t%s\t%d\n",
                       raw.datastr,
                       raw.data);
            }

           /there could be more file descriptors here, in the form like:
           /if (fds[1..n].revents & POLLIN) {}
            / 
        }
    }

    PX4_INFO("exiting");

    return 0;
}


���벢�ϴ��̼�

make px4fmu-v2_default upload
��NSH�в���

read_uart_sensor start /dev/ttyS2
px4_test


��ӵ��ű��ļ�

��rcS�з���mavlink��������ʽ����������Ӧ�ã�ʹ����ttyS2���ӵ�����Ĭ��Ϊ����״̬ 

read_uart_sensor start  /dev/ttyS2


�������ttyS2���������豸�������Զ������ģ���nsh�п���ֱ�ӵ���px4_testӦ�þͿ��Զ�ȡ�����ˣ���ʹ��px4_simple_app��ȡ�ڲ��������ķ�ʽ���졣
����ģ�;��������ˣ����������Ǿ���Ĵ����Ż��Լ�Ӧ���ˡ�
 ****************************************************************************/




 
/* 
 * ���ڶ�ȡ����
 * rw_uart.c 
 */
#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#include <stdbool.h>
#include <errno.h>
#include <drivers/drv_hrt.h>
#include <string.h>
#include <systemlib/err.h>
#include <systemlib/systemlib.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

__EXPORT int rw_uart_main(int argc, char *argv[]);

static int uart_init(char * uart_name);
static int set_uart_baudrate(const int fd, unsigned int baud);

int set_uart_baudrate(const int fd, unsigned int baud)
{
    int speed;

    switch (baud) {
        case 9600:   speed = B9600;   break;
        case 19200:  speed = B19200;  break;
        case 38400:  speed = B38400;  break;
        case 57600:  speed = B57600;  break;
        case 115200: speed = B115200; break;
        default:
            warnx("ERR: baudrate: %d\n", baud);
            return -EINVAL;
    }

    struct termios uart_config;

    int termios_state;

    /* ���µ��������ṹ�� */
    /* ����ĳ��ѡ���ô��ʹ��"|="���㣬
     * ����ر�ĳ��ѡ���ʹ��"&="��"~"����
     * */
    tcgetattr(fd, &uart_config); // ��ȡ�ն˲���

    /* clear ONLCR flag (which appends a CR for every LF) */
    uart_config.c_oflag &= ~ONLCR;// ��NLת����CR(�س�)-NL�������

    /* ��żУ�飬һ��ֹͣλ */
    uart_config.c_cflag &= ~(CSTOPB | PARENB);// CSTOPB ʹ������ֹͣλ��PARENB ��ʾżУ��

     /* ���ò����� */
    if ((termios_state = cfsetispeed(&uart_config, speed)) < 0) {
        warnx("ERR: %d (cfsetispeed)\n", termios_state);
        return false;
    }

    if ((termios_state = cfsetospeed(&uart_config, speed)) < 0) {
        warnx("ERR: %d (cfsetospeed)\n", termios_state);
        return false;
    }
    // �������ն���صĲ�����TCSANOW �����ı����
    if ((termios_state = tcsetattr(fd, TCSANOW, &uart_config)) < 0) {
        warnx("ERR: %d (tcsetattr)\n", termios_state);
        return false;
    }

    return true;
}


int uart_init(char * uart_name)
{
    int serial_fd = open(uart_name, O_RDWR | O_NOCTTY);
    /*Linux�У�������ļ����򿪴����豸�ʹ���ͨ�ļ�һ����ʹ�õ���open����ϵͳ����*/
    // ѡ�� O_NOCTTY ��ʾ���ܰѱ����ڵ��ɿ����նˣ������û��ļ���������Ϣ��Ӱ������ִ��
    if (serial_fd < 0) {
        err(1, "failed to open port: %s", uart_name);
        return false;
    }
//    printf("Open the %s\n",serial_fd);
    return serial_fd;
}

int rw_uart_main(int argc, char *argv[])
{
    char data = '0';
    char buffer[4] = "";
    /*
     * TELEM1 : /dev/ttyS1
     * TELEM2 : /dev/ttyS2
     * GPS    : /dev/ttyS3
     * NSH    : /dev/ttyS5
     * SERIAL4: /dev/ttyS6
     * N/A    : /dev/ttyS4
     * IO DEBUG (RX only):/dev/ttyS0
     */
    int uart_read = uart_init("/dev/ttyS2");
    if(false == uart_read)
        return -1;
    if(false == set_uart_baudrate(uart_read,9600)){
        printf("[JXF]set_uart_baudrate is failed\n");
        return -1;
    }
    printf("[JXF]uart init is successful\n");

    while(true){
        read(uart_read,&data,1);
        if(data == 'R'){
            for(int i = 0;i <4;++i){
                read(uart_read,&data,1);
                buffer[i] = data;
                data = '0';
            }
            printf("%s\n",buffer);
        }
    }

    return 0;
}
