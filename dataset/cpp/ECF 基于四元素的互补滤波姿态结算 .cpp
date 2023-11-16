/****************************************************************************
 \Firmware\src\modules\attitude_estimator_q
 
 http://blog.csdn.net/luoshi006/article/details/51513580
 
 ���ʽ���

1��������
�����ǣ��������ٶȣ����и߶�̬���ԣ�����һ����Ӳ����Ƕȵ�������
���������ǽǶȵĵ����������ٶȣ�Ҫ�����ٶȶ�ʱ����ֲ��ܵõ��Ƕȡ�
�������������Ӱ�죬�ڻ��������²��ϻ��ۣ����յ��������ǵĵ�Ƶ���ź�Ư�ơ�


2�����ٶȼ�
�����ǰ���ٶȣ������������ٶ� g���ķ���Ҳ��������Ӧ����������ͣʱ�����Ϊ g��
�������ԭ���µĸ�Ƶ�ź����У�ʹ�ü��ٶȼ����񶯻����и�Ƶ���Žϴ�

3��������
���Ϊ��ǰ������شų��ļнǡ�����ԭ����ָ�������ơ���Ƶ���ԽϺã�������Χ�ų����š� 
�����ƵĹ���ԭ��ο���WalkAnt�Ĳ���

4������ϵ
��������ϵ���ڶ������У��ֽе�������ϵ����������ϵ��ͨ�������ñ����أ�NED����������ϵ�� X,Y,Z �ᡣ
��������ϵ �������ڶ�����������ϣ���������ϵԭ�㶨λ�ڷ��������ĵ㣨�������ĵ������ĵ��غϣ���
���ں��յ����õ�������ϵ����ο� AIAA ϵ�д��顣�ڶ������У���Ϊֻ�ڵͿշ��У���ʱ��϶̣�ֻ��Ҫ����������


5����̬��ʾ
ŷ���� ����ֱ�ۣ�������������άŷʽ�ռ��е���̬����ʱ������ϵΪ��������ϵ�����Ÿ������ת����ת��ȱ�㣺��������� 
��Ԫ������һ����ά��������ʾ�������ά��ת���ʺ����ڼ�������㡣 
�������Ҿ���DCM����ŷ��������ֵ����Ԫ������ʾ������ϵ����ת��

6��mahony �˲�ԭ��
�����˲�Ҫ�������źŵĸ����������ڲ�ͬ��Ƶ�ʣ�ͨ�����������˲����Ľ�ֹƵ�ʣ�ȷ���ںϺ���ź��ܹ���������Ƶ�ʡ� 
�� IMU ����̬�����У������˲����������ǣ���Ƶ������ʹ�ø�ͨ�˲����Լ��ٶ�/�����ƣ���Ƶ������ʹ�õ�Ƶ�˲��� 
���˴���δ�Դ���������ʵ�⣬����������Ƶ��δ֪�������������ڲ��㣩

�����˲��У������˲����Ľ�ֹƵ��һ�£���ʱ����Ҫ��������Ƶ�ʵ�ֵ�������ȡ�ᡣ
����ˮƽʱ�����ٶȼ��޷������� Z �����ת������ƫ���ǡ�������Ҳ��ͬ�����⣬�޷����Ҫ�������ת����
�ʣ���Ҫ���ٶȼƺʹ�����ͬʱ�������ǽ���У����


7 Ԥ��- ��Ԫ����Ԥ�� 
�뿨�����˲����ƣ�mahony �˲�Ҳ��ΪԤ��-У�����¡� 
��Ԥ�⻷�ڣ������������ǲ�õĽ��ٶȣ�ͨ��ʽ��1���������Ԫ����̬Ԥ�⡣
qbe ��ʾ�ӵ�������ϵ����������ϵ�����������ϵ��̬�ڵ�������ϵ�µı�ʾ��
 
8  У��
��Ԥ�⻷�ڵõ�����Ԫ�� qbe(k) ��ͨ�����ٶȼƺʹ����Ƶ�ֵ����У�����û���ͨ����Ϊ�����֣�
���ٶ� У����Ԫ���� �� ��� roll  �Լ� ���� pitch
������ У����Ԫ���� �� ƫ�� yaw  
 
    ���ٶ� У��
        ���ٶȼ��ź����Ⱦ�����ͨ�˲�����������Ƶ��������
           y(k)  =  RC/(T+RC)*y(k-1) + T/(T+RC)*x(k)
        Ȼ�󣬶Եõ��Ľ�����й�һ����normalized��
           det_q_acc /= ||det_q_acc||  
        ����ƫ�
		   e_acc =    det_q_acc ���  v  ʽ�У� v ��ʾ���������ڻ�������ϵ������
		   v =[vx vy vz]ת�� = [2(q1*q3-q0*q2) 2(q2*q3+q0*q1)  q0^2-q1^2-q2^2+q3^2] ת�� 
	    ��ʱ���� v ����ٶȼ�������ֱ������ˣ��õ����ֵ������������������ͬ�������������ϲ��Ϊ�㡿
		   
    ������ У�� 
         ����Ԥ��������ٶȼ���ͬ�����˲���Ȼ���һ���õ� det_q_mag
		 1. �� GPS У׼ʱ��
		    ������ e_mag = det_q_mag ���  w
			    w ���㷽���� ��������� ��������ϵ ת���� ��������ϵ  ��ϵ�� Cb-n   H = C M
    			             ����XOZƽ���ͶӰ   B   bx = sqrt(hx^2 + hy^2)   bz = hz  by = 0
							 �ٴα任����������ϵ�� W = Cn-b * B			 
 		2. �� GPS У׼ʱ��
 		    ������ e_mag = det_q_mag ���  w
 		       w ���㷽���� �� px4 �У�������ʹ�� GPS ��Ϣ [0,0,mag] ����У׼���ʣ���ʽ����ٶȼ���ͬ��
 		                    W = Cn-b * [0,0,mag]
     	��ʱ���� w �������������ˣ��õ����ֵ������������������ͬ�������������ϲ��Ϊ�㡿

9 ������Ԫ��

�ɼ��ٶȼƺʹ�����У׼�õ������ֵ��
    e=e_acc+e_mag
�ɸ����ֵ�õ�����ֵ��
    �� = Kp * e  + Ki*sum(e)  ����������
    
������Ľ��ٶ�ֵ��
��= w_gyro+��		 
 
 ����һ����������������һ��΢�ַ��̣�
     q�B=f(q,��)
     q(t+T)=q(t)+T?f(q,��)	 
	 
���������Ԫ��΢�ַ��̵Ĳ����ʽ��

q0(t+T)=q0(t)+T/2[-w_x*q1(t)-w_y*q2(t)-w_z*q3(t)]	 

 ��Ԫ���淶����
 	  q = (q0+q1*i+q2*j+q3*k)/sqrt(q0^2 + q1^2 + q2^2 + q3^2)




	 
	 	
 ������Ԫ�ص���̬����    ��̬������һ���ֺ���Ҫ����Ҫ�Ļ������ǹ��Ե����Ͷഫ���������ں�
 http://blog.csdn.net/gen_ye/article/details/52522721
 http://blog.csdn.net/gen_ye/article/details/52524190
 
http://blog.csdn.net/qq_21842557/article/details/50923863 
 
  ��̬���㳣�õ��㷨��ŷ���Ƿ����������ҷ�����Ԫ������ ŷ���Ƿ��������̬ʱ������㣨���������������������ȫ��̬�Ľ��㣻
   �������ҿ�����ȫ��̬�Ľ��㵫�������󣬲�������ʵʱ��Ҫ�� 
  ��Ԫ�������������С��������ҿ�������������˶���������̬��ʵʱ���㡣

   ��̬�����ԭ������һ��ȷ�����������ò�ͬ������ϵ��ʾʱ����������ʾ�Ĵ�С�ͷ���һ������ͬ�ġ�
   ������������������ϵ����ת�����������ô��һ������������ôһ���������ڵ���ת�����
   ����һ������ϵ�п϶�������ֵ����ƫ��ģ�����ͨ�����ƫ�������������ת���������ת�����Ԫ������Ԫ����
   ���������ľ�����Ԫ����������̬�ͱ������ˡ�

   �����Ƕ�̬��Ӧ�������ã���������̬ʱ������ۻ��� 
   �����ƺͼ��ٶȼƲ�����̬û���ۻ�������̬��Ӧ�ϲ���������Ƶ�������Ի�����
   ���Բ��û����˲����ں������ִ����������ݣ���߲������Ⱥ�ϵͳ�Ķ�̬���ܡ�
   
 ****************************************************************************/

/*
 * @file attitude_estimator_q_main.cpp
 *
 * Attitude estimator (quaternion based)
 *
 * @author Anton Babushkin <anton.babushkin@me.com>
 */

#include <px4_config.h>
#include <px4_posix.h>
#include <px4_tasks.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <poll.h>
#include <fcntl.h>
#include <float.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <uORB/uORB.h>       //��Ϣ�����붩�Ļ��� 
#include <uORB/topics/sensor_combined.h>         // ��������Ϣ ���ܻ��� 
#include <uORB/topics/vehicle_attitude.h>
#include <uORB/topics/control_state.h>
#include <uORB/topics/vehicle_control_mode.h>
#include <uORB/topics/vehicle_global_position.h>
#include <uORB/topics/vision_position_estimate.h>
#include <uORB/topics/att_pos_mocap.h>
#include <uORB/topics/airspeed.h>
#include <uORB/topics/parameter_update.h>
#include <uORB/topics/estimator_status.h>
#include <drivers/drv_hrt.h>

#include <mathlib/mathlib.h>
#include <mathlib/math/filter/LowPassFilter2p.hpp>
#include <lib/geo/geo.h>

#include <systemlib/param/param.h>
#include <systemlib/perf_counter.h>
#include <systemlib/err.h>
#include <systemlib/mavlink_log.h>

//������Ϊc++�����أ�����cû�У�����c������Ҫ�����������ý�������
extern "C" __EXPORT int attitude_estimator_q_main(int argc, char *argv[]);
//�˴���extern ��C�� ��ʾ�� C ��ʽ���룻 __EXPORT ��ʾ ���������������������Linker����

// using �ؼ��� ��ʾ�������Ƶ� using ˵�����ֵ��������򡣡�
using math::Vector;     //���� 
using math::Matrix;     //���� 
using math::Quaternion; //��Ԫ�� 

class AttitudeEstimatorQ;
//���������ռ䣬ͨ�������ռ����instance;
namespace attitude_estimator_q
{
AttitudeEstimatorQ *instance;
}


class AttitudeEstimatorQ //�ඨ��;
{
public:
	/**
	 * Constructor
	 */
	AttitudeEstimatorQ();//���캯������ʼ������;

	/**
	 * Destructor, also kills task.
	 */
	~AttitudeEstimatorQ();//����������ɱ����������;

	/**
	 * Start task.
	 *
	 * @return		OK on success.
	 */
	 //��ʼ���񣬳ɹ�--����OK��
	int		start();// ������attitude_estimator_q�����̣���������ڣ� task_main_trampoline

	static void	task_main_trampoline(int argc, char *argv[]);
//��ת�� task_main() ��δʹ�ô��������static����ֻ�ܱ����ļ��еĺ������ã�
	void		task_main();//���������;

	void		print();//��ӡ��̬��Ϣ��

private:
	static constexpr float _dt_max = 0.02;//���ʱ����;
	 //constexpr(constant expression) �������ʽ��c11�¹ؼ��֣�
    //�Ż��﷨���ͱ����ٶȣ�
    
	bool		_task_should_exit = false;	 //���Ϊ true �������˳���	/**< if true, task should exit */
	int		_control_task = -1;	//����ID, Ĭ��-1��ʾû������	/**< task handle for task */
//���ĵĻ��� 
	int		_sensors_sub = -1;//sensor_combined subscribe(����);  ��������Ϣ 
	int		_params_sub = -1; //parameter_update subscribe;       ������Ϣ 
	int		_vision_sub = -1; //�Ӿ�λ�ù���;
	int		_mocap_sub = -1;  //vicon��̬λ�ù���;  ������׽ 
	int		_airspeed_sub = -1;//airspeed subscribe; ���� 
	int		_global_pos_sub = -1;//vehicle_global_position subscribe; ȫ��λ�� 
	
// �����Ļ���	
	orb_advert_t	_att_pub = nullptr;//vehicle_attitude publish(����);
	orb_advert_t	_ctrl_state_pub = nullptr;//  ��������״̬����control_state;
	orb_advert_t	_est_state_pub = nullptr; //  ��������״̬���� estimator_status

	struct {
		param_t	w_acc;//��������������ʾ 
		param_t	w_mag;
		param_t	w_ext_hdg;
		param_t	w_gyro_bias;
		param_t	mag_decl;
		param_t	mag_decl_auto;
		param_t	acc_comp;
		param_t	bias_max;
		param_t	ext_hdg_mode;
		param_t airspeed_mode;
	}		_params_handles; //���ò����ľ����	/**< handles for interesting parameters */

	float		_w_accel = 0.0f;   //���ٶȼ�Ȩ��
	float		_w_mag = 0.0f;     //������Ȩ��
	float		_w_ext_hdg = 0.0f; //�ⲿ����Ȩ��
	float		_w_gyro_bias = 0.0f;//������ƫ��Ȩ
	float		_mag_decl = 0.0f;	//��ƫ�ǣ��㣩
	bool		_mag_decl_auto = false;// ���û���GPS���Զ���ƫ��У��  ������ 
	bool		_acc_comp = false;	//���û���GPS�ٶȵļ��ٶȲ���	   ������ 
	float		_bias_max = 0.0f; 	//������ƫ������
	int		_ext_hdg_mode = 0;		//�ⲿ����ģʽ
	int 	_airspeed_mode = 0;		//����ģʽ 

	Vector<3>	_gyro; //������;
	Vector<3>	_accel;//������;
	Vector<3>	_mag;  //������;

	vision_position_estimate_s _vision = {};//�Ӿ� 
	Vector<3>	_vision_hdg;

	att_pos_mocap_s _mocap = {};//vicon��̬λ�ù���;  ������׽
	Vector<3>	_mocap_hdg;

	airspeed_s _airspeed = {};//���� 

	Quaternion	_q;//��Ԫ��;
	Vector<3>	_rates;//��̬���ٶ�;
	Vector<3>	_gyro_bias;//������ƫ��;

	vehicle_global_position_s _gpos = {};// ȫ��λ�� 
	Vector<3>	_vel_prev;//ǰһʱ�̵��ٶȣ�
	Vector<3>	_pos_acc; //���ٶȣ�body frame??��

	/* Low pass filter for accel��gyro */   //��̬ ���׵�ͨ�˲�����
	// �����Ǻͼ��ٶȼƵ� ��ͨ�˲�����  
	math::LowPassFilter2p _lp_accel_x;
	math::LowPassFilter2p _lp_accel_y;
	math::LowPassFilter2p _lp_accel_z;
	math::LowPassFilter2p _lp_gyro_x;
	math::LowPassFilter2p _lp_gyro_y;
	math::LowPassFilter2p _lp_gyro_z;
//����ʱ��(ms)
	hrt_abstime _vel_prev_t = 0;//ǰһʱ�̼����ٶ�ʱ�ľ���ʱ��;

	bool		_inited = false;//��ʼ����ʶ;
	bool		_data_good = false;//��ʼ����ʶ;
	bool		_ext_hdg_good = false;//�ⲿ�������;

	orb_advert_t	_mavlink_log_pub = nullptr;//mavlink log advert;

	void update_parameters(bool force);//���²���; 
//true: ��ȡ�²���, ���ɴ�ƫ�Ǹ�����Ԫ����  false: �鿴�����Ƿ���£�

	int update_subscriptions();//δʹ�á����飿����

	bool init();// �ɼ��ٶȼƺʹ����Ƴ�ʼ����ת������GPSʱ��У����ƫ�ǡ�

	bool update(float dt);//����init(); �����˲�

	// Update magnetic declination (in rads) immediately changing yaw rotation
	void update_mag_declination(float new_declination);//ʹ�ô�ƫ�Ǹ�����Ԫ��
};

//���캯������ʼ������;
AttitudeEstimatorQ::AttitudeEstimatorQ() :
	_vel_prev(0, 0, 0),//ǰһʱ�̵��ٶȣ�
	_pos_acc(0, 0, 0), //���ٶȣ�body frame??��
	_lp_accel_x(250.0f, 30.0f),// �����Ǻͼ��ٶȼƵ� ��ͨ�˲����� 
	_lp_accel_y(250.0f, 30.0f), //��ͨ�˲�������Ƶ��,��ֹƵ�ʣ�;
	_lp_accel_z(250.0f, 30.0f),
	_lp_gyro_x(250.0f, 30.0f),
	_lp_gyro_y(250.0f, 30.0f),
	_lp_gyro_z(250.0f, 30.0f)
{  ////�����Ƴ�ʱ;??? 
	_params_handles.w_acc		= param_find("ATT_W_ACC");   //��Щ����֮ǰ�ᵽ��ƥ��ϵͳ����
	_params_handles.w_mag		= param_find("ATT_W_MAG");
	_params_handles.w_ext_hdg	= param_find("ATT_W_EXT_HDG");
	_params_handles.w_gyro_bias	= param_find("ATT_W_GYRO_BIAS");
	_params_handles.mag_decl	= param_find("ATT_MAG_DECL");
	_params_handles.mag_decl_auto	= param_find("ATT_MAG_DECL_A");
	_params_handles.acc_comp	= param_find("ATT_ACC_COMP");
	_params_handles.bias_max	= param_find("ATT_BIAS_MAX");
	_params_handles.ext_hdg_mode	= param_find("ATT_EXT_HDG_M");
	_params_handles.airspeed_mode = param_find("FW_ARSP_MODE");
}

/**
 * Destructor, also kills task.  ����������ɱ����������;
 */
AttitudeEstimatorQ::~AttitudeEstimatorQ()
{
	if (_control_task != -1) {
		/* task wakes up every 100ms or so at the longest */
		_task_should_exit = true;

		/* wait for a second for the task to quit at our request */
		unsigned i = 0;

		do {
			/* wait 20ms */
			usleep(20000);

			/* if we have given up, kill it */
			if (++i > 50) {
				px4_task_delete(_control_task);
				break;
			}
		} while (_control_task != -1);
	}

	attitude_estimator_q::instance = nullptr;
}

int AttitudeEstimatorQ::start()//������attitude_estimator_q�����̣���������ڣ� task_main_trampoline
{
	ASSERT(_control_task == -1);

	/* start the task ��������  //�������񣬷��ؽ���ID; */
	_control_task = px4_task_spawn_cmd("attitude_estimator_q",//��������  ������ 
					   SCHED_DEFAULT,//Ĭ�ϵ���
					   SCHED_PRIORITY_MAX - 5,//���ȼ� 
					   2100, //ջ��С  �޸�  2000  2500
					   (px4_main_t)&AttitudeEstimatorQ::task_main_trampoline,//�߳���ں��� 
					   nullptr);//�������޲��� 

	if (_control_task < 0) {
		warn("task start failed");
		return -errno;
	}

	return OK;
}

//��ӡ��̬��Ϣ��
void AttitudeEstimatorQ::print()
{
    warnx("gyro status:");  // ���  
	_voter_gyro.print();
	warnx("accel status:");
	_voter_accel.print();
	warnx("mag status:");
	_voter_mag.print();
}

void AttitudeEstimatorQ::task_main_trampoline(int argc, char *argv[])
{
	attitude_estimator_q::instance->task_main();
}


/*�߳����������*/ 
void AttitudeEstimatorQ::task_main()
{

#ifdef __PX4_POSIX
//��¼�¼�ִ�������ѵ�ʱ�䣬performance counters;
	perf_counter_t _perf_accel(perf_alloc_once(PC_ELAPSED, "sim_accel_delay"));
	perf_counter_t _perf_mpu(perf_alloc_once(PC_ELAPSED, "sim_mpu_delay"));
	perf_counter_t _perf_mag(perf_alloc_once(PC_ELAPSED, "sim_mag_delay"));
#endif

	_sensors_sub = orb_subscribe(ORB_ID(sensor_combined));        // ���Ĵ�������Ϣ���� 
     //���Ĵ������������������ݲμ���Firmware/msg/sensor_combined.msg
	_vision_sub = orb_subscribe(ORB_ID(vision_position_estimate));// �����Ӿ� λ�ù��ƻ��� 
	_mocap_sub = orb_subscribe(ORB_ID(att_pos_mocap));            // vicon  ��̬λ���˶� ��׽���� 

	_airspeed_sub = orb_subscribe(ORB_ID(airspeed));              // ���ٻ��� 

	_params_sub = orb_subscribe(ORB_ID(parameter_update));        // �������»���  //bool saved;
	_global_pos_sub = orb_subscribe(ORB_ID(vehicle_global_position));//�ɻ�ȫ��λ�û���  λ�ù���ֵ(GPS);

	update_parameters(true);//�����Լ����ٽ�ȥ����  //��ȡ�²���;

	hrt_abstime last_time = 0;

	px4_pollfd_struct_t fds[1] = {};
	fds[0].fd = _sensors_sub;    //�ļ�������; �����ȴ�   sensor_combined ������Ϣ 
	fds[0].events = POLLIN;      //��ȡ�¼���ʶ;

	while (!_task_should_exit) {
		 /*poll��������������ʱ��Ϊ1s*/  //timeout = 1000; fds_size = 1; ���Linux��poll����;
		int ret = px4_poll(fds, 1, 1000);// ����ļ�������  struct pollfd�ṹ���͵�����  struct pollfd�ṹ���͵�����  ��poll��������������ʱ�䣬��λ������
		// ����ֵ��
    //>0������fds��׼���ö���д�����״̬����Щsocket����������������
    //==0:poll()����������timeout��ָ���ĺ���ʱ�䳤��֮�󷵻�;
    //-1:poll��������ʧ�ܣ�ͬʱ���Զ�����ȫ�ֱ���errno��

		if (ret < 0) {
			// Poll error, sleep and try again
			usleep(10000);
			PX4_WARN("Q POLL ERROR");
			continue;

		} else if (ret == 0) {
			// Poll timeout, do nothing
			PX4_WARN("Q POLL TIMEOUT");
			continue;
		}

		update_parameters(false);//���orb�Ƿ����;

		// Update sensors
		sensor_combined_s sensors;//���嶩�Ļ��� sensor_combined ��Ӧ����Ϣ���� �� ����  sensors

		if (!orb_copy(ORB_ID(sensor_combined), _sensors_sub, &sensors)) {//���������յ������ݿ�����  sensors�����ڴ��� 
			// Feed validator with recent sensor data

// ��ͨ�˲����� 
			if (sensors.timestamp > 0) {// ����������ʱ��� Ϊ ��Ϣ����� ʱ���  sensors.timestamp
				// �����������˲����� Filter gyro signal since it is not fildered in the drivers.
				_gyro(0) = _lp_gyro_x.apply(sensors.gyro_rad[0]);  // //�������ǵ�ֵ�Ȼ���Ȼ����΢�֣������ʵ������ƽ��  �������ȸ��ɿ�
				_gyro(1) = _lp_gyro_y.apply(sensors.gyro_rad[1]);
				_gyro(2) = _lp_gyro_z.apply(sensors.gyro_rad[2]);
			}
             // ���ٶ����ݵ�ʱ��� Ϊ  accelerometer_timestamp_relative + timestamp =  Accelerometer timestamp
			if (sensors.accelerometer_timestamp_relative != sensor_combined_s::RELATIVE_TIMESTAMP_INVALID) {
				// ���ٶ������˲�����  Filter accel signal since it is not fildered in the drivers.
				_accel(0) = _lp_accel_x.apply(sensors.accelerometer_m_s2[0]);
				_accel(1) = _lp_accel_y.apply(sensors.accelerometer_m_s2[1]);
				_accel(2) = _lp_accel_z.apply(sensors.accelerometer_m_s2[2]);

				if (_accel.length() < 0.01f) { //�˻����������ȣ��˴�Ϊ����ֵ��0��;
					PX4_DEBUG("WARNING: degenerate accel!"); //degenerate �� ���� �������������� ����Ӳ��������
					continue;
				}
			}
            // ��������Ϣ 
			if (sensors.magnetometer_timestamp_relative != sensor_combined_s::RELATIVE_TIMESTAMP_INVALID) {
				_mag(0) = sensors.magnetometer_ga[0];//δԤ���� 
				_mag(1) = sensors.magnetometer_ga[1];
				_mag(2) = sensors.magnetometer_ga[2];

				if (_mag.length() < 0.01f) { //�˻����������ȣ��˴�Ϊ����ֵ��0��;
					PX4_DEBUG("WARNING: degenerate mag!");//degenerate �� ���� �������������� ����Ӳ��������
					continue;
				}
			}

			_data_good = true;//���ݿ���;
		}

////////////////////// �����Ӿ� ����     ���  Update vision and motion capture heading
		bool vision_updated = false;
		orb_check(_vision_sub, &vision_updated);
        // ���� ����������   ��� 
		bool mocap_updated = false;
		orb_check(_mocap_sub, &mocap_updated);

        // �����Ӿ� ����
        //����Ӿ�����  �����Ӿ���Ϣ ���Ӿ�Ҳ��һ���ṹ�������кܶ���Ϣ ������x,y,x,vx,vy,vz,q[4] ���Ǹ����Ӿ�Ҳ���Ի�ȡλ����Ϣ
		if (vision_updated) {
			orb_copy(ORB_ID(vision_position_estimate), _vision_sub, &_vision);//���ƻ�����Ϣ�������ڴ� 
			math::Quaternion q(_vision.q);//��Ԫ�� 
                                                    // R-vision �����Ӿ� �õ���ת������
			math::Matrix<3, 3> Rvis = q.to_dcm();// ��Ԫ��Quaternion �任�� �������Ҿ���(Direction Cosine Matrix)Ҳ�� ��ת����(Rotation Matrix) 
			math::Vector<3> v(1.0f, 0.0f, 0.4f);
			
           //  http://www.07net01.com/2016/04/1472117.html ��������ά�ռ����ת
		    
			// Rvis is Rwr (robot respect to world) while v is respect to world.
			// Hence Rvis must be transposed having (Rwr)' * Vw
			// Rrw * Vw = vn. This way we have consistency
			 //ͨ���Ӿ��õ�����̬����q->Rvis����vת������������ϵ;
			_vision_hdg = Rvis.transposed() * v;//transposedת�� ��ΪR�Ǳ�׼������ ת��=��  ԭ���ɵ���->���壬ת�ú���� ����->����
			 //�²�Ӧ�������������Ĵ��� *��0 0 1�����޹���ôת�� ���õ���hdgӦ�þ����ô��ִ�����У׼�����
		}
		
        // ���� ����������   Update  motion capture heading����
        
		if (mocap_updated) {
			orb_copy(ORB_ID(att_pos_mocap), _mocap_sub, &_mocap);//���ƻ�����Ϣ�������ڴ�
			math::Quaternion q(_mocap.q);          //R-mocap ���ڶ�����׽ �õ���ת������
			math::Matrix<3, 3> Rmoc = q.to_dcm();// ��Ԫ��Quaternion �任�� �������Ҿ���(Direction Cosine Matrix)Ҳ�� ��ת����(Rotation Matrix) 

			math::Vector<3> v(1.0f, 0.0f, 0.4f);

			// Rmoc is Rwr (robot respect to world) while v is respect to world.
			// Hence Rmoc must be transposed having (Rwr)' * Vw
			// Rrw * Vw = vn. This way we have consistency
			_mocap_hdg = Rmoc.transposed() * v;  // Hdg����heading
		}

		// ���¿�����Ϣ  Update airspeed
		bool airspeed_updated = false;
		orb_check(_airspeed_sub, &airspeed_updated);

		if (airspeed_updated) {
			orb_copy(ORB_ID(airspeed), _airspeed_sub, &_airspeed);
		}

		// Check for timeouts on data
		if (_ext_hdg_mode == 1) {
			_ext_hdg_good = _vision.timestamp > 0 && (hrt_elapsed_time(&_vision.timestamp) < 500000);

		} else if (_ext_hdg_mode == 2) {
			_ext_hdg_good = _mocap.timestamp > 0 && (hrt_elapsed_time(&_mocap.timestamp) < 500000);
		}

//  //�ɻ�ȫ��λ������ 
/********GPSλ�ù���ֵУ����ƫ��**********/ 
		bool gpos_updated;//global position ȫ��λ��
		orb_check(_global_pos_sub, &gpos_updated);

		if (gpos_updated) { //global position ȫ��λ��
			orb_copy(ORB_ID(vehicle_global_position), _global_pos_sub, &_gpos);

			if (_mag_decl_auto && _gpos.eph < 20.0f && hrt_elapsed_time(&_gpos.timestamp) < 1000000) {
				/* set magnetic declination automatically  �Զ���ȡ��ƫ�� ��Ϊ��ͬ���� ��ͬλ�ô�ƫ�ǲ�ͬ ���ݾ��Ⱥ�ά�� �Զ���ȡ��ƫ�� */
				update_mag_declination(math::radians(get_mag_declination(_gpos.lat, _gpos.lon))); //latitudeγ�� longitude����
			}
			 //��ƫ�Զ�У������ˮƽƫ��ı�׼��С��20������λ�ù���ֵ(GPS)��vehicle_global_position��У����ƫ��;
            //get_mag_declination()�������õ��ش�ƫ�ǣ����в�����
		}
		
/********GPS�ٶȹ���ֵ������ٶ�ֵ**********/ 
		if (_acc_comp && _gpos.timestamp != 0 && hrt_absolute_time() < _gpos.timestamp + 20000 && _gpos.eph < 5.0f && _inited) {
		
			/* ʵ�ʵ�λ������ �� �� �� �� ����ϵ�� position data is actual */
		   //����GPS��λ���ٶ� ��Ϣ��΢�ֵõ����ٶ�ֵ;
			if (gpos_updated) {
				Vector<3> vel(_gpos.vel_n, _gpos.vel_e, _gpos.vel_d);// �� �� �� �� ����ϵ�� ���ٶ� �������������ٶ�

				/* velocity updated */
				if (_vel_prev_t != 0 && _gpos.timestamp != _vel_prev_t) {//�ٶȸ����� 
					float vel_dt = (_gpos.timestamp - _vel_prev_t) / 1000000.0f;//ʱ��������λ��s��
					
					/* ������������ϵ�ϵļ��ٶ� calculate acceleration in body frame */
					_pos_acc = _q.conjugate_inversed((vel - _vel_prev) / vel_dt);
					//��ned����ϵ�µ��ٶ������������ϵ�µļ��ٶ�;
				}

				_vel_prev_t = _gpos.timestamp;//����ʱ�� 
				_vel_prev = vel;              //�����ٶȱ���ֵ 
			}

		} else {
			/* position data is outdated, reset acceleration */
			_pos_acc.zero();
			_vel_prev.zero();
			_vel_prev_t = 0;
		}

		/* time from previous iteration */
		hrt_abstime now = hrt_absolute_time();
		//ʱ��������λ��s��
		float dt = (last_time > 0) ? ((now  - last_time) / 1000000.0f) : 0.00001f;//�ü�Сֵ0.00001��ʾ�㣬Ԥ���������;
		last_time = now;
		
        // ʱ���� �޷� 
		if (dt > _dt_max) {
			dt = _dt_max;//ʱ��������;
		}

/*���� update ������̬���µĺ������������Ӿ� mcap ���ٶ� ������ ���������ǣ�

��������Ԫ����΢�ַ��� ʵʱ���½�����̬��Ϣ�� �˺�������ǵõ����º����̬��Ϣ��  */

		if (!update(dt)) { //����update(dt)��**�����˲�**��������Ԫ��;
			continue;
		}

		Vector<3> euler = _q.to_euler();
		//�ø��µ���Ԫ����_q�����ŷ���ǣ��Ա��ڿ��ƹ�����ʵ�������Ŀ��ƣ����ƻ�����Ҫ��ֱ�����˵�ŷ���ǡ�

		struct vehicle_attitude_s att = {};// �ɻ���̬ ŷ����    ��
		// ע�� �� �汾vehicle_attitude_s��Ϣ���� ������  float32 rollspeed  float32 pitchspeed float32 yawspeed  float32[4] q
		att.timestamp = sensors.timestamp;

//##########################################################  wyw��� 
	//	att.roll = euler(0); //��ȡ��ŷ���Ǹ�ֵ��roll��pitch��yaw �Ƕ� 
	//	att.pitch = euler(1);
	//	att.yaw = euler(2);
//##################################################################

		att.rollspeed = _rates(0);  //��ȡroll��pitch��yaw����ת�ٶ�
		att.pitchspeed = _rates(1);
		att.yawspeed = _rates(2);

//################################################################## wyw��� 
	//	for (int i = 0; i < 3; i++) {
	//		att.g_comp[i] = _accel(i) - _pos_acc(i);////��ȡ��������ϵ���������ٶȣ�ǰ����ܹ�  ���ٶȲ���ֵ-�˶����ٶ�= �������ٶ�
	//	}
//##########################################################  

    //������ǶԸ��º����̬��Ϣ ��������д��ϵͳ ������һ�ε�ʹ��
		memcpy(&att.q[0], _q.data, sizeof(att.q));//��ʼ��Ϊ 0  Ϊ������������׼��  
		/* the instance count is not used here */
		int att_inst;
		orb_publish_auto(ORB_ID(vehicle_attitude), &_att_pub, &att, &att_inst, ORB_PRIO_HIGH);
		 //�㲥��̬��Ϣ;
		{ //ʹ�õ�ǰ��̬������control_state��������;
			struct control_state_s ctrl_state = {};  // ����״̬����  ��Ϣ���ͱ��� 

			ctrl_state.timestamp = sensors.timestamp;

			/* attitude quaternions for control state */
			ctrl_state.q[0] = _q(0); // ���� ״̬  ��Ԫ�� 
			ctrl_state.q[1] = _q(1);
			ctrl_state.q[2] = _q(2);
			ctrl_state.q[3] = _q(3);

			ctrl_state.x_acc = _accel(0); //���ٶ� 
			ctrl_state.y_acc = _accel(1);
			ctrl_state.z_acc = _accel(2);

			/* attitude rates for control state */
			ctrl_state.roll_rate = _rates(0); // ���ٶ� 
			ctrl_state.pitch_rate = _rates(1);
			ctrl_state.yaw_rate = _rates(2);

			/* TODO get bias estimates from estimator */
			ctrl_state.roll_rate_bias = 0.0f;   //����ƫ��  
			ctrl_state.pitch_rate_bias = 0.0f;
			ctrl_state.yaw_rate_bias = 0.0f;

			ctrl_state.airspeed_valid = false;

			if (_airspeed_mode == control_state_s::AIRSPD_MODE_MEAS) {
				// use measured airspeed
				if (PX4_ISFINITE(_airspeed.indicated_airspeed_m_s) && hrt_absolute_time() - _airspeed.timestamp < 1e6
				    && _airspeed.timestamp > 0) {
					ctrl_state.airspeed = _airspeed.indicated_airspeed_m_s;
					ctrl_state.airspeed_valid = true;
				}
			}

			else if (_airspeed_mode == control_state_s::AIRSPD_MODE_EST) {
				// use estimated body velocity as airspeed estimate
				if (hrt_absolute_time() - _gpos.timestamp < 1e6) {
					ctrl_state.airspeed = sqrtf(_gpos.vel_n * _gpos.vel_n + _gpos.vel_e * _gpos.vel_e + _gpos.vel_d * _gpos.vel_d);
					ctrl_state.airspeed_valid = true;
				}

			} else if (_airspeed_mode == control_state_s::AIRSPD_MODE_DISABLED) {
				// do nothing, airspeed has been declared as non-valid above, controllers
				// will handle this assuming always trim airspeed
			}

			/* the instance count is not used here */
			int ctrl_inst;
			
			/* publish to control state topic */
			orb_publish_auto(ORB_ID(control_state), &_ctrl_state_pub, &ctrl_state, &ctrl_inst, ORB_PRIO_HIGH);
		}

		{
			//struct estimator_status_s est = {};

			//est.timestamp = sensors.timestamp;

			/* the instance count is not used here */
			//int est_inst;
			/* publish to control state topic */
			// TODO handle attitude states in position estimators instead so we can publish all data at once
			// or we need to enable more thatn just one estimator_status topic
			// orb_publish_auto(ORB_ID(estimator_status), &_est_state_pub, &est, &est_inst, ORB_PRIO_HIGH);
		}
	}

#ifdef __PX4_POSIX
	perf_end(_perf_accel);
	perf_end(_perf_mpu);
	perf_end(_perf_mag);
#endif

	orb_unsubscribe(_sensors_sub);
	orb_unsubscribe(_vision_sub);
	orb_unsubscribe(_mocap_sub);
	orb_unsubscribe(_airspeed_sub);
	orb_unsubscribe(_params_sub);
	orb_unsubscribe(_global_pos_sub);
}

void AttitudeEstimatorQ::update_parameters(bool force)
{
	bool updated = force;

	if (!updated) {
		orb_check(_params_sub, &updated);
	}

	if (updated) {
		parameter_update_s param_update;
		orb_copy(ORB_ID(parameter_update), _params_sub, &param_update);

		param_get(_params_handles.w_acc, &_w_accel);
		param_get(_params_handles.w_mag, &_w_mag);
		param_get(_params_handles.w_ext_hdg, &_w_ext_hdg);
		param_get(_params_handles.w_gyro_bias, &_w_gyro_bias);
		float mag_decl_deg = 0.0f;
		param_get(_params_handles.mag_decl, &mag_decl_deg);
		update_mag_declination(math::radians(mag_decl_deg));
		int32_t mag_decl_auto_int;
		param_get(_params_handles.mag_decl_auto, &mag_decl_auto_int);
		_mag_decl_auto = mag_decl_auto_int != 0;
		int32_t acc_comp_int;
		param_get(_params_handles.acc_comp, &acc_comp_int);
		_acc_comp = acc_comp_int != 0;
		param_get(_params_handles.bias_max, &_bias_max);
		param_get(_params_handles.ext_hdg_mode, &_ext_hdg_mode);
		param_get(_params_handles.airspeed_mode, &_airspeed_mode);
	}
}
//����һ����ʼ����Ԫ��������Ԫ������ʾ��̬ʱ ͨ��΢�ַ���ʵʱ���½�����̬ �����������Ҳ����Ҫһ����ֵ����
//�����������в���Ҫ��������ٶȵĲ���ֵ��Ϊ������ ˵�������Ԫ������ ����ת������������ǵ����л��ǵ�֮ǰ��[0 0 1]g
//�õ��ĵ������������в�����ת����
//i= ��_mag - k * (_mag * k))֮������ô���� ֻ����ǿ��ʹk i���Ϊ�㣬�໥��ֱ��/
// k= -_accel Ȼ���һ��k��kΪ���ٶȴ��������������ٶȷ������������ڵ�һ�β�������ʱ���˻�һ��Ϊƽ��״̬���˶�״̬��
//���Կ���ֱ�ӽ��⵽�ļ��ٶ���Ϊ�������ٶ�g���Դ���Ϊdcm��ת����ĵ�����k��������ܹ��ˣ���
// i= ��_mag - k * (_mag * k)) _mag����ָ����������k*(_mag*k) ����correctionֵ���������յ���Ԫ����һ���Ժ�ķ�������������5%���ڣ�
//�о����硶DCM IMU:Theory������������ۡ�ǿ���������������ĺã�Renormalization�㷨��ardupilot���ϲ�Ӧ��AP_AHRS_DCM��ʹ�õ��ˡ�
// j= k%i ���������ˡ����������Vector<3>k = -_accel��Vector<3>�൱��һ�����ͣ�int������һ������k��Ȼ���-_accel��ֵ��k��
//�ڶ���_accelʱҲ��ʹ��Vector<3>������ͬһ�����͵ģ���Ҫ����Ϊ�˿�����ʵ�������̣����ƺ������أ�


// �����˲��õ�  ���ٶ� roll  pitch yaw 
// �ɼ��ٶȼƺʹ����Ƴ�ʼ����ת�������Ԫ��; 
bool AttitudeEstimatorQ::init()
{
	// Rotation matrix can be easily constructed from acceleration and mag field vectors
	// 'k' is Earth Z axis (Down) unit vector in body frame
	// k Ϊ��������ϵ��NED���� z �ᣨD���ڻ�������ϵ�еı�ʾ;  ����ϵ�У�D��������;
	Vector<3> k = -_accel;
	k.normalize();

	// 'i' is Earth X axis (North) unit vector in body frame, orthogonal with 'k'
	// i Ϊ��������ϵ��NED���� x �ᣨN���ڻ�������ϵ; 
	Vector<3> i = (_mag - k * (_mag * k));//ʩ����������;
	i.normalize();

	// 'j' is Earth Y axis (East) unit vector in body frame, orthogonal with 'k' and 'i'
	Vector<3> j = k % i;//j Ϊ��������ϵ��NED���� y �ᣨE���ڻ�������ϵ; 

	// Fill rotation matrix  // ������ת���� R 
	Matrix<3, 3> R;
	R.set_row(0, i);
	R.set_row(1, j);
	R.set_row(2, k);

	// ��ת���� ����Ԫ��  Convert to quaternion
	_q.from_dcm(R);

	// Compensate for magnetic declination  // ת��Ϊ��Ԫ�� q �������趨У����ƫ����һ��;
	Quaternion decl_rotation;
	decl_rotation.from_yaw(_mag_decl);
	_q = decl_rotation * _q;

	_q.normalize();

	if (PX4_ISFINITE(_q(0)) && PX4_ISFINITE(_q(1)) &&
	    PX4_ISFINITE(_q(2)) && PX4_ISFINITE(_q(3)) &&
	    _q.length() > 0.95f && _q.length() < 1.05f) {
		_inited = true;

	} else {
		_inited = false;
	}

	return _inited;
}

// �ɼ��ٶȼƺʹ����Ƴ�ʼ����ת������GPSʱ��У����ƫ�ǡ�  �����˲� 
bool AttitudeEstimatorQ::update(float dt) //���ڶ���Ԫ������_q���г�ʼ����ֵ ���� ���¡�
{
	if (!_inited) {// �����ж��Ƿ��ǵ�һ�ν���ú�������һ�ν���ú����Ƚ���init������ʼ��

		if (!_data_good) {
			return false;
		}

		return init();  //ִ��һ��;
	}

//������ǵ�һ�ν���ú��������ж���ʹ��ʲômode�������ģ�����vision��mocap��acc��mag
//��DJI����4�õ��Ӿ�����Ӧ�þ������vision����Hdg����heading��


    //������һ״̬���Ա�ָ�;
	Quaternion q_last = _q; //����_q����init()��ʼ���õ����Ǹ� ���� ����->��������ϵ

	// Angular rate of correction  У׼
	//corr�������������������ٶȼ���������vision��mocap��������gyro�в������Ľ��ٶ�ƫת����
    //����ΪcorrΪupdate�����ж���ı���������ÿ�ν���update������ʱ��ˢ��corr���������ݣ� _rateҲ��ˢ�����е����ݣ�����Ϊ������̬�ǵĽ��ٶȣ������󣩣�
    //_qΪ�ⲿ����ı����������������ֻ��+=�������¸�ֵ�����������ִ���᷵����һ�μ������_q��
    
	Vector<3> corr;//��ʼ��Ԫ��Ϊ0;  У׼���� 
	float spinRate = _gyro.length();

	if (_ext_hdg_mode > 0 && _ext_hdg_good) {//_ext_hdg_good��ʾ�ⲿ�������ݿ���;
		
		 //  _ext_hdg_mode== 1��2ʱ��������vision���ݺ�mocap���ݶ�gyro���ݽ��������������global frame������ν��earthframe��
		
		if (_ext_hdg_mode == 1) {
			// Vision heading correction
			//�Ӿ�����У��;
			// Project heading to global frame and extract XY component
			//������ͶӰ����������ϵ����ȡXY����;
			Vector<3> vision_hdg_earth = _q.conjugate(_vision_hdg); 
			//_q.conjugate��bϵ��nϵ����conjugate_inversednϵ��bϵ���ٲ��ص�bϵ
			float vision_hdg_err = _wrap_pi(atan2f(vision_hdg_earth(1), vision_hdg_earth(0)));
			// Project correction to body frame
			corr += _q.conjugate_inversed(Vector<3>(0.0f, 0.0f, -vision_hdg_err)) * _w_ext_hdg;
		}

		if (_ext_hdg_mode == 2) {
			// Mocap heading correction
			// �˶���׽  У�� 
			// Project heading to global frame and extract XY component
			Vector<3> mocap_hdg_earth = _q.conjugate(_mocap_hdg);
			float mocap_hdg_err = _wrap_pi(atan2f(mocap_hdg_earth(1), mocap_hdg_earth(0)));
			// Project correction to body frame
			corr += _q.conjugate_inversed(Vector<3>(0.0f, 0.0f, -mocap_hdg_err)) * _w_ext_hdg;
			 //����corrֵ���ڵ�λ������ת����R��bϵתnϵ����ת�ã��������Ϊ R��nϵתbϵ�������ԣ�0,0��-mag_err����
			 //�൱�ڻ�������ϵ�Ƶ�������ϵN�ᣨZ�ᣩת��arctan��mag_earth(1), mag_earth(0)���ȡ�
		}
	}

	if (_ext_hdg_mode == 0  || !_ext_hdg_good) {
		
		//_ext_hdg_mode== 0���ô����������� Magnetometer������ correction
		
		// Magnetometer correction
		// Project mag field vector to global frame and extract XY component
		// ��ʹ���ⲿ���򣬻��ⲿ�����쳣ʱ�����ô�����У׼; 
		Vector<3> mag_earth = _q.conjugate(_mag); //bϵ��nϵ  �������ƶ����ӻ�������ϵת������������ϵ;  Rn-b * _mag 
		float mag_err = _wrap_pi(atan2f(mag_earth(1), mag_earth(0)) - _mag_decl);
		//�������ƵĶ������������㵽�Ƕ�; �ýǶȼ�ȥ��ƫ���õ��ų�ƫ��; 
		//_mag_decl ��GPS�õ�; 
		//**atan2f: 2 ��ʾ�����������; ֧���������ڽǶȻ���; �������ֵ; 
		//**_wrap_pi: ��(0~2pi)�ĽǶ�ӳ�䵽(-pi~pi);
		// ֻ����Vector<3> mag_earth�е�ǰ��ά������mag_earth(1)��mag_earth(0)����x��y������z���ϵ�ƫ�ƣ���
		// ͨ��arctan�õ��ĽǶȺ�ǰ����ݾ�γ�Ȼ�ȡ�Ĵ�ƫ������ֵ�õ���ƫ���Ƕ�mag_err ��_wrap_pi�����������޶����-pi��pi�ĺ���
		float gainMult = 1.0f;
		const float fifty_dps = 0.873f;

		if (spinRate > fifty_dps) {
			gainMult = fmin(spinRate / fifty_dps, 10.0f);
		}

		// Project magnetometer correction to body frame
		// ���ų�ƫ��ת������������ϵ�����������ӦȨ��  mag_err  GPS ���; 
		corr += _q.conjugate_inversed(Vector<3>(0.0f, 0.0f, -mag_err)) * _w_mag * gainMult;    ////nϵ��bϵ
	}

	_q.normalize();//��Ԫ����һ��;   ����Ĺ�һ������У��update_mag_declination���ƫ�


	// // ���ٶ����������涼��������������� z�ᣬ���ǻ���x y��˭������ Accelerometer correction
	// Project 'k' unit vector of earth frame to body frame
	// Vector<3> k = _q.conjugate_inversed(Vector<3>(0.0f, 0.0f, 1.0f));
	// Optimized version with dropped zeros
	// �ѹ�һ����nϵ�������ٶ�ͨ����ת����R�����ת��bϵ  Vector<3> k = _q.conjugate_inversed(Vector<3>(0.0f, 0.0f, 1.0f)); R(n->b) ���ԣ�0,0��1��g
	Vector<3> k(
		2.0f * (_q(1) * _q(3) - _q(0) * _q(2)),
		2.0f * (_q(2) * _q(3) + _q(0) * _q(1)),
		(_q(0) * _q(0) - _q(1) * _q(1) - _q(2) * _q(2) + _q(3) * _q(3))
	);

	corr += (k % (_accel - _pos_acc).normalized()) * _w_accel;//w��ͷ����Ȩ��
	//���� k ���������ٶ�����(0,0,1)ת������������ϵ; 
	//_accel �Ǽ��ٶȼƵĶ���; 
	//_pos_acc ����λ�ù���(GPS) ΢�ֵõ��ļ��ٶ�; 
	//_accel - _pos_acc ��ʾ���������ٶ�������ȥ��ˮƽ����; 
	//k �� (_accel - _pos_acc)��ˣ��õ�ƫ��;    e=k��[��������������ٶ�]
	 
//�����mahony�㷨�еĵļ�����̣�ֻ�ǻ��˸���װ {k%��_accel�����ٶȼƵĲ���ֵ��-λ�Ƽ��ٶȣ��ĵ�λ����<Լ�����������ٶ�g>}*Ȩ�ء� 
//���￼�����˶����ٶȣ���ȥ�����ɿ���֮ǰ�������ˡ������ٶȼƲ������� �������ٶ�+�˶����ٶȣ�
//�ܼ��ٶȣ����ٶȻ�ȡ����ȥ�����˶����ٶȣ����岿�֣���ȡ�������ٶȣ�Ȼ����̬����Ĳ����о��������봿�������ٶ���������������
//��Ϊ�˶����ٶ����к��ĸ��ţ�����������㷨�����ۻ�����[0,0,1]����̬������ˡ��ò�ֵ��ȡ���������ٶȵķ����ǵ�������ϵ�µ�z�ᣬ
//�����˶����ٶ�֮���ܼ��ٶȵķ���Ͳ����뵼������ϵ������ƽ���ˣ�����Ҫ�������������_accel-_pos_acc����
//Ȼ����z�������õ�������У׼



	// // ������������ Gyro bias estimation 
	if (spinRate < 0.175f) {
		_gyro_bias += corr * (_w_gyro_bias * dt);  // gyro_bias+=[Mag*wmag+Acc*wacc]*w_gyro*dt
		//PI�������е�I�����֣�Ч����Ȼ���_gyro_bias��Լ������ _w��ͷ����weigh����Ȩ�� ����������KI��Ч��
		
		// �õ�����������ֵ 
		// �˴�����ֵΪ mahony �˲��� pi �������� ���ֲ���; 
		// ��Ϊ _gyro_bias �����㣬�ʲ����ۻ�; 

		for (int i = 0; i < 3; i++) {
			_gyro_bias(i) = math::constrain(_gyro_bias(i), -_bias_max, _bias_max);
		}

	}

	_rates = _gyro + _gyro_bias;//���ٶ� = �����ǵĲ���ֵ + ���У׼

	// Feed forward gyro
	corr += _rates; // ǰ�� Feed forward gyro  ������Ϊ����� corr ���Ǹ��º�Ľ��ٶ�;

	// Apply correction to state
	//������ʹ�����������ݸ�����Ԫ��������_rates��_gyro_bias��������´ε���ʱʹ�á�
    // ��״̬����������ʵ���� ���ǹؼ�����̬���� Apply correction to state
	_q += _q.derivative(corr) * dt;//������Ԫ����derivativeΪ�󵼺�����ʹ��һ����������󵼡� 
	
    // �ǳ���Ҫ�����õ���΢�ַ�����ɢ����˼�롣��ǰ����DCM������¹�����Ҳ���õ��˸�˼�롣
     // �ȿ������룬�е�֣��־͹���derivative���������������ϣ�ƽʱһ���Ƶ����ĺ��ڿ����涼���õ�omga *Q ����ʽ��
   //������Ĵ���ʵ��ȷ���õ�Q * omga����ʽ�����Թ����4*4�����ÿһ�еķ��žͲ�һ���ˡ�
    //http://blog.csdn.NET/qq_21842557/article/details/51058206���


	// Normalize quaternion  ���� 
	_q.normalize();

	if (!(PX4_ISFINITE(_q(0)) && PX4_ISFINITE(_q(1)) &&
	      PX4_ISFINITE(_q(2)) && PX4_ISFINITE(_q(3)))) {
		  // Reset quaternion to last good state
		  _q = q_last;  //����_rates��_gyro_bias��������´ε���ʱʹ�á�
		  _rates.zero();
		  _gyro_bias.zero();
		  return false;
	}

	return true;
}


// ʹ�ô�ƫ�Ǹ�����Ԫ��
void AttitudeEstimatorQ::update_mag_declination(float new_declination)
{
	// Apply initial declination or trivial rotations without changing estimation
	if (!_inited || fabsf(new_declination - _mag_decl) < 0.0001f) { //����΢С��ת; 
		_mag_decl = new_declination;

	} else {
		// Immediately rotate current estimation to avoid gyro bias growth
		Quaternion decl_rotation;////��ƫ����һ��ֵ��У����̬;
		decl_rotation.from_yaw(new_declination - _mag_decl);
		_q = decl_rotation * _q;  //�ɴ�ƫ�Ƕ�ת��Ϊ��Ԫ��; 
		_mag_decl = new_declination;//   //��Ԫ����˱�ʾ�Ƕ����;
	}
}



int attitude_estimator_q_main(int argc, char *argv[])
{
	// �����в������� 
	//�ⲿ���ýӿ�;
	if (argc < 2) {
		warnx("usage: attitude_estimator_q {start|stop|status}");
		return 1;
	}

	if (!strcmp(argv[1], "start")) {//���� 

		if (attitude_estimator_q::instance != nullptr) {
			warnx("already running");
			return 1;
		}
		
        //ʵ������instance;
		attitude_estimator_q::instance = new AttitudeEstimatorQ;

		if (attitude_estimator_q::instance == nullptr) {
			warnx("alloc failed");
			return 1;
		}

		if (OK != attitude_estimator_q::instance->start()) {
			delete attitude_estimator_q::instance;
			attitude_estimator_q::instance = nullptr;
			warnx("start failed");
			return 1;
		}

		return 0;
	}

	if (!strcmp(argv[1], "stop")) {//ֹͣ  
		if (attitude_estimator_q::instance == nullptr) {
			warnx("not running");
			return 1;
		}

		delete attitude_estimator_q::instance;//ɾ��ʵ��������ָ���ÿ�;
		attitude_estimator_q::instance = nullptr;
		return 0;
	}

	if (!strcmp(argv[1], "status")) { //��ӡ��ǰ��̬��Ϣ;
		if (attitude_estimator_q::instance) {
			attitude_estimator_q::instance->print();
			warnx("running");
			return 0;

		} else {
			warnx("not running");
			return 1;
		}
	}

	warnx("unrecognized command");
	return 1;
}


/*
 ���˵�������⣺��ֵ���Ĵ���������ǵ�Ʈ�ݼ�PI�������������ǣ�

       ��׼ȷ�������źŽ���׼ȷ�ķ��Ż��ֽ���õ�׼ȷ�ġ���ȷ����ת���󡣼�ʹ����׼ȷ�������źţ���ֵ�����Ի�����������ֵ��
1. ��������ֵ���ֲ�������ʱ�䲽���;������޲����ʵ����ݡ�������ʹ�õ���ֵ���ַ������Բ������������ض��ļ��衣
������ʹ�õķ���������ÿ��ʱ�䲽������ת�ٶȺ㶨���䡣�⽫������������ת���ٶȵ���
2. ����������ʹ�����ַ�����ʾ��ֵ����Щ��ﶼ�����޵ģ����Ի������������ģ��ת����ʼ��
��ִ���κ��޷���������������λ���ļ��㣬�������������ۻ���
��ת�����һ���������������������ԣ����������������ĳ���ο�ϵ���Ǵ�ֱ�ģ���ô����������Ĳο�ϵ�ж��Ǵ�ֱ�ġ�
ͬ������ͬ�ο�ϵ�µ�ͬһ����������ͬ��Ȼ����ֵ�����ƻ���һ���������磬��ת��������ж��ǵ�λ���������ǵĳ���Ӧ�õ���1��
������ֵ���ᵼ�����ǵĳ��ȱ�С���߱�󡣲����ۻ���ȥ�����ǵĳ��Ƚ����Ϊ0������������⣬��ת���������Ӧ�����໥��ֱ�ģ�
��ֵ���ᵼ�������໥��б������ͼ��ʾ��

�ع淶��
������ֵ���Ĵ��ڣ��������Ҿ��������������ԣ���ʽ(5)�Ҷ˲����ϸ���ڵ�λ������ʵ�ϣ���ʱ����ϵ��������һ�����塣
���˵��ǣ���ֵ�����۵طǳ����������Լ�ʱ��������һ�����ѵ����顣
���ǰ�ʹ�������Ҿ������������ԵĲ�����Ϊ�ع淶�����кܶ��ַ�������ʵ���ع淶����������������ʾ���ǵ�Ч�����ȽϺã�
���������򵥵�һ�ַ����������������£����ȼ��㷽�����Ҿ���X����Y����ڻ���
��������ϸ���������ô������Ӧ����0������������ʵ���Ϸ�ӳ��X����Y���໥��ת�����ĳ̶ȡ�


�����ǵ�Ư�����⣬PID����������·�����ݲ�����

�����˲������ڶ�ʱ���ڲ��������ǵõ��ĽǶ���Ϊ����ֵ����ʱ�Լ��ٶȲ������ļ��ٶ�ֵ����ȡƽ��ֵ��У�������ǵĵõ��ĽǶȡ�
��ʱ�����������ǱȽ�׼ȷ������Ϊ������ʱ���ü��ٶȼƱȽ�׼ȷ����ʱ��Ӵ����ı��أ�����ǻ����ˣ��������ٶȼ�Ҫ�˵���Ƶ�źţ�
������Ҫ�˵���Ƶ�źţ������˲������Ǹ��ݴ��������Բ�ͬ��ͨ����ͬ���˲�������ͨ���ͨ�������ģ���Ȼ������ӵõ�����Ƶ�����źš�
���磬���ٶȼƲ���ٶ�ֵ���䶯̬��Ӧ�������ڸ�Ƶʱ�źŲ����ã����Կ�ͨ����ͨ�˲������Ƹ�Ƶ���ţ���������Ӧ�죬���ֺ�ɲ���ǣ�
�����������Ư�Ƶȣ��ڵ�Ƶ���źŲ��ã�ͨ����ͨ�˲��������Ƶ�Ƶ���š������߽�ϣ��ͽ������Ǻͼ��ٶȼƵ��ŵ��ں�������
�õ��ڸ�Ƶ�͵�Ƶ���Ϻõ��źţ������˲���Ҫѡ���л���Ƶ�ʵ㣬����ͨ�͵�ͨ��Ƶ�ʡ�

    ��Ȼ���ݱ����൱�ã���ָ������ÿ�������С�������ն������ǵ�Ư�����ǲ��ò���һЩ���顣
	��Ҫ���ľ���������������Ĳο���̽������ƫ�ƣ��ṩһ�������PID����������·�����ݲ�������ͼ1��ʾ���������£�

    1. ʹ�÷���ο�����̽�ⶨ����ͨ������һ����תʸ����������ֵ�ͼ���ֵ�Ĳο�ʸ��������

    2. ͨ��һ���������֣� PI�� ������������������������תУ׼�ٶȣ�����תʸ���������� PI�������ǳ��õ�
PID������������һ��������D����΢�֡������ǵ���������У����ǲ���Ҫ�õ�΢�����

    3. ���ϣ� ���߼�ȥ���������������ת���ķ���Լ���� �������ֿ������������ʵ�ʵ��������źš�

    �Է���ο���������ҪҪ���ǣ�����Ư�ơ���˲̬���ܲ�����ô��Ҫ����Ϊ���ݶԷ��������˲̬�����ԡ�
GPS�ͼ��ٶȼ�Ϊ�����ṩ�������ο������ṩ��������Ҳ�Ƿǳ����õģ��ر��ƫ���Ŀ��ƣ����ǶԷ��������е�
ָ�򣬽�һ��ȫ��λϵͳ�����úܺá������ʹ�ô����ƣ���Ӧ��ʹ��һ������������ṩһ��ʸ���ο��� �ͳ�
��������������г��Ͽ��Ժܷ�����򵽡�

    ���������ο������е���һ����ͨ����õ������Ľ���˻���ⷽ���������õ�ʸ���÷������Ҿ�����ơ�
	�������ر�������������ɡ����Ĵ�С������ʸ���н������ҳ����ȣ�ͬʱ���ķ���ֱ������ʸ����
����������һ����ת�����ת������Ҫ��ת���������ʸ��ʹ�������������Ƶ�ʸ��ƽ�С�
    ���仰˵�������ڷ�����ת���ĸ�ֵ��ͨ���������ֿ��������������������ǣ���λ�����𽥱��ȸ��ٲο�������
	�������ݵ�Ư�ƾͱ������ˡ������ӷ������Ҿ���������Ӧ�Ĳο�ʸ���Ĳ����һ�������ָʾ������Լ���������ת��
	�����ò�������ο�ʸ�����������ʸ��У׼�����Ǹ���Ȥ������תУ�����������������Ҫ���ڷ������Ҿ����У�
	��������ת���ĸ�ֵ��ͨ�����������������Ժܷ���ļ���������������ת����ͨ���ο�ʸ���ķ������ҵ�ʸ�����ƵĲ����
	�����ñ������ֿ�����������������ת����Ϊ���ȶ���ͬʱ�����������ȫ�����������ǵ�Ư�ƣ�������Ư�ƣ�����෽����
	�ο�ʸ�������ͨ���������Ҿ���ķ�ʽӳ�䵽���ݵġ�����������ӳ��ȡ���ڹ��Բ�����Ԫ��
	���磬GPS�ο�ʸ�����ܽ���X��Y��Z����X��Y��Z�������źţ�ȡ���ڵ�������ϵ��ķ���

*/




