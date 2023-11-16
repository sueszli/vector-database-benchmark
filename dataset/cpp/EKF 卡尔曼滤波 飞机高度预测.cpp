/****************************************************************************
 *
 ���� �������˲� �ĵ��ι���    ʹ�� ������̬  ��ͨ����������  �Լ����봫�������� 
 Ԥ����� + ��������  + У������  Ԥ��  �ɻ������߶�   z���ٶ�     z����ٶ�ƫ��(����) 
 ��̬���� ����  ��������       GPS���� У��Z���ٶ� 
\Firmware\src\lib\terrain_estimation
 ****************************************************************************/

/**
 * @file terrain_estimator.cpp
 * A terrain estimation kalman filter.
 */

#include "terrain_estimator.h"
#include <geo/geo.h>

#define DISTANCE_TIMEOUT 100000		// ΢�� time in usec after which laser is considered dead

TerrainEstimator::TerrainEstimator() :
	_distance_last(0.0f),
	_terrain_valid(false),
	_time_last_distance(0),
	_time_last_gps(0)
{
	/*
	// "terrain_estimator.h" �еı����������� 
	 // �������˲��� ��ر��� 
	matrix::Vector<float, n_x> _x;  // Ԥ��״̬����  1*3  ����߶�  �ٶ� z�� ���ٶ�ƫ�� state: ground distance, velocity, accel bias in z direction
	float  _u_z;		        	// acceleration in earth z direction
	matrix::Matrix<float, 3, 3> _P;	   // ���ڹ��ƿ��������� �� �� ���� Ԥ���Э������� covariance matrix  3*3 һ��Ϊ ���������� ���� 
	*/
	
	memset(&_x._data[0], 0, sizeof(_x._data));//��ʼ������Ϊ0 ��ʼ״̬ Ϊ0  
	_u_z = 0.0f;                              //Z�� ���ٶ� 
	_P.setIdentity();                         // Э��������ʼ��Ϊ ��λ�� 
	//[1 0 0]
	//[0 1 0]
	//[0 0 1]
}

// �жϾ��������� �Ƿ���Ч 
bool TerrainEstimator::is_distance_valid(float distance)
{
	if (distance > 40.0f || distance < 0.00001f) {
		return false;

	} else {
		return true;
	}
}

// �������˲� Ԥ�ⲿ��          ʱ��       ��̬   ��ͨ����������   ���봫��������(�״� ���� ����)
void TerrainEstimator::predict(float dt, const struct vehicle_attitude_s *attitude,
			       const struct sensor_combined_s *sensor,
			       const struct distance_sensor_s *distance)
{
	//��ȡ��ǰ��̬��ת����   ��Ԫ��-> ��ת���� 
	matrix::Dcmf R_att = matrix::Quatf(attitude->q);
	matrix::Vector<float, 3> a(&sensor->accelerometer_m_s2[0]);//���ٶȼƶ���(��������ϵ) 
	matrix::Vector<float, 3> u;                                //���ٶȼƶ���(��������ϵ)
	
	u = R_att * a;//�����ٶȼƶ���ͶӰ����������ϵ��
	//�õ�Z��� ���ٶȼ�
	_u_z = u(2) + CONSTANTS_ONE_G;//�����������ٶ� 9.80665f  m/s^2 compensate for gravity

	// dynamics matrix   Ԥ��ת�ƾ��� 
	matrix::Matrix<float, n_x, n_x> A;
	A.setZero();//[0 1 0]
	A(0, 1) = 1;//[0 0 1]
	A(1, 2) = 1;//[0 0 0]

	// input matrix     �������� 
	matrix::Matrix<float, n_x, 1>  B;
	B.setZero(); //[0]
	B(1, 0) = 1; //[1]  
				 //[0]
				 
	// input noise variance  �������� ���� 
	 // ������Ϊ�˴� R Ϊ���ٶȼƲ���
	float R = 0.135f;  

	// process noise convariance  Ԥ��ת�ƹ��� ��˹���� 
	//�����QΪ���󣬺���û���ã����ñ��Ϸ���R���棻
	matrix::Matrix<float, n_x, n_x>  Q;
	Q(0, 0) = 0; //[0 0 0] 
	Q(1, 1) = 0; //[0 0 0]
				 //[0 0 0]
				 
	// do prediction Ԥ�����   ת�ƹ��� + �������� 
	matrix::Vector<float, n_x>  dx = (A * _x) * dt;//״̬�仯 ����   QΪ0 ��û�м���Ԥ��������� 
	dx(1) += B(1, 0) * _u_z * dt;// z�� ��ֱ�ٶ�  �ļ��ٶ� ����   �������� 
    // dx = [�ٶ�*ʱ�� �� v*dt]          = ����仯�� 
	//		[(z����ٶ�ƫ��+���ٶ�)*dt]  = �ٶȱ仯�� 
	//		[0]                          = z����ٶ�ƫ��仯�� 
	// propagate state and covariance matrix
	_x += dx;//״̬�ۼ�  Ԥ��� ״̬ 
	_P += (A * _P + _P * A.transpose() + //Ԥ���Э���    ת�ƹ���  + �������� 
	       B * R * B.transpose() + Q) * dt;
}

// У�� 
void TerrainEstimator::measurement_update(uint64_t time_ref, const struct vehicle_gps_position_s *gps,
		const struct distance_sensor_s *distance,
		const struct vehicle_attitude_s *attitude)
{
	// terrain estimate is invalid if we have range sensor timeout
	// ��ʱ��  ����Ԥ�ⲻ׼ȷ 
	if (time_ref - distance->timestamp > DISTANCE_TIMEOUT) {
		_terrain_valid = false;
	}

/***********�ɻ���������  �Ĳ���У��(����̬�ٶȲ���)**********/
    // ���������и��� 
	if (distance->timestamp > _time_last_distance) {
		matrix::Quatf q(attitude->q);//��̬��Ԫ�� 
		matrix::Eulerf euler(q);     //�õ�ŷ���� 
		float d = distance->current_distance;//��ǰ���� 

		matrix::Matrix<float, 1, n_x> C;     //������������
		C(0, 0) = -1; // measured altitude,

		float R = 0.009f;//�������ݷ��� 
		
		// ʵ�ʲ���ֵ 
		//У�� roll  pitch ����ĸ߶ȱ仯
		matrix::Vector<float, 1> y;
		y(0) = d * cosf(euler.phi()) * cosf(euler.theta());//ʵ�ʲ�������ĵ��Ľ�� 

		// residual  �в�  
		//���㿨��������   K  =   P' * ������������C.ת�� *��������������C * P' * ������������C.ת�� + �������̸�˹��������R).inv
		matrix::Matrix<float, 1, 1> S_I = (C * _P * C.transpose());
		S_I(0, 0) += R;
		S_I = matrix::inv<float, 1> (S_I);//���� 
		matrix::Matrix<float, n_x, 1> K = _P * C.transpose() * S_I;//���������� 
		
		//��� = ʵ�ʲ���ֵ y -  ������������C * Ԥ���� _x 
		matrix::Vector<float, 1> r = y - C * _x;
		// some sort of outlayer rejection Ұֵ����
		if (fabsf(distance->current_distance - _distance_last) < 1.0f) {//����������������Χ�� 
			_x += K * r; //У������Ԥ��ֵ   �������Ź���ֵ X = X�� + K * ���
			_P -= K * C * _P;//����Ԥ�����С���������� P = ( 1 - K * C) * P' = P' - K * C * P'
		}

		// if the current and the last range measurement are bad then we consider the terrain
		// estimate to be invalid
		if (!is_distance_valid(distance->current_distance) && !is_distance_valid(_distance_last)) {
			_terrain_valid = false;

		} else {
			_terrain_valid = true;//��ǰ��������ϴξ�����һ�����Ϸ�Χ�����ι������ݾ���Ч 
		}

		_time_last_distance = distance->timestamp;   //����ʱ�� ��¼ 
		_distance_last = distance->current_distance; //���¾��� ��¼ 
	}
	
	
/***************Z���ٶ� ��У��������ʹ��GPS����************************************/
    // GPS���ݿ��� 
	if (gps->timestamp > _time_last_gps && gps->fix_type >= 3) {
		matrix::Matrix<float, 1, n_x> C;//������������ 
		C(0, 1) = 1;

		float R = 0.056f;//GPS�������� 
     
	   // ʵ�ʲ���ֵ 
		matrix::Vector<float, 1> y;
		y(0) = gps->vel_d_m_s;

		// residual  ���㿨��������  K  =   P' * ������������C.ת�� *��������������C * P' * ������������C.ת�� + �������̸�˹��������R).inv
		matrix::Matrix<float, 1, 1> S_I = (C * _P * C.transpose());
		S_I(0, 0) += R;
		S_I = matrix::inv<float, 1>(S_I);//���� 
		matrix::Matrix<float, n_x, 1> K = _P * C.transpose() * S_I;//���������� 
		
		//��� = ʵ�ʲ���ֵ y -  ������������C * Ԥ���� _x 
		matrix::Vector<float, 1> r = y - C * _x;
		_x += K * r;       //У������Ԥ��ֵ   �������Ź���ֵ X = X�� + K * ���
		_P -= K * C * _P;  //����Ԥ�����С���������� P = ( 1 - K * C) * P' = P' - K * C * P'

		_time_last_gps = gps->timestamp;//ʱ����� 
	}

	// reinitialise filter if we find bad data
	// ������� �д��� ���³�ʼ�� �������˲��� ����  
	bool reinit = false;

	for (int i = 0; i < n_x; i++) {
		if (!PX4_ISFINITE(_x(i))) {//����Ԥ�� ֵ�ļ�� 
			reinit = true;
		}
	}

	for (int i = 0; i < n_x; i++) {
		for (int j = 0; j < n_x; j++) {
			if (!PX4_ISFINITE(_P(i, j))) {// Ԥ�����Э����ļ�� 
				reinit = true;
			}
		}
	}

	if (reinit) {
		memset(&_x._data[0], 0, sizeof(_x._data));//Ԥ��״̬��ʼ��Ϊ0 
		_P.setZero();                             //Э�����ʼ��Ϊ0 
		_P(0, 0) = _P(1, 1) = _P(2, 2) = 0.1f;
	}
	//[0.1 0 0]  ������ 
	//[0 0.1 0]
	//[0 0 0.1]

}
