/*************************************************************************
        > File Name: SM2.c
        > Author:NEWPLAN
        > E-mail:newplan001@163.com
        > Created Time: Thu Apr 13 23:55:50 2017
 ************************************************************************/


#include "sm2.h"
#include "part1.h"
#include "part2.h"
#include "part3.h"
#include "part4.h"

/*
  ��Ϊ������֤������ǩ������Կ�������ӽ���

  ���� SM2��Բ���߹�Կ�����㷨 �ĵ�������ʾ�����㣬
  ֻ��f2m_257 ��Կ���� �����û��Ӵ�ֵZʱ��һ��

  ecp->point_byte_length��ʾ��ͬ����ʹ�õĶ�����λ��

  DEFINE_SHOW_BIGNUM, 16������ʾ��������ֵ
  DEFINE_SHOW_STRING��16������ʾ�������ַ���
*/
int main(int argc, char* argv[])
{
	{
		//������֤
		printf("********************������֤********************\n");
		test_part1(sm2_param_fp_192, TYPE_GFp, 192);

		test_part1(sm2_param_fp_256, TYPE_GFp, 256);
		test_part1(sm2_param_f2m_193, TYPE_GF2m, 193);
		test_part1(sm2_param_f2m_257, TYPE_GF2m, 257);
		system("pause");
		//����ǩ��
		printf("********************����ǩ��********************\n");
		test_part2(sm2_param_fp_256, TYPE_GFp, 256);
		test_part2(sm2_param_f2m_257, TYPE_GF2m, 257);
		system("pause");
		//��Կ����
		printf("********************��Կ����********************\n");
		test_part3(sm2_param_fp_256, TYPE_GFp, 256);
		//a = 0ʱ���û�hash Z���㲻һ��, ��������Կ��ͬ
		test_part3(sm2_param_f2m_257, TYPE_GF2m, 257);
		system("pause");
		//�ӽ���
		//192, 193λ��ʹ�õ�d, k���ضϴ���
		printf("********************�ӡ�����********************\n");
		test_part4(sm2_param_fp_192, TYPE_GFp, 192);
		test_part4(sm2_param_fp_256, TYPE_GFp, 256);
		test_part4(sm2_param_f2m_193, TYPE_GF2m, 193);
		test_part4(sm2_param_f2m_257, TYPE_GF2m, 257);
		system("pause");
	}

	//�Ƽ�����
	printf("********************������֤********************\n");
	test_part1(sm2_param_recommand, TYPE_GFp, 256);
	system("pause");
	printf("********************����ǩ��********************\n");
	test_part2(sm2_param_recommand, TYPE_GFp, 256);
	system("pause");
	printf("********************��Կ����********************\n");
	test_part3(sm2_param_recommand, TYPE_GFp, 256);
	system("pause");
	printf("********************�ӡ�����********************\n");
	test_part4(sm2_param_recommand, TYPE_GFp, 256);

	//system("pause");
	return 0;
}

