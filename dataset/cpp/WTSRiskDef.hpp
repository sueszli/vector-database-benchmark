/*!
 * \file WTSRiskDef.hpp
 * \project	WonderTrader
 *
 * \author Wesley
 * \date 2020/03/30
 * 
 * \brief WT���������ݶ���
 */
#pragma once
#include "WTSObject.hpp"

NS_WTP_BEGIN
typedef struct _TradeStatInfo
{
	char		_code[MAX_INSTRUMENT_LENGTH];
	//��ƽͳ��
	double	l_openvol;	//���տ������
	double	l_closevol;	//����ƽ�����
	double	l_closetvol;//����ƽ������
	double	s_openvol;	//���տ��ղ���
	double	s_closevol;	//����ƽ�ղ���
	double	s_closetvol;//����ƽ��ղ���

	//�ҵ�ͳ��
	uint32_t	b_orders;	//ί�����
	double		b_ordqty;	//ί������
	uint32_t	s_orders;	//ί������
	double		s_ordqty;	//ί������

	//����ͳ��
	uint32_t	b_cancels;	//�������
	double		b_canclqty;	//��������
	uint32_t	s_cancels;	//��������
	double		s_canclqty;	//��������

	//�Զ�����ͳ��
	uint32_t	b_auto_cancels;		//�������
	double		b_auto_canclqty;	//��������
	uint32_t	s_auto_cancels;		//��������
	double		s_auto_canclqty;	//��������

	//��ͳ��
	uint32_t	b_wrongs;	//������
	double		b_wrongqty;	//������
	uint32_t	s_wrongs;	//������
	double		s_wrongqty;	//������

	uint32_t	_infos;		//��Ϣ��

	_TradeStatInfo()
	{
		memset(this, 0, sizeof(_TradeStatInfo));
	}
} TradeStatInfo;

class WTSTradeStateInfo : public WTSObject
{
protected:
	WTSTradeStateInfo(){}

public:
	static WTSTradeStateInfo* create(const char* code)
	{
		WTSTradeStateInfo* pRet = new WTSTradeStateInfo();
		wt_strcpy(pRet->_trd_stat_info._code, code);

		return pRet;
	}

	inline TradeStatInfo&	statInfo(){ return _trd_stat_info; }
	inline const TradeStatInfo& statInfo() const{ return _trd_stat_info; }

	inline const char* code() const{ return _trd_stat_info._code; }

	inline double open_volume_long() const{ return _trd_stat_info.l_openvol; }
	inline double close_volume_long() const{ return _trd_stat_info.l_closevol; }
	inline double closet_volume_long() const{ return _trd_stat_info.l_closetvol; }
	inline double open_volume_short() const{ return _trd_stat_info.s_openvol; }
	inline double close_volume_short() const{ return _trd_stat_info.s_closevol; }
	inline double closet_volume_short() const{ return _trd_stat_info.s_closetvol; }

	inline uint32_t orders_buy() const{ return _trd_stat_info.b_orders; }
	inline double ordqty_buy() const{ return _trd_stat_info.b_ordqty; }
	inline uint32_t orders_sell() const{ return _trd_stat_info.s_orders; }
	inline double ordqty_sell() const{ return _trd_stat_info.s_ordqty; }

	inline uint32_t cancels_buy() const{ return _trd_stat_info.b_cancels; }
	inline double cancelqty_buy() const{ return _trd_stat_info.b_canclqty; }
	inline uint32_t cancels_sell() const{ return _trd_stat_info.s_cancels; }
	inline double cancelqty_sell() const{ return _trd_stat_info.s_canclqty; }

	inline uint32_t total_cancels() const{ return _trd_stat_info.b_cancels + _trd_stat_info.s_cancels; }
	inline uint32_t total_orders() const { return _trd_stat_info.b_orders + _trd_stat_info.s_orders; }

	inline uint32_t infos() const { return _trd_stat_info._infos; }

private:
	TradeStatInfo	_trd_stat_info;
};

//����ʽ�����
typedef struct _WTSFundStruct
{
	double		_predynbal;		//�ڳ���̬Ȩ��
	double		_prebalance;	//�ڳ���̬Ȩ��
	double		_balance;		//��̬Ȩ��
	double		_profit;		//ƽ��ӯ��
	double		_dynprofit;		//����ӯ��
	double		_fees;			//Ӷ��
	uint32_t	_last_date;		//�ϴν��㽻����

	double		_max_dyn_bal;	//�������ֵ
	uint32_t	_max_time;		//���ڸߵ����ʱ��
	double		_min_dyn_bal;	//������С��ֵ
	uint32_t	_min_time;		//���ڵ͵����ʱ��

	int64_t		_update_time;	//���ݸ���ʱ��

	typedef struct _DynBalPair
	{
		uint32_t	_date;
		double		_dyn_balance;

		_DynBalPair()
		{
			memset(this, 0, sizeof(_DynBalPair));
		}
	} DynBalPair;

	DynBalPair	_max_md_dyn_bal;	//���̬��ֵ
	DynBalPair	_min_md_dyn_bal;	//��С��̬��ֵ

	_WTSFundStruct()
	{
		memset(this, 0, sizeof(_WTSFundStruct));
		_max_dyn_bal = DBL_MAX;
		_min_dyn_bal = DBL_MAX;
	}
} WTSFundStruct;


class WTSPortFundInfo : public WTSObject
{
protected:
	WTSPortFundInfo(){}

public:
	static WTSPortFundInfo* create()
	{
		WTSPortFundInfo* pRet = new WTSPortFundInfo();
		return pRet;
	}

	WTSFundStruct&	fundInfo(){ return _fund_info; }
	const WTSFundStruct& fundInfo() const{ return _fund_info; }

	double predynbalance() const{ return _fund_info._predynbal; }
	double balance() const{ return _fund_info._balance; }
	double profit() const{ return _fund_info._profit; }
	double dynprofit() const{ return _fund_info._dynprofit; }
	double fees() const{ return _fund_info._fees; }

	double max_dyn_balance() const{ return _fund_info._max_dyn_bal; }
	double min_dyn_balance() const{ return _fund_info._min_dyn_bal; }

	double max_md_dyn_balance() const{ return _fund_info._max_md_dyn_bal._dyn_balance; }
	double min_md_dyn_balance() const{ return _fund_info._min_md_dyn_bal._dyn_balance; }

	uint32_t max_dynbal_time() const{ return _fund_info._max_time; }
	uint32_t min_dynbal_time() const{ return _fund_info._min_time; }

	uint32_t last_settle_date() const{ return _fund_info._last_date; }

	uint32_t max_md_dynbal_date() const{ return _fund_info._max_md_dyn_bal._date; }
	uint32_t min_md_dynbal_date() const{ return _fund_info._min_md_dyn_bal._date; }


private:
	WTSFundStruct	_fund_info;
};

NS_WTP_END