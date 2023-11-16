/*!
 * \file WtMinImpactExeUnit.cpp
 *
 * \author Wesley
 * \date 2020/03/30
 *
 * 
 */
#include "WtMinImpactExeUnit.h"

#include "../Includes/WTSVariant.hpp"
#include "../Includes/WTSContractInfo.hpp"
#include "../Includes/WTSSessionInfo.hpp"
#include "../Share/decimal.h"
#include "../Share/StrUtil.hpp"
#include "../Share//fmtlib.h"

extern const char* FACT_NAME;

const char* PriceModeNames[] =
{
	"BESTPX",		//���ż�
	"LASTPX",		//���¼�
	"MARKET",		//���ּ�
	"AUTOPX"		//�Զ�
};

inline double get_real_target(double target)
{
	if (target == DBL_MAX)			 
		return 0;

	return target;
}

inline bool is_clear(double target)
{
	return (target == DBL_MAX);
}


WtMinImpactExeUnit::WtMinImpactExeUnit()
	: _last_tick(NULL)
	, _comm_info(NULL)
	, _price_mode(0)
	, _price_offset(0)
	, _expire_secs(0)
	, _cancel_cnt(0)
	, _target_pos(0)
	, _cancel_times(0)
	, _last_place_time(0)
	, _last_tick_time(0)
	, _in_calc(false)
	, _min_open_lots(1)
{
}


WtMinImpactExeUnit::~WtMinImpactExeUnit()
{
	if (_last_tick)
		_last_tick->release();

	if (_comm_info)
		_comm_info->release();
}

const char* WtMinImpactExeUnit::getFactName()
{
	return FACT_NAME;
}

const char* WtMinImpactExeUnit::getName()
{
	return "WtMinImpactExeUnit";
}

void WtMinImpactExeUnit::init(ExecuteContext* ctx, const char* stdCode, WTSVariant* cfg)
{
	ExecuteUnit::init(ctx, stdCode, cfg);

	_comm_info = ctx->getCommodityInfo(stdCode);//��ȡƷ�ֲ���
	if (_comm_info)
		_comm_info->retain();

	_sess_info = ctx->getSessionInfo(stdCode);//��ȡ����ʱ��ģ����Ϣ
	if (_sess_info)
		_sess_info->retain();

	_price_offset = cfg->getInt32("offset");	//�۸�ƫ��������һ��Ͷ���ͬ����
	_expire_secs = cfg->getUInt32("expire");	//������ʱ����
	_price_mode = cfg->getInt32("pricemode");	//�۸�����,0-���¼�,-1-���ż�,1-���ּ�,2-�Զ�,Ĭ��Ϊ0
	_entrust_span = cfg->getUInt32("span");		//����ʱ��������λ����
	_by_rate = cfg->getBoolean("byrate");		//�Ƿ��ն��ֵĹҵ����ı����µ��������true����rate�ֶ���Ч�������false��lots�ֶ���Ч
	_order_lots = cfg->getDouble("lots");		//���η�������
	_qty_rate = cfg->getDouble("rate");			//�µ���������

	if (cfg->has("minopenlots"))  
		_min_open_lots = cfg->getDouble("minopenlots");	//��С��������

	ctx->writeLog(fmtutil::format("MiniImpactExecUnit of {} inited, order price @ {}��{} ticks, expired after {} secs, reorder after {} millisec, lots policy: {} @ {:.2f}, min open lots: {}",
		stdCode, PriceModeNames[_price_mode + 1], _price_offset, _expire_secs, _entrust_span, _by_rate ? "byrate" : "byvol", _by_rate ? _qty_rate : _order_lots, _min_open_lots));
}
/*
 *	�����ر�
 *	localid	���ص���
 *	code	��Լ����
 *	isBuy	��or��
 *	leftover	ʣ������
 *	price	ί�м۸�
 *	isCanceled	�Ƿ��ѳ���
 */
void WtMinImpactExeUnit::on_order(uint32_t localid, const char* stdCode, bool isBuy, double leftover, double price, bool isCanceled)
{
	{
		if (!_orders_mon.has_order(localid)) //���û�ж�Ӧ���� ����
			return;

		if (isCanceled || leftover == 0)  //�ѳ�����ʣ�ඩ��Ϊ0
		{
			_orders_mon.erase_order(localid);
			if (_cancel_cnt > 0)
			{
				_cancel_cnt--;
				_ctx->writeLog(fmtutil::format("[{}@{}] Order of {} cancelling done, cancelcnt -> {}", __FILE__, __LINE__, _code.c_str(), _cancel_cnt));
			}
		}

		if (leftover == 0 && !isCanceled)
			_cancel_times = 0;
	}

	//����г���,Ҳ�������¼���
	if (isCanceled)
	{
		_ctx->writeLog(fmtutil::format("Order {} of {} canceled, recalc will be done", localid, stdCode));
		_cancel_times++;
		do_calc();
	}
}
/*
 *	����ͨ�������ص�
 */
void WtMinImpactExeUnit::on_channel_ready()
{
	double undone = _ctx->getUndoneQty(_code.c_str());//��ȡδ�������

	if(!decimal::eq(undone, 0) && !_orders_mon.has_order())
	{
		/*
		 *	���δ��ɵ���Ϊ0����OMSû�ж���
		 *	��˵����δ��ɵ����ڼ��֮��,ȫ��������
		 *	��Ϊ��Щ����û�б��ض����ţ��޷�ֱ�ӽ��й���
		 *	������������Ǹ�������ʱ���ϴε�δ��ɵ������ⲿ�Ĺҵ�
		 */
		_ctx->writeLog(fmtutil::format("Unmanaged live orders with qty {} of {} found, cancel all", undone, _code.c_str()));

		bool isBuy = (undone > 0);
		OrderIDs ids = _ctx->cancel(_code.c_str(), isBuy);//���ݱ��ض����ų���
		_orders_mon.push_order(ids.data(), ids.size(), _ctx->getCurTime()); //push into orderpair
		_cancel_cnt += ids.size();

		_ctx->writeLog(fmtutil::format("[{}@{}]cancelcnt -> {}", __FILE__, __LINE__, _cancel_cnt));
	}
	else if (decimal::eq(undone, 0) && _orders_mon.has_order())
	{
		/*
		 *	By Wesey @ 2021.12.13
		 *	���δ��ɵ�Ϊ0������OMS�����ж�����
		 *	˵��OMS���Ǵ�����Ҫ���������Ȼ��ʱ�����ͻ����
		 *	���������һ���Ƕ��������Ժ�֮ǰ�³�ȥ�Ķ�������û���������͵���̨
		 *	����������Ҫ��������ض���
		 */
		_ctx->writeLog(fmtutil::format("Local orders of {} not confirmed in trading channel, clear all", _code.c_str()));
		_orders_mon.clear_orders();
	}
	else
	{
		_ctx->writeLog(fmtutil::format("Unrecognized condition while channle ready, {:.2f} live orders of {} exists, local orders {}exist",
			undone, _code.c_str(), _orders_mon.has_order() ? "" : "not "));
	}


	do_calc();
}
/*
	 *	����ͨ����ʧ�ص�
	 */
void WtMinImpactExeUnit::on_channel_lost()
{
	
}
/*
 *	tick���ݻص�
 *	newTick	���µ�tick����
 */
void WtMinImpactExeUnit::on_tick(WTSTickData* newTick)
{
	if (newTick == NULL || _code.compare(newTick->code()) != 0)
		return;

	//���ԭ����tick��Ϊ��,��Ҫ�ͷŵ�
	if (_last_tick)
	{
		_last_tick->release();
	}
	else
	{
		//�������ʱ�䲻�ڽ���ʱ��,�������һ���Ǽ��Ͼ��۵��������,�µ���ʧ��,����ֱ�ӹ��˵��������
		if (_sess_info != NULL && !_sess_info->isInTradingTime(newTick->actiontime() / 100000))
			return;
	}

	//�µ�tick����,Ҫ����
	_last_tick = newTick;
	_last_tick->retain();

	/*
	 *	������Կ���һ��
	 *	���д����һ�ζ���ȥ�ĵ��Ӳ����ﵽĿ���λ
	 *	��ô���µ��������ݽ�����ʱ������ٴδ��������߼�
	 */
		//*********��ontick�жԶ����������У�顣 �����в���Ծ��Լ��У�����  ��֮����
	if(_expire_secs != 0 && _orders_mon.has_order() && _cancel_cnt==0)  //������ʱ������=0&&hasorder && ������==0
	{
		uint64_t now = _ctx->getCurTime();

		_orders_mon.check_orders(_expire_secs, now, [this](uint32_t localid) {
			if (_ctx->cancel(localid))
			{
				_cancel_cnt++;
				_ctx->writeLog(fmtutil::format("[{}@{}] Expired order of {} canceled, cancelcnt -> {}", __FILE__, __LINE__, _code.c_str(), _cancel_cnt));
			}
		});
	}
	
	do_calc();
}
/*
 *	�ɽ��ر�
 *	code	��Լ����
 *	isBuy	��or��
 *	vol		�ɽ�����,����û������,ͨ��isBuyȷ�����뻹������
 *	price	�ɽ��۸�
 */
void WtMinImpactExeUnit::on_trade(uint32_t localid, const char* stdCode, bool isBuy, double vol, double price)
{
	//���ô���,������ontick�ﴥ����
}

/*
 *	�µ�����ر�
 */
void WtMinImpactExeUnit::on_entrust(uint32_t localid, const char* stdCode, bool bSuccess, const char* message)
{
	if (!bSuccess)
	{
		//��������ҷ���ȥ�Ķ���,�ҾͲ�����
		if (!_orders_mon.has_order(localid))
			return;

		_orders_mon.erase_order(localid);

		do_calc();
	}
}

void WtMinImpactExeUnit::do_calc()
{
	CalcFlag flag(&_in_calc);
	if (flag)
		return;

	if (_cancel_cnt != 0)
		return;

	//�����һ��������Ҫԭ����ʵ�̹����з���
	//���޸�Ŀ���λ��ʱ�򣬻ᴥ��һ��do_calc
	//��ontickҲ�ᴥ��һ��do_calc�����ε����Ǵ������̷ֱ߳𴥷��ģ����Ի����ͬʱ���������
	//������������ͻ���������
	//���������ԭ����SimpleExecUnitû�г��֣���ΪSimpleExecUnitֻ��set_position��ʱ�򴥷�
	StdUniqueLock lock(_mtx_calc );

	double newVol = get_real_target(_target_pos);//��ʵ�۸�Ŀ���λ  newvol
	const char* stdCode = _code.c_str();

	double undone = _ctx->getUndoneQty(stdCode);  //undone  δ��ɶ�����
	double realPos = _ctx->getPosition(stdCode);  //realpos ��ȡ��λ 
	double diffPos = newVol - realPos;			  //diffpos = ��ʵĿ���λ-��ȡ��λ

	//��δ��ɶ�������ʵ�ʲ�λ�䶯�����෴
	//����Ҫ�������ж���
	if (decimal::lt(diffPos * undone, 0)) //diff*undone ʣ����Ҫ���*δ��ɶ��� <0   true   -> cancel
	{
		bool isBuy = decimal::gt(undone, 0);   //δ��ɵ�>0   isbuy = 1 ��
		OrderIDs ids = _ctx->cancel(stdCode, isBuy);
		if(!ids.empty())
		{
			_orders_mon.push_order(ids.data(), ids.size(), _ctx->getCurTime());
			_cancel_cnt += ids.size();
			_ctx->writeLog(fmtutil::format("[{}@{}] live opposite order of {} canceled, cancelcnt -> {}", __FILE__, __LINE__, _code.c_str(), _cancel_cnt));//�෴�Ķ�����ȡ��
		}
		return;
	}

	//��Ϊ����ʷ�������������в���Ҫ������δ��ɵ������ݲ�����
	//****�˴����ָ �𵥵Ļغ�����һ�ʶ�����δ��ᣬ�򲻷����� ��ʷ���ָÿ���غϳɽ��ĵ�
	if (!decimal::eq(undone, 0))
		return;

	double curPos = realPos;   //realPos��ȡ��λ   curPos  Ϊ��ȡ���ڵĲ�λ

	if (_last_tick == NULL)
	{
		_ctx->writeLog(fmtutil::format("No lastest tick data of {}, execute later", _code.c_str()));
		return;
	}

	//����µ�ʱ����
	uint64_t now = TimeUtils::makeTime(_last_tick->actiondate(), _last_tick->actiontime());
	if (now - _last_place_time < _entrust_span) //��ǰʱ��-��ticktime <����ʱ����
		return;

	if (decimal::eq(curPos, newVol)) 
	{
		//��ǰ��λ�����²�λƥ��ʱ���������ȫ����ֵ�������ֱ���˳�������
		if (!is_clear(_target_pos))
			return;

		//�������ֵ����󣬻�Ҫ�ٽ��жԱ�
		//�����ͷΪ0��˵���Ѿ�ȫ��������ˣ���ֱ���˳�
		double lPos = _ctx->getPosition(stdCode, true , 1); //��ȡ����ͷ���ֲ�  ���óֲ�   ��ͷ //����ֵ	��ƽ��Ĳ�λ: ���>0, �ղ�<0
		if (decimal::eq(lPos, 0))
			return;

		//������ж�ͷ��λ����Ŀ���λ����Ϊ��0��ǿ�ƴ���                      
		newVol = -min(lPos, _order_lots);//  -min(��ȡ����ͷ���ֲ�     ���η�������) 
		_ctx->writeLog(fmtutil::format("Clearing process triggered, target position of {} has been set to {}", _code.c_str(), newVol));
	}

	bool bForceClose = is_clear(_target_pos); //target==del_max  return 1

	bool isBuy = decimal::gt(newVol, curPos);//����ж���������    ��ʵĿ���λ-��ȡ�ֲ�   > 0 ��ͷ

	//�������ϴ�û�и��µ�tick���������Ȳ��µ�����ֹ����ǰ�����µ�����ͨ������
	uint64_t curTickTime = (uint64_t)_last_tick->actiondate() * 1000000000 + _last_tick->actiontime();
	if (curTickTime <= _last_tick_time)
	{
		_ctx->writeLog(fmtutil::format("No tick of {} updated, {} <= {}, execute later", _code, curTickTime, _last_tick_time));
		return;
	}

	_last_tick_time = curTickTime;

	double this_qty = _order_lots; 	//���η�������
	if (_by_rate)//�Ƿ��ն��ֵĹҵ����ı����µ��������true����rate�ֶ���Ч�������false��lots�ֶ���Ч
	{
		this_qty = isBuy ? _last_tick->askqty(0) : _last_tick->bidqty(0);	//isbuy ture   askqty �����
		this_qty = round(this_qty*_qty_rate);
		if (decimal::lt(this_qty, 1))  //this_qty <0 ʱ��return this_qty =1 
			this_qty = 1;
	}

	//By Wesley @ 2022.09.13
	//����Ҫ���µ�������һ������
	this_qty = min(this_qty, abs(newVol - curPos));//��ʵ-��ȡ��λ			

	//�Ƿ񿪲֣�����ֲִ��ڵ���0�����룬���ֲ߳�С�ڵ���0�����������ǿ���
	bool isOpen = (isBuy && decimal::ge(curPos, 0)) || (!isBuy && decimal::le(curPos, 0));

	//���ƽ�ֵĻ�
	//�Ե����µ���һ����������֤ƽ�ֺͿ��ֲ���ͬʱ�µ�
	if (!isOpen)
	{
		this_qty = min(this_qty, abs(curPos)); //curPos ���ڵĲ�λ
	}									

	/*
	 *	By Wesley @ 2022.12.15
	 *	����һ������С�µ������������߼�
	 */
	if (isOpen && decimal::lt(this_qty, _min_open_lots))//if ����&&this_qty<min_open_lots����С����������
	{
		this_qty = _min_open_lots; 
		_ctx->writeLog(fmtutil::format("Lots of {} changed from {} to {} due to minimum open lots", _code, this_qty, _min_open_lots));
	}

	double buyPx, sellPx;
	if (_price_mode == 2)//�۸�����,0-���¼�,-1-���ż�,1-���ּ�,2-�Զ�,Ĭ��Ϊ0
	{
		double mp = (_last_tick->bidqty(0) - _last_tick->askqty(0))*1.0 / (_last_tick->bidqty(0) + _last_tick->askqty(0));
		bool isUp = (mp > 0);
		if (isUp)
		{
			buyPx = _last_tick->askprice(0);
			sellPx = _last_tick->askprice(0);
		}
		else
		{
			buyPx = _last_tick->bidprice(0);
			sellPx = _last_tick->bidprice(0);
		}

		/*
		 *	By Wesley @ 2022.03.07
		 *	������۸�Ϊ0������һ������			�۸�Ϊ0�����ܵ���û�н��ף�����ȡ��һ�������յ����̼�
		 */																		
		if (decimal::eq(buyPx, 0.0)) //buypx==0��return ture
			buyPx = decimal::eq(_last_tick->price(), 0.0) ? _last_tick->preclose() : _last_tick->price(); //�����һtick ==0  ȡlasttick ���̼�

		if (decimal::eq(sellPx, 0.0))
			sellPx = decimal::eq(_last_tick->price(), 0.0) ? _last_tick->preclose() : _last_tick->price();

		buyPx += _comm_info->getPriceTick() * _cancel_times;   // // * ��������    ����׷������
		sellPx -= _comm_info->getPriceTick() * _cancel_times;
	}
	else
	{
		if (_price_mode == -1)//0 ���¼� -1 ���ż� 1 ���ּ� 
		{
			buyPx = _last_tick->bidprice(0);
			sellPx = _last_tick->askprice(0);
		}
		else if (_price_mode == 0)
		{
			buyPx = _last_tick->price();
			sellPx = _last_tick->price();
		}
		else if (_price_mode == 1)
		{
			buyPx = _last_tick->askprice(0);
			sellPx = _last_tick->bidprice(0);
		}

		/*
		 *	By Wesley @ 2022.03.07
		 *	������۸�Ϊ0������һ������
		 */
		if (decimal::eq(buyPx, 0.0))
			buyPx = decimal::eq(_last_tick->price(), 0.0)? _last_tick->preclose(): _last_tick->price();

		if (decimal::eq(sellPx, 0.0))
			sellPx = decimal::eq(_last_tick->price(), 0.0) ? _last_tick->preclose() : _last_tick->price();

		buyPx += _comm_info->getPriceTick() * _price_offset;//�۸�ƫ������
		sellPx -= _comm_info->getPriceTick() * _price_offset;
	}
	

	//����ǵ�ͣ��
	bool isCanCancel = true;  
	if (!decimal::eq(_last_tick->upperlimit(), 0) && decimal::gt(buyPx, _last_tick->upperlimit())) //upperlimit��ͣ��
	{
		_ctx->writeLog(fmtutil::format("Buy price {} of {} modified to upper limit price", buyPx, _code.c_str(), _last_tick->upperlimit()));//����ۣ�������Ϊ���޼�
		buyPx = _last_tick->upperlimit();
		isCanCancel = false;	//����۸�����Ϊ�ǵ�ͣ�ۣ��������ɳ���
	}
	
	if (!decimal::eq(_last_tick->lowerlimit(), 0) && decimal::lt(sellPx, _last_tick->lowerlimit()))
	{
		_ctx->writeLog(fmtutil::format("Sell price {} of {} modified to lower limit price", sellPx, _code.c_str(), _last_tick->lowerlimit()));
		sellPx = _last_tick->lowerlimit();
		isCanCancel = false;	//����۸�����Ϊ�ǵ�ͣ�ۣ��������ɳ���
	}

	if (isBuy)
	{
		OrderIDs ids = _ctx->buy(stdCode, buyPx, this_qty, bForceClose);
		_orders_mon.push_order(ids.data(), ids.size(), _ctx->getCurTime(), isCanCancel);
	}
	else
	{
		OrderIDs ids  = _ctx->sell(stdCode, sellPx, this_qty, bForceClose);
		_orders_mon.push_order(ids.data(), ids.size(), _ctx->getCurTime(), isCanCancel);
	}

	_last_place_time = now;
}
/*
 *	�����µ�Ŀ���λ
 *	code	��Լ����
 *	newVol	�µ�Ŀ���λ
 */
void WtMinImpactExeUnit::set_position(const char* stdCode, double newVol)
{
	if (_code.compare(stdCode) != 0) //code��stdcode�����  return 
		return;

	//���ԭ����Ŀ���λ��DBL_MAX��˵���Ѿ����������߼�
	//������ʱ��������Ϊ0����ֱ��������
	if (is_clear(_target_pos) && decimal::eq(newVol, 0))
	{
		_ctx->writeLog(fmtutil::format("{} is in clearing processing, position can not be set to 0", stdCode));
		return;
	}

	if (decimal::eq(_target_pos, newVol))
		return;

	_target_pos = newVol;

	if (is_clear(_target_pos))
		_ctx->writeLog(fmtutil::format("{} is set to be in clearing processing", stdCode));
	else
		_ctx->writeLog(fmtutil::format("Target position of {} is set tb be {}", stdCode, _target_pos));

	do_calc();
}

void WtMinImpactExeUnit::clear_all_position(const char* stdCode)
{
	if (_code.compare(stdCode) != 0)
		return;

	_target_pos = DBL_MAX;

	do_calc();
}
