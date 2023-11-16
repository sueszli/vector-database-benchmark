#include "WtHftStraDemo.h"
#include "../Includes/IHftStraCtx.h"

#include "../Includes/WTSVariant.hpp"
#include "../Includes/WTSDataDef.hpp"
#include "../Includes/WTSContractInfo.hpp"
#include "../Share/TimeUtils.hpp"
#include "../Share/decimal.h"
#include "../Share/fmtlib.h"

extern const char* FACT_NAME;

WtHftStraDemo::WtHftStraDemo(const char* id)
	: HftStrategy(id)
	, _last_tick(NULL)
	, _last_entry_time(UINT64_MAX)
	, _channel_ready(false)
	, _last_calc_time(0)
	, _stock(false)
	, _unit(1)
	, _cancel_cnt(0)
	, _reserved(0)
{
}


WtHftStraDemo::~WtHftStraDemo()
{
	if (_last_tick)
		_last_tick->release();
}

const char* WtHftStraDemo::getName()
{
	return "HftDemoStrategy";
}

const char* WtHftStraDemo::getFactName()
{
	return FACT_NAME;
}

bool WtHftStraDemo::init(WTSVariant* cfg)
{
	//������ʾһ���ⲿ��������Ļ�ȡ
	_code = cfg->getCString("code");
	_secs = cfg->getUInt32("second");
	_freq = cfg->getUInt32("freq");
	_offset = cfg->getUInt32("offset");
	_reserved = cfg->getDouble("reserve");

	_stock = cfg->getBoolean("stock");
	_unit = _stock ? 100 : 1;

	return true;
}

void WtHftStraDemo::on_entrust(uint32_t localid, bool bSuccess, const char* message, const char* userTag)
{

}

void WtHftStraDemo::on_init(IHftStraCtx* ctx)
{
	//WTSTickSlice* ticks = ctx->stra_get_ticks(_code.c_str(), _count);
	//if (ticks)
	//	ticks->release();

	WTSKlineSlice* kline = ctx->stra_get_bars(_code.c_str(), "m1", 30);
	if (kline)
		kline->release();

	ctx->stra_sub_ticks(_code.c_str());

	_ctx = ctx;
}

void WtHftStraDemo::do_calc(IHftStraCtx* ctx)
{
	const char* code = _code.c_str();

	//30���ڲ��ظ�����
	uint64_t now = TimeUtils::makeTime(ctx->stra_get_date(), ctx->stra_get_time() * 100000 + ctx->stra_get_secs());//(uint64_t)ctx->stra_get_date()*1000000000 + (uint64_t)ctx->stra_get_time()*100000 + ctx->stra_get_secs();
	if (now - _last_entry_time <= _freq * 1000)
	{
		return;
	}

	WTSTickData* curTick = ctx->stra_get_last_tick(code);
	if (curTick == NULL)
		return;

	uint32_t curMin = curTick->actiontime() / 100000;	//actiontime�Ǵ������,Ҫȡ�÷���,����Ҫ����10w
	if (curMin > _last_calc_time)
	{//���spread�ϴμ����ʱ��С�ڵ�ǰ����,������spread
		//WTSKlineSlice* kline = ctx->stra_get_bars(code, "m5", 30);
		//if (kline)
		//	kline->release();

		//���������Ժ�,���¼���ʱ��
		_last_calc_time = curMin;
	}

	int32_t signal = 0;
	double price = curTick->price();
	//���㲿��
	double pxInThry = (curTick->bidprice(0)*curTick->askqty(0) + curTick->askprice(0)*curTick->bidqty(0)) / (curTick->bidqty(0) + curTick->askqty(0));

	//���ۼ۸�������¼�
	if (pxInThry > price)
	{
		//�����ź�
		signal = 1;
	}
	else if (pxInThry < price)
	{
		//�����ź�
		signal = -1;
	}

	if (signal != 0)
	{
		double curPos = ctx->stra_get_position(code);
		curPos -= _reserved;

		WTSCommodityInfo* cInfo = ctx->stra_get_comminfo(code);

		if (signal > 0 && curPos <= 0)
		{//�����ź�,�ҵ�ǰ��λС�ڵ���0
			//���¼�+2���µ�
			double targetPx = price + cInfo->getPriceTick() * _offset;
			auto ids = ctx->stra_buy(code, targetPx, _unit, "enterlong");

			_mtx_ords.lock();
			for (auto localid : ids)
			{
				_orders.insert(localid);
			}
			_mtx_ords.unlock();
			_last_entry_time = now;
		}
		else if (signal < 0 && (curPos > 0 || ((!_stock || !decimal::eq(_reserved, 0)) && curPos == 0)))
		{//�����ź�,�ҵ�ǰ��λ����0,���߲�λΪ0�����ǹ�Ʊ,���߲�λΪ0���ǻ�����λ������
			//���¼�-2���µ�
			double targetPx = price - cInfo->getPriceTick()*_offset;
			auto ids = ctx->stra_sell(code, targetPx, _unit, "entershort");

			_mtx_ords.lock();
			for (auto localid : ids)
			{
				_orders.insert(localid);
			}
			_mtx_ords.unlock();
			_last_entry_time = now;
		}
	}

	curTick->release();
}

void WtHftStraDemo::on_tick(IHftStraCtx* ctx, const char* code, WTSTickData* newTick)
{	
	if (_code.compare(code) != 0)
		return;

	if (!_orders.empty())
	{
		check_orders();
		return;
	}

	if (!_channel_ready)
		return;

	do_calc(ctx);
}

void WtHftStraDemo::check_orders()
{
	if (!_orders.empty() && _last_entry_time != UINT64_MAX)
	{
		uint64_t now = TimeUtils::makeTime(_ctx->stra_get_date(), _ctx->stra_get_time() * 100000 + _ctx->stra_get_secs());
		if (now - _last_entry_time >= _secs * 1000)	//�������һ��ʱ��û�гɽ���,����
		{
			_mtx_ords.lock();
			for (auto localid : _orders)
			{
				_ctx->stra_cancel(localid);
				_cancel_cnt++;
				_ctx->stra_log_info(fmt::format("Order expired, cancelcnt updated to {}", _cancel_cnt).c_str());
			}
			_mtx_ords.unlock();
		}
	}
}

void WtHftStraDemo::on_bar(IHftStraCtx* ctx, const char* code, const char* period, uint32_t times, WTSBarStruct* newBar)
{
	
}

void WtHftStraDemo::on_trade(IHftStraCtx* ctx, uint32_t localid, const char* stdCode, bool isBuy, double qty, double price, const char* userTag)
{
	do_calc(ctx);
}

void WtHftStraDemo::on_position(IHftStraCtx* ctx, const char* stdCode, bool isLong, double prevol, double preavail, double newvol, double newavail)
{
	
}

void WtHftStraDemo::on_order(IHftStraCtx* ctx, uint32_t localid, const char* stdCode, bool isBuy, double totalQty, double leftQty, double price, bool isCanceled, const char* userTag)
{
	//��������ҷ���ȥ�Ķ���,�ҾͲ�����
	auto it = _orders.find(localid);
	if (it == _orders.end())
		return;

	//����ѳ�������ʣ������Ϊ0,�������ԭ�е�id��¼
	if(isCanceled || leftQty == 0)
	{
		_mtx_ords.lock();
		_orders.erase(it);
		if (_cancel_cnt > 0)
		{
			_cancel_cnt--;
			_ctx->stra_log_info(fmt::format("cancelcnt -> {}", _cancel_cnt).c_str());
		}
		_mtx_ords.unlock();

		do_calc(ctx);
	}
}


void WtHftStraDemo::on_channel_ready(IHftStraCtx* ctx)
{
	double undone = _ctx->stra_get_undone(_code.c_str());
	if (!decimal::eq(undone, 0) && _orders.empty())
	{
		//��˵����δ��ɵ����ڼ��֮��,�ȳ���
		_ctx->stra_log_info(fmt::format("{}�в��ڹ����е�δ��ɵ� {} ��,ȫ������", _code, undone).c_str());

		bool isBuy = (undone > 0);
		OrderIDs ids = _ctx->stra_cancel(_code.c_str(), isBuy, undone);
		for (auto localid : ids)
		{
			_orders.insert(localid);
		}
		_cancel_cnt += ids.size();

		_ctx->stra_log_info(fmt::format("cancelcnt -> {}", _cancel_cnt).c_str());
	}

	_channel_ready = true;
}

void WtHftStraDemo::on_channel_lost(IHftStraCtx* ctx)
{
	_channel_ready = false;
}