/*******************************************************
 Copyright (C) 2013,2014 Guan Lisheng (guanlisheng@gmail.com)
 Copyright (C) 2022 Mark Whalley (mark@ipx.co.uk)

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 ********************************************************/

#include "Model_Asset.h"
#include "Model_Translink.h"
#include "Model_CurrencyHistory.h"

const std::vector<std::pair<Model_Asset::RATE, wxString> > Model_Asset::RATE_CHOICES = 
{
    {Model_Asset::RATE_NONE, wxString(wxTRANSLATE("None"))}
    , {Model_Asset::RATE_APPRECIATE, wxString(wxTRANSLATE("Appreciates"))}
    , {Model_Asset::RATE_DEPRECIATE, wxString(wxTRANSLATE("Depreciates"))}
};

const std::vector<std::pair<Model_Asset::RATEMODE, wxString> > Model_Asset::RATEMODE_CHOICES = 
{
    {Model_Asset::PERCENTAGE, wxString(wxTRANSLATE("Percentage"))}
    , {Model_Asset::LINEAR, wxString(wxTRANSLATE("Linear"))}
};

const wxString Model_Asset::PERCENTAGE_STR = all_ratemode()[PERCENTAGE];
const wxString Model_Asset::LINEAR_STR = all_ratemode()[LINEAR];

const std::vector<std::pair<Model_Asset::TYPE, wxString> > Model_Asset::TYPE_CHOICES = 
{
    {Model_Asset::TYPE_PROPERTY, wxString(wxTRANSLATE("Property"))}
    , {Model_Asset::TYPE_AUTO, wxString(wxTRANSLATE("Automobile"))}
    , {Model_Asset::TYPE_HOUSE, wxString(wxTRANSLATE("Household Object"))}
    , {Model_Asset::TYPE_ART, wxString(wxTRANSLATE("Art"))}
    , {Model_Asset::TYPE_JEWELLERY, wxString(wxTRANSLATE("Jewellery"))}
    , {Model_Asset::TYPE_CASH, wxString(wxTRANSLATE("Cash"))}
    , {Model_Asset::TYPE_OTHER, wxString(wxTRANSLATE("Other"))}
};

const std::vector<std::pair<Model_Asset::STATUS, wxString> > Model_Asset::STATUS_CHOICES = 
{
    {Model_Asset::STATUS_CLOSED, wxString(wxTRANSLATE("Closed"))}
    , {Model_Asset::STATUS_OPEN, wxString(wxTRANSLATE("Open"))}
};

const wxString Model_Asset::OPEN_STR = all_status()[STATUS_OPEN];
const wxString Model_Asset::CLOSED_STR = all_status()[STATUS_CLOSED];

Model_Asset::Model_Asset()
: Model<DB_Table_ASSETS_V1>()
{
}

Model_Asset::~Model_Asset()
{
}

/**
* Initialize the global Model_Asset table.
* Reset the Model_Asset table or create the table if it does not exist.
*/
Model_Asset& Model_Asset::instance(wxSQLite3Database* db)
{
    Model_Asset& ins = Singleton<Model_Asset>::instance();
    ins.db_ = db;
    ins.destroy_cache();
    ins.ensure(db);

    return ins;
}

/** Return the static instance of Model_Asset table */
Model_Asset& Model_Asset::instance()
{
    return Singleton<Model_Asset>::instance();
}

wxString Model_Asset::get_asset_name(int asset_id)
{
    Data* asset = instance().get(asset_id);
    if (asset)
        return asset->ASSETNAME;
    else
        return _("Asset Error");
}

wxArrayString Model_Asset::all_rate()
{
    wxArrayString rates;
    for (const auto& item: RATE_CHOICES) rates.Add(item.second);
    return rates;
}

wxArrayString Model_Asset::all_ratemode()
{
    wxArrayString ratemodes;
    for (const auto& item: RATEMODE_CHOICES) ratemodes.Add(item.second);
    return ratemodes;
}

wxArrayString Model_Asset::all_type()
{
    wxArrayString types;
    for (const auto& item: TYPE_CHOICES) types.Add(item.second);
    return types;
}

wxArrayString Model_Asset::all_status()
{
    wxArrayString statusList;
    for (const auto& item: STATUS_CHOICES) statusList.Add(item.second);
    return statusList;
}

double Model_Asset::balance()
{
    double balance = 0.0;
    for (const auto& r: this->all())
    {
        balance += value(r);
    }
    return balance;
}

DB_Table_ASSETS_V1::ASSETTYPE Model_Asset::ASSETTYPE(TYPE type, OP op)
{
    return DB_Table_ASSETS_V1::ASSETTYPE(all_type()[type], op);
}

DB_Table_ASSETS_V1::STARTDATE Model_Asset::STARTDATE(const wxDate& date, OP op)
{
    return DB_Table_ASSETS_V1::STARTDATE(date.FormatISODate(), op);
}

wxDate Model_Asset::STARTDATE(const Data* r)
{
    return Model::to_date(r->STARTDATE);
}

wxDate Model_Asset::STARTDATE(const Data& r)
{
    return Model::to_date(r.STARTDATE);
}

Model_Asset::TYPE Model_Asset::type(const Data* r)
{
    for (const auto& item : TYPE_CHOICES) if (item.second.CmpNoCase(r->ASSETTYPE) == 0) return item.first;

    return TYPE(-1);
}

Model_Asset::TYPE Model_Asset::type(const Data& r)
{
    return type(&r);
}

Model_Asset::RATE Model_Asset::rate(const Data* r)
{
    for (const auto & item : RATE_CHOICES) if (item.second.CmpNoCase(r->VALUECHANGE) == 0) return item.first;
    return RATE(-1);
}

Model_Asset::RATE Model_Asset::rate(const Data& r)
{
    return rate(&r);
}

Model_Asset::RATEMODE Model_Asset::ratemode(const Data* r)
{
    for (const auto & item : RATEMODE_CHOICES) if (item.second.CmpNoCase(r->VALUECHANGEMODE) == 0) return item.first;
    return RATEMODE(-1);
}

Model_Asset::RATEMODE Model_Asset::ratemode(const Data& r)
{
    return ratemode(&r);
}

Model_Asset::STATUS Model_Asset::status(const Data* r)
{
    for (const auto & item : STATUS_CHOICES) if (item.second.CmpNoCase(r->ASSETSTATUS) == 0) return item.first;
    return STATUS(-1);
}

Model_Asset::STATUS Model_Asset::status(const Data& r)
{
    return status(&r);
}

Model_Currency::Data* Model_Asset::currency(const Data* /* r */)
{
    return Model_Currency::instance().GetBaseCurrency();
}

double Model_Asset::value(const Data* r)
{
    return instance().valueAtDate(r, wxDate::Today());
}

double Model_Asset::value(const Data& r)
{
    return instance().valueAtDate(&r, wxDate::Today());
}

double Model_Asset::valueAtDate(const Data* r, const wxDate date)
{
    double balance = 0;
    if (date >= STARTDATE(r)) {
        Model_Translink::Data_Set translink_records = Model_Translink::instance().find(Model_Translink::LINKRECORDID(r->ASSETID), Model_Translink::LINKTYPE(Model_Attachment::reftype_desc(Model_Attachment::ASSET)));
        if (!translink_records.empty())
        {
            for (const auto& link : translink_records)
            {
                const Model_Checking::Data* tran = Model_Checking::instance().get(link.CHECKINGACCOUNTID);
                const wxDate tranDate = Model_Checking::TRANSDATE(tran);
                if (tranDate <= date)
                {
                    double amount = -1 * Model_Checking::balance(tran, tran->ACCOUNTID) *
                        Model_CurrencyHistory::getDayRate(Model_Account::instance().get(tran->ACCOUNTID)->CURRENCYID, tranDate);
                    wxTimeSpan diff_time = date - tranDate;
                    double diff_time_in_days = static_cast<double>(diff_time.GetDays());

                    switch (rate(r))
                    {
                    case RATE_NONE:
                        break;
                    case RATE_APPRECIATE:
                        amount *= pow(1.0 + (r->VALUECHANGERATE / 36500.0), diff_time_in_days);
                        break;
                    case RATE_DEPRECIATE:
                        amount *= pow(1.0 - (r->VALUECHANGERATE / 36500.0), diff_time_in_days);
                        break;
                    default:
                        break;
                    }

                    balance += amount;
                }
            }
        }
        else {
            balance = r->VALUE;
            wxTimeSpan diff_time = date - STARTDATE(r);
            double diff_time_in_days = static_cast<double>(diff_time.GetDays());

            switch (rate(r))
            {
            case RATE_NONE:
                break;
            case RATE_APPRECIATE:
                balance *= pow(1.0 + (r->VALUECHANGERATE / 36500.0), diff_time_in_days);
                break;
            case RATE_DEPRECIATE:
                balance *= pow(1.0 - (r->VALUECHANGERATE / 36500.0), diff_time_in_days);
                break;
            default:
                break;
            }
        }
    }
    return balance;
}