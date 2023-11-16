#include "../include/GuiLite.h"
#include "../source/ui_ctrl_ex/value_view.h"
#include "../source/ui_ctrl_ex/value_ctrl.h"
#include "../source/ui_ctrl_ex/value_sub_ctrl.h"
#include "../include/ctrl_id.h"
#include "../source/manager/value_manager.h"
#include "../source/manager/value_ctrl_manager.h"
#include <string.h>
#include "temp_value_view.h"

void c_temp_value_view::on_init_children(void)
{
	c_value_ctrl *p_value_t1 = (c_value_ctrl*)get_wnd_ptr(ID_TEMP_VIEW_T1_VALUE);
	c_value_sub_ctrl *p_value_t2 = (c_value_sub_ctrl*)get_wnd_ptr(ID_TEMP_VIEW_T2_VALUE);
	c_value_sub_ctrl *p_value_td = (c_value_sub_ctrl*)get_wnd_ptr(ID_TEMP_VIEW_TD_VALUE);
	if ((p_value_t1==0) || (p_value_t2==0) || (p_value_td == 0))
	{
	    ASSERT(false);
		return;
	}

	c_value_ctrl_manage::get_instance()->config_param_ctrl_att(VALUE_TEMP_T1,p_value_t1);
	c_value_ctrl_manage::get_instance()->config_param_ctrl_att(VALUE_TEMP_T2,p_value_t2);
	c_value_ctrl_manage::get_instance()->config_param_ctrl_att(VALUE_TEMP_TD,p_value_td);

	c_value_view::register_value_view(this);
}
