#include "../include/GuiLite.h"
#include "../include/ctrl_id.h"

#include "../source/ui_ctrl_ex/value_view.h"
#include "../source/ui_ctrl_ex/value_ctrl.h"
#include "../source/manager/value_manager.h"
#include "../source/manager/value_ctrl_manager.h"
#include "nibp_value_view.h"

void c_nibp_value_view::on_init_children(void)
{
	//Register param ctrls.
	c_label *p_name = (c_label*)get_wnd_ptr(ID_NIBP_VIEW);
	c_value_ctrl *p_sys_value = (c_value_ctrl*)get_wnd_ptr(ID_NIBP_VIEW_SYS_VALUE);
	c_value_ctrl *p_dia_value = (c_value_ctrl*)get_wnd_ptr(ID_NIBP_VIEW_DIA_VALUE);
	c_value_ctrl *p_mean_value = (c_value_ctrl*)get_wnd_ptr( ID_NIBP_VIEW_MEAN_VALUE);

	if ((p_name==0) ||(p_sys_value==0) || (p_dia_value==0) || (p_mean_value==0))
	{
        ASSERT(false);
		return;
	}

	p_name->set_font_type((const LATTICE_FONT_INFO*)c_theme::get_font(FONT_DEFAULT));
	p_name->set_font_color(GL_RGB(255, 255, 255));

	c_value_ctrl_manage::get_instance()->config_param_ctrl_att(VALUE_NIBP_SYS, p_sys_value);
	c_value_ctrl_manage::get_instance()->config_param_ctrl_att(VALUE_NIBP_DIA, p_dia_value);
	c_value_ctrl_manage::get_instance()->config_param_ctrl_att(VALUE_NIBP_MEAN,p_mean_value);

	c_value_view::register_value_view(this);
}
