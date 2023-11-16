// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "vw/experimental/workspace.h"

VW_DLL_PUBLIC VWStatus vw_create_workspace(VWOptions* options_handle, VWTraceMessageFunc trace_listener,
    void* trace_context, VWWorkspace** output_handle, VWErrorInfo* err_info_container) noexcept
{
  return VW_not_implemented;
}

VW_DLL_PUBLIC VWStatus vw_create_workspace_with_model(VWOptions* options_handle, void* context, VWReadFunc* read_func,
    VWTraceMessageFunc* trace_listener, void* trace_context, VWWorkspace** output_handle,
    VWErrorInfo* err_info_container) noexcept
{
  return VW_not_implemented;
}

VW_DLL_PUBLIC VWStatus vw_create_worspace_with_seed(const VWWorkspace* existing_workspace_handle,
    VWOptions* extra_options_handle, VWTraceMessageFunc* trace_listener, void* trace_context,
    VWWorkspace** output_handle, VWErrorInfo* err_info_container) noexcept
{
  return VW_not_implemented;
}

VW_DLL_PUBLIC VWStatus vw_workspace_get_model_id(
    const VWWorkspace* workspace_handle, const char** model_id, VWErrorInfo* err_info_container) noexcept
{
  return VW_not_implemented;
}

VW_DLL_PUBLIC VWStatus vw_workspace_set_model_id(
    VWWorkspace* workspace_handle, const char* model_id, VWErrorInfo* err_info_container) noexcept
{
  return VW_not_implemented;
}

VW_DLL_PUBLIC VWStatus vw_workspace_get_command_line(
    const VWWorkspace* workspace_handle, VWString* command_line, VWErrorInfo* err_info_container) noexcept
{
  return VW_not_implemented;
}

VW_DLL_PUBLIC VWStatus vw_workspace_finish(VWWorkspace* workspace_handle, VWErrorInfo* err_info_container) noexcept
{
  return VW_not_implemented;
}

VW_DLL_PUBLIC VWStatus vw_destroy_workspace(VWWorkspace* workspace_handle, VWErrorInfo* err_info_container) noexcept
{
  return VW_not_implemented;
}

VW_DLL_PUBLIC VWStatus vw_workspace_get_prediction_type(
    const VWWorkspace* workspace_handle, VWPredictionType* prediction_type, VWErrorInfo* err_info_container) noexcept
{
  return VW_not_implemented;
}

VW_DLL_PUBLIC VWStatus vw_workspace_get_label_type(
    const VWWorkspace* workspace_handle, VWLabelType* label_type, VWErrorInfo* err_info_container) noexcept
{
  return VW_not_implemented;
}

VW_DLL_PUBLIC VWStatus vw_workspace_learn_legacy(
    VWWorkspace* workspace_handle, VWExample* example_handle, VWErrorInfo* err_info_container) noexcept
{
  return VW_not_implemented;
}

VW_DLL_PUBLIC VWStatus vw_workspace_learn_multiline_legacy(VWWorkspace* workspace_handle,
    VWExample** example_handle_list, size_t example_handle_list_length, VWErrorInfo* err_info_container) noexcept
{
  return VW_not_implemented;
}

VW_DLL_PUBLIC VWStatus vw_workspace_predict_legacy(
    VWWorkspace* workspace_handle, VWExample* example_handle, VWErrorInfo* err_info_container) noexcept
{
  return VW_not_implemented;
}

VW_DLL_PUBLIC VWStatus vw_workspace_predict_multiline_legacy(VWWorkspace* workspace_handle,
    VWExample* example_handle_list, size_t example_handle_list_length, VWErrorInfo* err_info_container) noexcept
{
  return VW_not_implemented;
}

VW_DLL_PUBLIC VWStatus vw_workspace_finish_example(
    VWWorkspace* workspace_handle, VWExample* example_handle, VWErrorInfo* err_info_container) noexcept
{
  return VW_not_implemented;
}

VW_DLL_PUBLIC VWStatus vw_workspace_finish_example_multiline(VWWorkspace* workspace_handle,
    VWExample* example_handle_list, size_t example_handle_list_length, VWErrorInfo* err_info_container) noexcept
{
  return VW_not_implemented;
}

VW_DLL_PUBLIC VWStatus vw_workspace_end_pass(VWWorkspace* workspace_handle, VWErrorInfo* err_info_container) noexcept
{
  return VW_not_implemented;
}

VW_DLL_PUBLIC VWStatus vw_workspace_get_search(
    const VWWorkspace* workspace_handle, VWSearch** search_handle, VWErrorInfo* err_info_container) noexcept
{
  return VW_not_implemented;
}
