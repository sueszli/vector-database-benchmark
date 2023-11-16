from google.protobuf.descriptor import FieldDescriptor
from parameterized import parameterized
from streamlit.proto.Alert_pb2 import Alert
from streamlit.proto.AppPage_pb2 import AppPage
from streamlit.proto.Common_pb2 import FileURLs, FileURLsRequest, FileURLsResponse
from streamlit.proto.Exception_pb2 import Exception as Exception_
from streamlit.proto.NewSession_pb2 import Config, CustomThemeConfig, EnvironmentInfo, FontFace, FontSizes, Initialize, NewSession, Radii, UserInfo
from streamlit.proto.ParentMessage_pb2 import ParentMessage
from streamlit.proto.SessionStatus_pb2 import SessionStatus
FD = FieldDescriptor

@parameterized.expand([(AppPage, {('page_script_hash', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('page_name', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('icon', FD.LABEL_OPTIONAL, FD.TYPE_STRING)}), (NewSession, {('initialize', FD.LABEL_OPTIONAL, FD.TYPE_MESSAGE), ('script_run_id', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('name', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('main_script_path', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('config', FD.LABEL_OPTIONAL, FD.TYPE_MESSAGE), ('custom_theme', FD.LABEL_OPTIONAL, FD.TYPE_MESSAGE), ('app_pages', FD.LABEL_REPEATED, FD.TYPE_MESSAGE), ('page_script_hash', FD.LABEL_OPTIONAL, FD.TYPE_STRING)}), (Initialize, {('user_info', FD.LABEL_OPTIONAL, FD.TYPE_MESSAGE), ('environment_info', FD.LABEL_OPTIONAL, FD.TYPE_MESSAGE), ('session_status', FD.LABEL_OPTIONAL, FD.TYPE_MESSAGE), ('command_line', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('session_id', FD.LABEL_OPTIONAL, FD.TYPE_STRING)}), (Config, {('gather_usage_stats', FD.LABEL_OPTIONAL, FD.TYPE_BOOL), ('max_cached_message_age', FD.LABEL_OPTIONAL, FD.TYPE_INT32), ('mapbox_token', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('allow_run_on_save', FD.LABEL_OPTIONAL, FD.TYPE_BOOL), ('hide_top_bar', FD.LABEL_OPTIONAL, FD.TYPE_BOOL), ('hide_sidebar_nav', FD.LABEL_OPTIONAL, FD.TYPE_BOOL), ('toolbar_mode', FD.LABEL_OPTIONAL, FD.TYPE_ENUM)}), (CustomThemeConfig, {('primary_color', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('secondary_background_color', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('background_color', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('text_color', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('font', FD.LABEL_OPTIONAL, FD.TYPE_ENUM), ('base', FD.LABEL_OPTIONAL, FD.TYPE_ENUM), ('widget_background_color', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('widget_border_color', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('radii', FD.LABEL_OPTIONAL, FD.TYPE_MESSAGE), ('body_font', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('code_font', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('font_faces', FD.LABEL_REPEATED, FD.TYPE_MESSAGE), ('font_sizes', FD.LABEL_OPTIONAL, FD.TYPE_MESSAGE)}), (FontFace, {('url', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('family', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('weight', FD.LABEL_OPTIONAL, FD.TYPE_INT32), ('style', FD.LABEL_OPTIONAL, FD.TYPE_STRING)}), (Radii, {('base_widget_radius', FD.LABEL_OPTIONAL, FD.TYPE_INT32), ('checkbox_radius', FD.LABEL_OPTIONAL, FD.TYPE_INT32)}), (FontSizes, {('tiny_font_size', FD.LABEL_OPTIONAL, FD.TYPE_INT32), ('small_font_size', FD.LABEL_OPTIONAL, FD.TYPE_INT32), ('base_font_size', FD.LABEL_OPTIONAL, FD.TYPE_INT32)}), (UserInfo, {('installation_id', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('installation_id_v3', FD.LABEL_OPTIONAL, FD.TYPE_STRING)}), (EnvironmentInfo, {('streamlit_version', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('python_version', FD.LABEL_OPTIONAL, FD.TYPE_STRING)}), (SessionStatus, {('run_on_save', FD.LABEL_OPTIONAL, FD.TYPE_BOOL), ('script_is_running', FD.LABEL_OPTIONAL, FD.TYPE_BOOL)})])
def test_new_session_protos_stable(proto_class, expected_fields):
    if False:
        i = 10
        return i + 15
    d = proto_class.DESCRIPTOR
    assert {(f.name, f.label, f.type) for f in d.fields} == expected_fields

@parameterized.expand([(FileURLsRequest, {('request_id', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('file_names', FD.LABEL_REPEATED, FD.TYPE_STRING), ('session_id', FD.LABEL_OPTIONAL, FD.TYPE_STRING)}), (FileURLs, {('file_id', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('upload_url', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('delete_url', FD.LABEL_OPTIONAL, FD.TYPE_STRING)}), (FileURLsResponse, {('response_id', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('file_urls', FD.LABEL_REPEATED, FD.TYPE_MESSAGE), ('error_msg', FD.LABEL_OPTIONAL, FD.TYPE_STRING)})])
def test_file_uploader_protos_stable(proto_class, expected_fields):
    if False:
        i = 10
        return i + 15
    d = proto_class.DESCRIPTOR
    assert {(f.name, f.label, f.type) for f in d.fields} == expected_fields

def test_alert_proto_stable():
    if False:
        i = 10
        return i + 15
    d = Alert.DESCRIPTOR
    assert {(f.name, f.label, f.type) for f in d.fields} == {('body', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('format', FD.LABEL_OPTIONAL, FD.TYPE_ENUM), ('icon', FD.LABEL_OPTIONAL, FD.TYPE_STRING)}

def test_exception_proto_stable():
    if False:
        for i in range(10):
            print('nop')
    d = Exception_.DESCRIPTOR
    assert {(f.name, f.label, f.type) for f in d.fields} == {('type', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('message', FD.LABEL_OPTIONAL, FD.TYPE_STRING), ('message_is_markdown', FD.LABEL_OPTIONAL, FD.TYPE_BOOL), ('stack_trace', FD.LABEL_REPEATED, FD.TYPE_STRING), ('is_warning', FD.LABEL_OPTIONAL, FD.TYPE_BOOL)}

def test_parent_message_proto_stable():
    if False:
        while True:
            i = 10
    d = ParentMessage.DESCRIPTOR
    assert {(f.name, f.label, f.type) for f in d.fields} == {('message', FD.LABEL_OPTIONAL, FD.TYPE_STRING)}