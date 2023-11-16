import logging
import socket
import sys
from binascii import unhexlify
from Cryptodome.Cipher import ARC4
from impacket import ntlm, LOG
from impacket.structure import Structure, pack, unpack
from impacket.krb5 import kerberosv5, gssapi
from impacket.uuid import uuidtup_to_bin, generate, stringver_to_bin, bin_to_uuidtup
from impacket.dcerpc.v5.dtypes import UCHAR, ULONG, USHORT
from impacket.dcerpc.v5.ndr import NDRSTRUCT
from impacket import hresult_errors
from threading import Thread
MSRPC_REQUEST = 0
MSRPC_PING = 1
MSRPC_RESPONSE = 2
MSRPC_FAULT = 3
MSRPC_WORKING = 4
MSRPC_NOCALL = 5
MSRPC_REJECT = 6
MSRPC_ACK = 7
MSRPC_CL_CANCEL = 8
MSRPC_FACK = 9
MSRPC_CANCELACK = 10
MSRPC_BIND = 11
MSRPC_BINDACK = 12
MSRPC_BINDNAK = 13
MSRPC_ALTERCTX = 14
MSRPC_ALTERCTX_R = 15
MSRPC_AUTH3 = 16
MSRPC_SHUTDOWN = 17
MSRPC_CO_CANCEL = 18
MSRPC_ORPHANED = 19
MSRPC_RTS = 20
PFC_FIRST_FRAG = 1
PFC_LAST_FRAG = 2
MSRPC_SUPPORT_SIGN = 4
MSRPC_PENDING_CANCEL = 4
PFC_RESERVED_1 = 8
PFC_CONC_MPX = 16
PFC_DID_NOT_EXECUTE = 32
PFC_MAYBE = 64
PFC_OBJECT_UUID = 128
RPC_C_AUTHN_NONE = 0
RPC_C_AUTHN_GSS_NEGOTIATE = 9
RPC_C_AUTHN_WINNT = 10
RPC_C_AUTHN_GSS_SCHANNEL = 14
RPC_C_AUTHN_GSS_KERBEROS = 16
RPC_C_AUTHN_NETLOGON = 68
RPC_C_AUTHN_DEFAULT = 255
RPC_C_AUTHN_LEVEL_NONE = 1
RPC_C_AUTHN_LEVEL_CONNECT = 2
RPC_C_AUTHN_LEVEL_CALL = 3
RPC_C_AUTHN_LEVEL_PKT = 4
RPC_C_AUTHN_LEVEL_PKT_INTEGRITY = 5
RPC_C_AUTHN_LEVEL_PKT_PRIVACY = 6
rpc_provider_reason = {0: 'reason_not_specified', 1: 'abstract_syntax_not_supported', 2: 'proposed_transfer_syntaxes_not_supported', 3: 'local_limit_exceeded', 4: 'protocol_version_not_specified', 8: 'authentication_type_not_recognized', 9: 'invalid_checksum'}
MSRPC_CONT_RESULT_ACCEPT = 0
MSRPC_CONT_RESULT_USER_REJECT = 1
MSRPC_CONT_RESULT_PROV_REJECT = 2
rpc_cont_def_result = {0: 'acceptance', 1: 'user_rejection', 2: 'provider_rejection'}
rpc_status_codes = {5: 'rpc_s_access_denied', 8: 'Authentication type not recognized', 1752: 'rpc_fault_cant_perform', 1734: 'rpc_x_invalid_bound', 1764: 'rpc_s_cannot_support: The requested operation is not supported.', 1783: 'rpc_x_bad_stub_data', 469827585: 'nca_s_comm_failure', 469827586: 'nca_s_op_rng_error', 469827587: 'nca_s_unk_if', 469827590: 'nca_s_wrong_boot_time', 469827593: 'nca_s_you_crashed', 469827595: 'nca_s_proto_error', 469827603: 'nca_s_out_args_too_big ', 469827604: 'nca_s_server_too_busy', 469827605: 'nca_s_fault_string_too_long', 469827607: 'nca_s_unsupported_type ', 469762049: 'nca_s_fault_int_div_by_zero', 469762050: 'nca_s_fault_addr_error ', 469762051: 'nca_s_fault_fp_div_zero', 469762052: 'nca_s_fault_fp_underflow', 469762053: 'nca_s_fault_fp_overflow', 469762054: 'nca_s_fault_invalid_tag', 469762055: 'nca_s_fault_invalid_bound ', 469762056: 'nca_s_rpc_version_mismatch', 469762057: 'nca_s_unspec_reject ', 469762058: 'nca_s_bad_actid', 469762059: 'nca_s_who_are_you_failed', 469762060: 'nca_s_manager_not_entered ', 469762061: 'nca_s_fault_cancel', 469762062: 'nca_s_fault_ill_inst', 469762063: 'nca_s_fault_fp_error', 469762064: 'nca_s_fault_int_overflow', 469762066: 'nca_s_fault_unspec', 469762067: 'nca_s_fault_remote_comm_failure ', 469762068: 'nca_s_fault_pipe_empty ', 469762069: 'nca_s_fault_pipe_closed', 469762070: 'nca_s_fault_pipe_order ', 469762071: 'nca_s_fault_pipe_discipline', 469762072: 'nca_s_fault_pipe_comm_error', 469762073: 'nca_s_fault_pipe_memory', 469762074: 'nca_s_fault_context_mismatch ', 469762075: 'nca_s_fault_remote_no_memory ', 469762076: 'nca_s_invalid_pres_context_id', 469762077: 'nca_s_unsupported_authn_level', 469762079: 'nca_s_invalid_checksum ', 469762080: 'nca_s_invalid_crc', 469762081: 'nca_s_fault_user_defined', 469762082: 'nca_s_fault_tx_open_failed', 469762083: 'nca_s_fault_codeset_conv_error', 469762084: 'nca_s_fault_object_not_found ', 469762085: 'nca_s_fault_no_client_stub', 382312448: 'rpc_s_mod', 382312449: 'rpc_s_op_rng_error', 382312450: 'rpc_s_cant_create_socket', 382312451: 'rpc_s_cant_bind_socket', 382312452: 'rpc_s_not_in_call', 382312453: 'rpc_s_no_port', 382312454: 'rpc_s_wrong_boot_time', 382312455: 'rpc_s_too_many_sockets', 382312456: 'rpc_s_illegal_register', 382312457: 'rpc_s_cant_recv', 382312458: 'rpc_s_bad_pkt', 382312459: 'rpc_s_unbound_handle', 382312460: 'rpc_s_addr_in_use', 382312461: 'rpc_s_in_args_too_big', 382312462: 'rpc_s_string_too_long', 382312463: 'rpc_s_too_many_objects', 382312464: 'rpc_s_binding_has_no_auth', 382312465: 'rpc_s_unknown_authn_service', 382312466: 'rpc_s_no_memory', 382312467: 'rpc_s_cant_nmalloc', 382312468: 'rpc_s_call_faulted', 382312469: 'rpc_s_call_failed', 382312470: 'rpc_s_comm_failure', 382312471: 'rpc_s_rpcd_comm_failure', 382312472: 'rpc_s_illegal_family_rebind', 382312473: 'rpc_s_invalid_handle', 382312474: 'rpc_s_coding_error', 382312475: 'rpc_s_object_not_found', 382312476: 'rpc_s_cthread_not_found', 382312477: 'rpc_s_invalid_binding', 382312478: 'rpc_s_already_registered', 382312479: 'rpc_s_endpoint_not_found', 382312480: 'rpc_s_invalid_rpc_protseq', 382312481: 'rpc_s_desc_not_registered', 382312482: 'rpc_s_already_listening', 382312483: 'rpc_s_no_protseqs', 382312484: 'rpc_s_no_protseqs_registered', 382312485: 'rpc_s_no_bindings', 382312486: 'rpc_s_max_descs_exceeded', 382312487: 'rpc_s_no_interfaces', 382312488: 'rpc_s_invalid_timeout', 382312489: 'rpc_s_cant_inq_socket', 382312490: 'rpc_s_invalid_naf_id', 382312491: 'rpc_s_inval_net_addr', 382312492: 'rpc_s_unknown_if', 382312493: 'rpc_s_unsupported_type', 382312494: 'rpc_s_invalid_call_opt', 382312495: 'rpc_s_no_fault', 382312496: 'rpc_s_cancel_timeout', 382312497: 'rpc_s_call_cancelled', 382312498: 'rpc_s_invalid_call_handle', 382312499: 'rpc_s_cannot_alloc_assoc', 382312500: 'rpc_s_cannot_connect', 382312501: 'rpc_s_connection_aborted', 382312502: 'rpc_s_connection_closed', 382312503: 'rpc_s_cannot_accept', 382312504: 'rpc_s_assoc_grp_not_found', 382312505: 'rpc_s_stub_interface_error', 382312506: 'rpc_s_invalid_object', 382312507: 'rpc_s_invalid_type', 382312508: 'rpc_s_invalid_if_opnum', 382312509: 'rpc_s_different_server_instance', 382312510: 'rpc_s_protocol_error', 382312511: 'rpc_s_cant_recvmsg', 382312512: 'rpc_s_invalid_string_binding', 382312513: 'rpc_s_connect_timed_out', 382312514: 'rpc_s_connect_rejected', 382312515: 'rpc_s_network_unreachable', 382312516: 'rpc_s_connect_no_resources', 382312517: 'rpc_s_rem_network_shutdown', 382312518: 'rpc_s_too_many_rem_connects', 382312519: 'rpc_s_no_rem_endpoint', 382312520: 'rpc_s_rem_host_down', 382312521: 'rpc_s_host_unreachable', 382312522: 'rpc_s_access_control_info_inv', 382312523: 'rpc_s_loc_connect_aborted', 382312524: 'rpc_s_connect_closed_by_rem', 382312525: 'rpc_s_rem_host_crashed', 382312526: 'rpc_s_invalid_endpoint_format', 382312527: 'rpc_s_unknown_status_code', 382312528: 'rpc_s_unknown_mgr_type', 382312529: 'rpc_s_assoc_creation_failed', 382312530: 'rpc_s_assoc_grp_max_exceeded', 382312531: 'rpc_s_assoc_grp_alloc_failed', 382312532: 'rpc_s_sm_invalid_state', 382312533: 'rpc_s_assoc_req_rejected', 382312534: 'rpc_s_assoc_shutdown', 382312535: 'rpc_s_tsyntaxes_unsupported', 382312536: 'rpc_s_context_id_not_found', 382312537: 'rpc_s_cant_listen_socket', 382312538: 'rpc_s_no_addrs', 382312539: 'rpc_s_cant_getpeername', 382312540: 'rpc_s_cant_get_if_id', 382312541: 'rpc_s_protseq_not_supported', 382312542: 'rpc_s_call_orphaned', 382312543: 'rpc_s_who_are_you_failed', 382312544: 'rpc_s_unknown_reject', 382312545: 'rpc_s_type_already_registered', 382312546: 'rpc_s_stop_listening_disabled', 382312547: 'rpc_s_invalid_arg', 382312548: 'rpc_s_not_supported', 382312549: 'rpc_s_wrong_kind_of_binding', 382312550: 'rpc_s_authn_authz_mismatch', 382312551: 'rpc_s_call_queued', 382312552: 'rpc_s_cannot_set_nodelay', 382312553: 'rpc_s_not_rpc_tower', 382312554: 'rpc_s_invalid_rpc_protid', 382312555: 'rpc_s_invalid_rpc_floor', 382312556: 'rpc_s_call_timeout', 382312557: 'rpc_s_mgmt_op_disallowed', 382312558: 'rpc_s_manager_not_entered', 382312559: 'rpc_s_calls_too_large_for_wk_ep', 382312560: 'rpc_s_server_too_busy', 382312561: 'rpc_s_prot_version_mismatch', 382312562: 'rpc_s_rpc_prot_version_mismatch', 382312563: 'rpc_s_ss_no_import_cursor', 382312564: 'rpc_s_fault_addr_error', 382312565: 'rpc_s_fault_context_mismatch', 382312566: 'rpc_s_fault_fp_div_by_zero', 382312567: 'rpc_s_fault_fp_error', 382312568: 'rpc_s_fault_fp_overflow', 382312569: 'rpc_s_fault_fp_underflow', 382312570: 'rpc_s_fault_ill_inst', 382312571: 'rpc_s_fault_int_div_by_zero', 382312572: 'rpc_s_fault_int_overflow', 382312573: 'rpc_s_fault_invalid_bound', 382312574: 'rpc_s_fault_invalid_tag', 382312575: 'rpc_s_fault_pipe_closed', 382312576: 'rpc_s_fault_pipe_comm_error', 382312577: 'rpc_s_fault_pipe_discipline', 382312578: 'rpc_s_fault_pipe_empty', 382312579: 'rpc_s_fault_pipe_memory', 382312580: 'rpc_s_fault_pipe_order', 382312581: 'rpc_s_fault_remote_comm_failure', 382312582: 'rpc_s_fault_remote_no_memory', 382312583: 'rpc_s_fault_unspec', 382312584: 'uuid_s_bad_version', 382312585: 'uuid_s_socket_failure', 382312586: 'uuid_s_getconf_failure', 382312587: 'uuid_s_no_address', 382312588: 'uuid_s_overrun', 382312589: 'uuid_s_internal_error', 382312590: 'uuid_s_coding_error', 382312591: 'uuid_s_invalid_string_uuid', 382312592: 'uuid_s_no_memory', 382312593: 'rpc_s_no_more_entries', 382312594: 'rpc_s_unknown_ns_error', 382312595: 'rpc_s_name_service_unavailable', 382312596: 'rpc_s_incomplete_name', 382312597: 'rpc_s_group_not_found', 382312598: 'rpc_s_invalid_name_syntax', 382312599: 'rpc_s_no_more_members', 382312600: 'rpc_s_no_more_interfaces', 382312601: 'rpc_s_invalid_name_service', 382312602: 'rpc_s_no_name_mapping', 382312603: 'rpc_s_profile_not_found', 382312604: 'rpc_s_not_found', 382312605: 'rpc_s_no_updates', 382312606: 'rpc_s_update_failed', 382312607: 'rpc_s_no_match_exported', 382312608: 'rpc_s_entry_not_found', 382312609: 'rpc_s_invalid_inquiry_context', 382312610: 'rpc_s_interface_not_found', 382312611: 'rpc_s_group_member_not_found', 382312612: 'rpc_s_entry_already_exists', 382312613: 'rpc_s_nsinit_failure', 382312614: 'rpc_s_unsupported_name_syntax', 382312615: 'rpc_s_no_more_elements', 382312616: 'rpc_s_no_ns_permission', 382312617: 'rpc_s_invalid_inquiry_type', 382312618: 'rpc_s_profile_element_not_found', 382312619: 'rpc_s_profile_element_replaced', 382312620: 'rpc_s_import_already_done', 382312621: 'rpc_s_database_busy', 382312622: 'rpc_s_invalid_import_context', 382312623: 'rpc_s_uuid_set_not_found', 382312624: 'rpc_s_uuid_member_not_found', 382312625: 'rpc_s_no_interfaces_exported', 382312626: 'rpc_s_tower_set_not_found', 382312627: 'rpc_s_tower_member_not_found', 382312628: 'rpc_s_obj_uuid_not_found', 382312629: 'rpc_s_no_more_bindings', 382312630: 'rpc_s_invalid_priority', 382312631: 'rpc_s_not_rpc_entry', 382312632: 'rpc_s_invalid_lookup_context', 382312633: 'rpc_s_binding_vector_full', 382312634: 'rpc_s_cycle_detected', 382312635: 'rpc_s_nothing_to_export', 382312636: 'rpc_s_nothing_to_unexport', 382312637: 'rpc_s_invalid_vers_option', 382312638: 'rpc_s_no_rpc_data', 382312639: 'rpc_s_mbr_picked', 382312640: 'rpc_s_not_all_objs_unexported', 382312641: 'rpc_s_no_entry_name', 382312642: 'rpc_s_priority_group_done', 382312643: 'rpc_s_partial_results', 382312644: 'rpc_s_no_env_setup', 382312645: 'twr_s_unknown_sa', 382312646: 'twr_s_unknown_tower', 382312647: 'twr_s_not_implemented', 382312648: 'rpc_s_max_calls_too_small', 382312649: 'rpc_s_cthread_create_failed', 382312650: 'rpc_s_cthread_pool_exists', 382312651: 'rpc_s_cthread_no_such_pool', 382312652: 'rpc_s_cthread_invoke_disabled', 382312653: 'ept_s_cant_perform_op', 382312654: 'ept_s_no_memory', 382312655: 'ept_s_database_invalid', 382312656: 'ept_s_cant_create', 382312657: 'ept_s_cant_access', 382312658: 'ept_s_database_already_open', 382312659: 'ept_s_invalid_entry', 382312660: 'ept_s_update_failed', 382312661: 'ept_s_invalid_context', 382312662: 'ept_s_not_registered', 382312663: 'ept_s_server_unavailable', 382312664: 'rpc_s_underspecified_name', 382312665: 'rpc_s_invalid_ns_handle', 382312666: 'rpc_s_unknown_error', 382312667: 'rpc_s_ss_char_trans_open_fail', 382312668: 'rpc_s_ss_char_trans_short_file', 382312669: 'rpc_s_ss_context_damaged', 382312670: 'rpc_s_ss_in_null_context', 382312671: 'rpc_s_socket_failure', 382312672: 'rpc_s_unsupported_protect_level', 382312673: 'rpc_s_invalid_checksum', 382312674: 'rpc_s_invalid_credentials', 382312675: 'rpc_s_credentials_too_large', 382312676: 'rpc_s_call_id_not_found', 382312677: 'rpc_s_key_id_not_found', 382312678: 'rpc_s_auth_bad_integrity', 382312679: 'rpc_s_auth_tkt_expired', 382312680: 'rpc_s_auth_tkt_nyv', 382312681: 'rpc_s_auth_repeat', 382312682: 'rpc_s_auth_not_us', 382312683: 'rpc_s_auth_badmatch', 382312684: 'rpc_s_auth_skew', 382312685: 'rpc_s_auth_badaddr', 382312686: 'rpc_s_auth_badversion', 382312687: 'rpc_s_auth_msg_type', 382312688: 'rpc_s_auth_modified', 382312689: 'rpc_s_auth_badorder', 382312690: 'rpc_s_auth_badkeyver', 382312691: 'rpc_s_auth_nokey', 382312692: 'rpc_s_auth_mut_fail', 382312693: 'rpc_s_auth_baddirection', 382312694: 'rpc_s_auth_method', 382312695: 'rpc_s_auth_badseq', 382312696: 'rpc_s_auth_inapp_cksum', 382312697: 'rpc_s_auth_field_toolong', 382312698: 'rpc_s_invalid_crc', 382312699: 'rpc_s_binding_incomplete', 382312700: 'rpc_s_key_func_not_allowed', 382312701: 'rpc_s_unknown_stub_rtl_if_vers', 382312702: 'rpc_s_unknown_ifspec_vers', 382312703: 'rpc_s_proto_unsupp_by_auth', 382312704: 'rpc_s_authn_challenge_malformed', 382312705: 'rpc_s_protect_level_mismatch', 382312706: 'rpc_s_no_mepv', 382312707: 'rpc_s_stub_protocol_error', 382312708: 'rpc_s_class_version_mismatch', 382312709: 'rpc_s_helper_not_running', 382312710: 'rpc_s_helper_short_read', 382312711: 'rpc_s_helper_catatonic', 382312712: 'rpc_s_helper_aborted', 382312713: 'rpc_s_not_in_kernel', 382312714: 'rpc_s_helper_wrong_user', 382312715: 'rpc_s_helper_overflow', 382312716: 'rpc_s_dg_need_way_auth', 382312717: 'rpc_s_unsupported_auth_subtype', 382312718: 'rpc_s_wrong_pickle_type', 382312719: 'rpc_s_not_listening', 382312720: 'rpc_s_ss_bad_buffer', 382312721: 'rpc_s_ss_bad_es_action', 382312722: 'rpc_s_ss_wrong_es_version', 382312723: 'rpc_s_fault_user_defined', 382312724: 'rpc_s_ss_incompatible_codesets', 382312725: 'rpc_s_tx_not_in_transaction', 382312726: 'rpc_s_tx_open_failed', 382312727: 'rpc_s_partial_credentials', 382312728: 'rpc_s_ss_invalid_codeset_tag', 382312729: 'rpc_s_mgmt_bad_type', 382312730: 'rpc_s_ss_invalid_char_input', 382312731: 'rpc_s_ss_short_conv_buffer', 382312732: 'rpc_s_ss_iconv_error', 382312733: 'rpc_s_ss_no_compat_codeset', 382312734: 'rpc_s_ss_no_compat_charsets', 382312735: 'dce_cs_c_ok', 382312736: 'dce_cs_c_unknown', 382312737: 'dce_cs_c_notfound', 382312738: 'dce_cs_c_cannot_open_file', 382312739: 'dce_cs_c_cannot_read_file', 382312740: 'dce_cs_c_cannot_allocate_memory', 382312741: 'rpc_s_ss_cleanup_failed', 382312742: 'rpc_svc_desc_general', 382312743: 'rpc_svc_desc_mutex', 382312744: 'rpc_svc_desc_xmit', 382312745: 'rpc_svc_desc_recv', 382312746: 'rpc_svc_desc_dg_state', 382312747: 'rpc_svc_desc_cancel', 382312748: 'rpc_svc_desc_orphan', 382312749: 'rpc_svc_desc_cn_state', 382312750: 'rpc_svc_desc_cn_pkt', 382312751: 'rpc_svc_desc_pkt_quotas', 382312752: 'rpc_svc_desc_auth', 382312753: 'rpc_svc_desc_source', 382312754: 'rpc_svc_desc_stats', 382312755: 'rpc_svc_desc_mem', 382312756: 'rpc_svc_desc_mem_type', 382312757: 'rpc_svc_desc_dg_pktlog', 382312758: 'rpc_svc_desc_thread_id', 382312759: 'rpc_svc_desc_timestamp', 382312760: 'rpc_svc_desc_cn_errors', 382312761: 'rpc_svc_desc_conv_thread', 382312762: 'rpc_svc_desc_pid', 382312763: 'rpc_svc_desc_atfork', 382312764: 'rpc_svc_desc_cma_thread', 382312765: 'rpc_svc_desc_inherit', 382312766: 'rpc_svc_desc_dg_sockets', 382312767: 'rpc_svc_desc_timer', 382312768: 'rpc_svc_desc_threads', 382312769: 'rpc_svc_desc_server_call', 382312770: 'rpc_svc_desc_nsi', 382312771: 'rpc_svc_desc_dg_pkt', 382312772: 'rpc_m_cn_ill_state_trans_sa', 382312773: 'rpc_m_cn_ill_state_trans_ca', 382312774: 'rpc_m_cn_ill_state_trans_sg', 382312775: 'rpc_m_cn_ill_state_trans_cg', 382312776: 'rpc_m_cn_ill_state_trans_sr', 382312777: 'rpc_m_cn_ill_state_trans_cr', 382312778: 'rpc_m_bad_pkt_type', 382312779: 'rpc_m_prot_mismatch', 382312780: 'rpc_m_frag_toobig', 382312781: 'rpc_m_unsupp_stub_rtl_if', 382312782: 'rpc_m_unhandled_callstate', 382312783: 'rpc_m_call_failed', 382312784: 'rpc_m_call_failed_no_status', 382312785: 'rpc_m_call_failed_errno', 382312786: 'rpc_m_call_failed_s', 382312787: 'rpc_m_call_failed_c', 382312788: 'rpc_m_errmsg_toobig', 382312789: 'rpc_m_invalid_srchattr', 382312790: 'rpc_m_nts_not_found', 382312791: 'rpc_m_invalid_accbytcnt', 382312792: 'rpc_m_pre_v2_ifspec', 382312793: 'rpc_m_unk_ifspec', 382312794: 'rpc_m_recvbuf_toosmall', 382312795: 'rpc_m_unalign_authtrl', 382312796: 'rpc_m_unexpected_exc', 382312797: 'rpc_m_no_stub_data', 382312798: 'rpc_m_eventlist_full', 382312799: 'rpc_m_unk_sock_type', 382312800: 'rpc_m_unimp_call', 382312801: 'rpc_m_invalid_seqnum', 382312802: 'rpc_m_cant_create_uuid', 382312803: 'rpc_m_pre_v2_ss', 382312804: 'rpc_m_dgpkt_pool_corrupt', 382312805: 'rpc_m_dgpkt_bad_free', 382312806: 'rpc_m_lookaside_corrupt', 382312807: 'rpc_m_alloc_fail', 382312808: 'rpc_m_realloc_fail', 382312809: 'rpc_m_cant_open_file', 382312810: 'rpc_m_cant_read_addr', 382312811: 'rpc_svc_desc_libidl', 382312812: 'rpc_m_ctxrundown_nomem', 382312813: 'rpc_m_ctxrundown_exc', 382312814: 'rpc_s_fault_codeset_conv_error', 382312815: 'rpc_s_no_call_active', 382312816: 'rpc_s_cannot_support', 382312817: 'rpc_s_no_context_available'}

class DCERPCException(Exception):
    """
    This is the exception every client should catch regardless of the underlying
    DCERPC Transport used.
    """

    def __init__(self, error_string=None, error_code=None, packet=None):
        if False:
            while True:
                i = 10
        "\n        :param string error_string: A string you want to show explaining the exception. Otherwise the default ones will be used\n        :param integer error_code: the error_code if we're using a dictionary with error's descriptions\n        :param NDR packet: if successfully decoded, the NDR packet of the response call. This could probably have useful\n        information\n        "
        Exception.__init__(self)
        self.packet = packet
        self.error_string = error_string
        if packet is not None:
            try:
                self.error_code = packet['ErrorCode']
            except:
                self.error_code = error_code
        else:
            self.error_code = error_code

    def get_error_code(self):
        if False:
            for i in range(10):
                print('nop')
        return self.error_code

    def get_packet(self):
        if False:
            for i in range(10):
                print('nop')
        return self.packet

    def __str__(self):
        if False:
            i = 10
            return i + 15
        key = self.error_code
        if self.error_string is not None:
            return self.error_string
        if key in rpc_status_codes:
            error_msg_short = rpc_status_codes[key]
            return 'DCERPC Runtime Error: code: 0x%x - %s ' % (self.error_code, error_msg_short)
        else:
            return 'DCERPC Runtime Error: unknown error code: 0x%x' % self.error_code

class CtxItem(Structure):
    structure = (('ContextID', '<H=0'), ('TransItems', 'B=0'), ('Pad', 'B=0'), ('AbstractSyntax', '20s=""'), ('TransferSyntax', '20s=""'))

class CtxItemResult(Structure):
    structure = (('Result', '<H=0'), ('Reason', '<H=0'), ('TransferSyntax', '20s=""'))

class SEC_TRAILER(Structure):
    commonHdr = (('auth_type', 'B=10'), ('auth_level', 'B=0'), ('auth_pad_len', 'B=0'), ('auth_rsvrd', 'B=0'), ('auth_ctx_id', '<L=747920'))

class MSRPCHeader(Structure):
    _SIZE = 16
    commonHdr = (('ver_major', 'B=5'), ('ver_minor', 'B=0'), ('type', 'B=0'), ('flags', 'B=0'), ('representation', '<L=0x10'), ('frag_len', '<H=self._SIZE+len(auth_data)+(16 if (self["flags"] & 0x80) > 0 else 0)+len(pduData)+len(pad)+len(sec_trailer)'), ('auth_len', '<H=len(auth_data)'), ('call_id', '<L=1'))
    structure = (('dataLen', '_-pduData', 'self["frag_len"]-self["auth_len"]-self._SIZE-(8 if self["auth_len"] > 0 else 0)'), ('pduData', ':'), ('_pad', '_-pad', '(4 - ((self._SIZE + (16 if (self["flags"] & 0x80) > 0 else 0) + len(self["pduData"])) & 3) & 3)'), ('pad', ':'), ('_sec_trailer', '_-sec_trailer', '8 if self["auth_len"] > 0 else 0'), ('sec_trailer', ':'), ('auth_dataLen', '_-auth_data', 'self["auth_len"]'), ('auth_data', ':'))

    def __init__(self, data=None, alignment=0):
        if False:
            while True:
                i = 10
        Structure.__init__(self, data, alignment)
        if data is None:
            self['ver_major'] = 5
            self['ver_minor'] = 0
            self['flags'] = PFC_FIRST_FRAG | PFC_LAST_FRAG
            self['type'] = MSRPC_REQUEST
            self.__frag_len_set = 0
            self['auth_len'] = 0
            self['pduData'] = b''
            self['auth_data'] = b''
            self['sec_trailer'] = b''
            self['pad'] = b''

    def get_header_size(self):
        if False:
            i = 10
            return i + 15
        return self._SIZE + (16 if self['flags'] & PFC_OBJECT_UUID > 0 else 0)

    def get_packet(self):
        if False:
            while True:
                i = 10
        if self['auth_data'] != b'':
            self['auth_len'] = len(self['auth_data'])
        return self.getData()

class MSRPCRequestHeader(MSRPCHeader):
    _SIZE = 24
    commonHdr = MSRPCHeader.commonHdr + (('alloc_hint', '<L=0'), ('ctx_id', '<H=0'), ('op_num', '<H=0'), ('_uuid', '_-uuid', '16 if self["flags"] & 0x80 > 0 else 0'), ('uuid', ':'))

    def __init__(self, data=None, alignment=0):
        if False:
            return 10
        MSRPCHeader.__init__(self, data, alignment)
        if data is None:
            self['type'] = MSRPC_REQUEST
            self['ctx_id'] = 0
            self['uuid'] = b''

class MSRPCRespHeader(MSRPCHeader):
    _SIZE = 24
    commonHdr = MSRPCHeader.commonHdr + (('alloc_hint', '<L=0'), ('ctx_id', '<H=0'), ('cancel_count', '<B=0'), ('padding', '<B=0'))

    def __init__(self, aBuffer=None, alignment=0):
        if False:
            return 10
        MSRPCHeader.__init__(self, aBuffer, alignment)
        if aBuffer is None:
            self['type'] = MSRPC_RESPONSE
            self['ctx_id'] = 0

class MSRPCBind(Structure):
    _CTX_ITEM_LEN = len(CtxItem())
    structure = (('max_tfrag', '<H=4280'), ('max_rfrag', '<H=4280'), ('assoc_group', '<L=0'), ('ctx_num', 'B=0'), ('Reserved', 'B=0'), ('Reserved2', '<H=0'), ('_ctx_items', '_-ctx_items', 'self["ctx_num"]*self._CTX_ITEM_LEN'), ('ctx_items', ':'))

    def __init__(self, data=None, alignment=0):
        if False:
            return 10
        Structure.__init__(self, data, alignment)
        if data is None:
            self['max_tfrag'] = 4280
            self['max_rfrag'] = 4280
            self['assoc_group'] = 0
            self['ctx_num'] = 1
            self['ctx_items'] = b''
        self.__ctx_items = []

    def addCtxItem(self, item):
        if False:
            for i in range(10):
                print('nop')
        self.__ctx_items.append(item)

    def getData(self):
        if False:
            while True:
                i = 10
        self['ctx_num'] = len(self.__ctx_items)
        for i in self.__ctx_items:
            self['ctx_items'] += i.getData()
        return Structure.getData(self)

class MSRPCBindAck(MSRPCHeader):
    _SIZE = 26
    _CTX_ITEM_LEN = len(CtxItemResult())
    structure = (('max_tfrag', '<H=0'), ('max_rfrag', '<H=0'), ('assoc_group', '<L=0'), ('SecondaryAddrLen', '<H&SecondaryAddr'), ('SecondaryAddr', 'z'), ('PadLen', '_-Pad', '(4-((self["SecondaryAddrLen"]+self._SIZE) % 4))%4'), ('Pad', ':'), ('ctx_num', 'B=0'), ('Reserved', 'B=0'), ('Reserved2', '<H=0'), ('_ctx_items', '_-ctx_items', 'self["ctx_num"]*self._CTX_ITEM_LEN'), ('ctx_items', ':'), ('_sec_trailer', '_-sec_trailer', '8 if self["auth_len"] > 0 else 0'), ('sec_trailer', ':'), ('auth_dataLen', '_-auth_data', 'self["auth_len"]'), ('auth_data', ':'))

    def __init__(self, data=None, alignment=0):
        if False:
            i = 10
            return i + 15
        self.__ctx_items = []
        MSRPCHeader.__init__(self, data, alignment)
        if data is None:
            self['Pad'] = b''
            self['ctx_items'] = b''
            self['sec_trailer'] = b''
            self['auth_data'] = b''

    def getCtxItems(self):
        if False:
            print('Hello World!')
        return self.__ctx_items

    def getCtxItem(self, index):
        if False:
            return 10
        return self.__ctx_items[index - 1]

    def fromString(self, data):
        if False:
            i = 10
            return i + 15
        Structure.fromString(self, data)
        data = self['ctx_items']
        for i in range(self['ctx_num']):
            item = CtxItemResult(data)
            self.__ctx_items.append(item)
            data = data[len(item):]

class MSRPCBindNak(Structure):
    structure = (('RejectedReason', '<H=0'), ('SupportedVersions', ':'))

    def __init__(self, data=None, alignment=0):
        if False:
            return 10
        Structure.__init__(self, data, alignment)
        if data is None:
            self['SupportedVersions'] = b''

class DCERPC:
    NDRSyntax = uuidtup_to_bin(('8a885d04-1ceb-11c9-9fe8-08002b104860', '2.0'))
    NDR64Syntax = uuidtup_to_bin(('71710533-BEBA-4937-8319-B5DBEF9CCC36', '1.0'))
    transfer_syntax = NDRSyntax

    def __init__(self, transport):
        if False:
            print('Hello World!')
        self._transport = transport
        self.set_ctx_id(0)
        self._max_user_frag = None
        self.set_default_max_fragment_size()
        self._ctx = None

    def get_rpc_transport(self):
        if False:
            for i in range(10):
                print('nop')
        return self._transport

    def set_ctx_id(self, ctx_id):
        if False:
            print('Hello World!')
        self._ctx = ctx_id

    def connect(self):
        if False:
            for i in range(10):
                print('nop')
        return self._transport.connect()

    def disconnect(self):
        if False:
            return 10
        return self._transport.disconnect()

    def set_max_fragment_size(self, fragment_size):
        if False:
            i = 10
            return i + 15
        if fragment_size == -1:
            self.set_default_max_fragment_size()
        else:
            self._max_user_frag = fragment_size

    def set_default_max_fragment_size(self):
        if False:
            return 10
        self._max_user_frag = 0

    def send(self, data):
        if False:
            for i in range(10):
                print('nop')
        raise RuntimeError('virtual method. Not implemented in subclass')

    def recv(self):
        if False:
            return 10
        raise RuntimeError('virtual method. Not implemented in subclass')

    def alter_ctx(self, newUID, bogus_binds=''):
        if False:
            i = 10
            return i + 15
        raise RuntimeError('virtual method. Not implemented in subclass')

    def set_credentials(self, username, password, domain='', lmhash='', nthash='', aesKey='', TGT=None, TGS=None):
        if False:
            print('Hello World!')
        pass

    def set_auth_level(self, auth_level):
        if False:
            while True:
                i = 10
        pass

    def set_auth_type(self, auth_type, callback=None):
        if False:
            print('Hello World!')
        pass

    def get_idempotent(self):
        if False:
            print('Hello World!')
        return 0

    def set_idempotent(self, flag):
        if False:
            return 10
        pass

    def call(self, function, body, uuid=None):
        if False:
            while True:
                i = 10
        if hasattr(body, 'getData'):
            return self.send(DCERPC_RawCall(function, body.getData(), uuid))
        else:
            return self.send(DCERPC_RawCall(function, body, uuid))

    def request(self, request, uuid=None, checkError=True):
        if False:
            while True:
                i = 10
        if self.transfer_syntax == self.NDR64Syntax:
            request.changeTransferSyntax(self.NDR64Syntax)
            isNDR64 = True
        else:
            isNDR64 = False
        self.call(request.opnum, request, uuid)
        answer = self.recv()
        __import__(request.__module__)
        module = sys.modules[request.__module__]
        respClass = getattr(module, request.__class__.__name__ + 'Response')
        if answer[-4:] != b'\x00\x00\x00\x00' and checkError is True:
            error_code = unpack('<L', answer[-4:])[0]
            if error_code in rpc_status_codes:
                exception = DCERPCException(error_code=error_code)
            else:
                sessionErrorClass = getattr(module, 'DCERPCSessionError')
                try:
                    response = respClass(answer, isNDR64=isNDR64)
                except:
                    exception = sessionErrorClass(error_code=error_code)
                else:
                    exception = sessionErrorClass(packet=response, error_code=error_code)
            raise exception
        else:
            response = respClass(answer, isNDR64=isNDR64)
            return response

class DCERPC_v4(DCERPC):
    pass

class DCERPC_v5(DCERPC):

    def __init__(self, transport):
        if False:
            for i in range(10):
                print('nop')
        DCERPC.__init__(self, transport)
        self.__auth_level = RPC_C_AUTHN_LEVEL_NONE
        self.__auth_type = RPC_C_AUTHN_WINNT
        self.__auth_type_callback = None
        self.__auth_flags = 0
        self.__username = None
        self.__password = None
        self.__domain = ''
        self.__lmhash = ''
        self.__nthash = ''
        self.__aesKey = ''
        self.__TGT = None
        self.__TGS = None
        self.__clientSigningKey = b''
        self.__serverSigningKey = b''
        self.__clientSealingKey = b''
        self.__clientSealingHandle = b''
        self.__serverSealingKey = b''
        self.__serverSealingHandle = b''
        self.__sequence = 0
        self.transfer_syntax = uuidtup_to_bin(('8a885d04-1ceb-11c9-9fe8-08002b104860', '2.0'))
        self.__callid = 1
        self._ctx = 0
        self.__sessionKey = None
        self.__max_xmit_size = 0
        self.__flags = 0
        self.__cipher = None
        self.__confounder = b''
        self.__gss = None

    def set_session_key(self, session_key):
        if False:
            while True:
                i = 10
        self.__sessionKey = session_key

    def get_session_key(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__sessionKey

    def set_auth_level(self, auth_level):
        if False:
            return 10
        self.__auth_level = auth_level

    def set_auth_type(self, auth_type, callback=None):
        if False:
            while True:
                i = 10
        self.__auth_type = auth_type
        self.__auth_type_callback = callback

    def get_auth_type(self):
        if False:
            print('Hello World!')
        return self.__auth_type

    def set_max_tfrag(self, size):
        if False:
            return 10
        self.__max_xmit_size = size

    def get_credentials(self):
        if False:
            return 10
        return (self.__username, self.__password, self.__domain, self.__lmhash, self.__nthash, self.__aesKey, self.__TGT, self.__TGS)

    def set_credentials(self, username, password, domain='', lmhash='', nthash='', aesKey='', TGT=None, TGS=None):
        if False:
            i = 10
            return i + 15
        self.set_auth_level(RPC_C_AUTHN_LEVEL_CONNECT)
        self.__username = username
        self.__password = password
        self.__domain = domain
        self.__aesKey = aesKey
        self.__TGT = TGT
        self.__TGS = TGS
        if lmhash != '' or nthash != '':
            if len(lmhash) % 2:
                lmhash = '0%s' % lmhash
            if len(nthash) % 2:
                nthash = '0%s' % nthash
            try:
                self.__lmhash = unhexlify(lmhash)
                self.__nthash = unhexlify(nthash)
            except:
                self.__lmhash = lmhash
                self.__nthash = nthash
                pass

    def bind(self, iface_uuid, alter=0, bogus_binds=0, transfer_syntax=('8a885d04-1ceb-11c9-9fe8-08002b104860', '2.0')):
        if False:
            print('Hello World!')
        bind = MSRPCBind()
        ctx = self._ctx
        for i in range(bogus_binds):
            item = CtxItem()
            item['ContextID'] = ctx
            item['TransItems'] = 1
            item['ContextID'] = ctx
            item['AbstractSyntax'] = generate() + stringver_to_bin('2.0')
            item['TransferSyntax'] = uuidtup_to_bin(transfer_syntax)
            bind.addCtxItem(item)
            self._ctx += 1
            ctx += 1
        item = CtxItem()
        item['AbstractSyntax'] = iface_uuid
        item['TransferSyntax'] = uuidtup_to_bin(transfer_syntax)
        item['ContextID'] = ctx
        item['TransItems'] = 1
        bind.addCtxItem(item)
        packet = MSRPCHeader()
        packet['type'] = MSRPC_BIND
        packet['pduData'] = bind.getData()
        packet['call_id'] = self.__callid
        if alter:
            packet['type'] = MSRPC_ALTERCTX
        if self.__auth_level != RPC_C_AUTHN_LEVEL_NONE:
            if self.__username is None or self.__password is None:
                (self.__username, self.__password, self.__domain, self.__lmhash, self.__nthash, self.__aesKey, self.__TGT, self.__TGS) = self._transport.get_credentials()
            if self.__auth_type == RPC_C_AUTHN_WINNT:
                auth = ntlm.getNTLMSSPType1('', '', signingRequired=True, use_ntlmv2=self._transport.doesSupportNTLMv2())
            elif self.__auth_type == RPC_C_AUTHN_NETLOGON:
                from impacket.dcerpc.v5 import nrpc
                auth = nrpc.getSSPType1(self.__username[:-1], self.__domain, signingRequired=True)
            elif self.__auth_type == RPC_C_AUTHN_GSS_NEGOTIATE:
                (self.__cipher, self.__sessionKey, auth) = kerberosv5.getKerberosType1(self.__username, self.__password, self.__domain, self.__lmhash, self.__nthash, self.__aesKey, self.__TGT, self.__TGS, self._transport.getRemoteName(), self._transport.get_kdcHost())
            else:
                raise DCERPCException('Unsupported auth_type 0x%x' % self.__auth_type)
            sec_trailer = SEC_TRAILER()
            sec_trailer['auth_type'] = self.__auth_type
            sec_trailer['auth_level'] = self.__auth_level
            sec_trailer['auth_ctx_id'] = self._ctx + 79231
            pad = (4 - len(packet.get_packet()) % 4) % 4
            if pad != 0:
                packet['pduData'] += b'\xff' * pad
                sec_trailer['auth_pad_len'] = pad
            packet['sec_trailer'] = sec_trailer
            packet['auth_data'] = auth
        self._transport.send(packet.get_packet())
        s = self._transport.recv()
        if s != 0:
            resp = MSRPCHeader(s)
        else:
            return 0
        if resp['type'] == MSRPC_BINDACK or resp['type'] == MSRPC_ALTERCTX_R:
            bindResp = MSRPCBindAck(resp.getData())
        elif resp['type'] == MSRPC_BINDNAK or resp['type'] == MSRPC_FAULT:
            if resp['type'] == MSRPC_FAULT:
                resp = MSRPCRespHeader(resp.getData())
                status_code = unpack('<L', resp['pduData'][:4])[0]
            else:
                resp = MSRPCBindNak(resp['pduData'])
                status_code = resp['RejectedReason']
            if status_code in rpc_status_codes:
                raise DCERPCException(error_code=status_code)
            elif status_code in rpc_provider_reason:
                raise DCERPCException('Bind context rejected: %s' % rpc_provider_reason[status_code])
            else:
                raise DCERPCException('Unknown DCE RPC fault status code: %.8x' % status_code)
        else:
            raise DCERPCException('Unknown DCE RPC packet type received: %d' % resp['type'])
        for ctx in range(bogus_binds + 1, bindResp['ctx_num'] + 1):
            ctxItems = bindResp.getCtxItem(ctx)
            if ctxItems['Result'] != 0:
                msg = 'Bind context %d rejected: ' % ctx
                msg += rpc_cont_def_result.get(ctxItems['Result'], 'Unknown DCE RPC context result code: %.4x' % ctxItems['Result'])
                msg += '; '
                reason = bindResp.getCtxItem(ctx)['Reason']
                msg += rpc_provider_reason.get(reason, 'Unknown reason code: %.4x' % reason)
                if (ctxItems['Result'], reason) == (2, 1):
                    msg += " (this usually means the interface isn't listening on the given endpoint)"
                raise DCERPCException(msg)
            self.transfer_syntax = ctxItems['TransferSyntax']
        self.__max_xmit_size = bindResp['max_rfrag']
        if self.__auth_level != RPC_C_AUTHN_LEVEL_NONE:
            if self.__auth_type == RPC_C_AUTHN_WINNT:
                (response, self.__sessionKey) = ntlm.getNTLMSSPType3(auth, bindResp['auth_data'], self.__username, self.__password, self.__domain, self.__lmhash, self.__nthash, use_ntlmv2=self._transport.doesSupportNTLMv2())
                self.__flags = response['flags']
            elif self.__auth_type == RPC_C_AUTHN_NETLOGON:
                response = None
            elif self.__auth_type == RPC_C_AUTHN_GSS_NEGOTIATE:
                (self.__cipher, self.__sessionKey, response) = kerberosv5.getKerberosType3(self.__cipher, self.__sessionKey, bindResp['auth_data'])
            self.__sequence = 0
            if self.__auth_level in (RPC_C_AUTHN_LEVEL_CONNECT, RPC_C_AUTHN_LEVEL_PKT_INTEGRITY, RPC_C_AUTHN_LEVEL_PKT_PRIVACY):
                if self.__auth_type == RPC_C_AUTHN_WINNT:
                    if self.__flags & ntlm.NTLMSSP_NEGOTIATE_EXTENDED_SESSIONSECURITY:
                        self.__clientSigningKey = ntlm.SIGNKEY(self.__flags, self.__sessionKey)
                        self.__serverSigningKey = ntlm.SIGNKEY(self.__flags, self.__sessionKey, b'Server')
                        self.__clientSealingKey = ntlm.SEALKEY(self.__flags, self.__sessionKey)
                        self.__serverSealingKey = ntlm.SEALKEY(self.__flags, self.__sessionKey, b'Server')
                        cipher3 = ARC4.new(self.__clientSealingKey)
                        self.__clientSealingHandle = cipher3.encrypt
                        cipher4 = ARC4.new(self.__serverSealingKey)
                        self.__serverSealingHandle = cipher4.encrypt
                    else:
                        self.__clientSigningKey = self.__sessionKey
                        self.__serverSigningKey = self.__sessionKey
                        self.__clientSealingKey = self.__sessionKey
                        self.__serverSealingKey = self.__sessionKey
                        cipher = ARC4.new(self.__clientSigningKey)
                        self.__clientSealingHandle = cipher.encrypt
                        self.__serverSealingHandle = cipher.encrypt
                elif self.__auth_type == RPC_C_AUTHN_NETLOGON:
                    if self.__auth_level == RPC_C_AUTHN_LEVEL_PKT_INTEGRITY:
                        self.__confounder = b''
                    else:
                        self.__confounder = b'12345678'
            sec_trailer = SEC_TRAILER()
            sec_trailer['auth_type'] = self.__auth_type
            sec_trailer['auth_level'] = self.__auth_level
            sec_trailer['auth_ctx_id'] = self._ctx + 79231
            if response is not None:
                if self.__auth_type == RPC_C_AUTHN_GSS_NEGOTIATE:
                    alter_ctx = MSRPCHeader()
                    alter_ctx['type'] = MSRPC_ALTERCTX
                    alter_ctx['pduData'] = bind.getData()
                    alter_ctx['sec_trailer'] = sec_trailer
                    alter_ctx['auth_data'] = response
                    self._transport.send(alter_ctx.get_packet(), forceWriteAndx=1)
                    self.__gss = gssapi.GSSAPI(self.__cipher)
                    self.__sequence = 0
                    self.recv()
                    self.__sequence = 0
                else:
                    auth3 = MSRPCHeader()
                    auth3['type'] = MSRPC_AUTH3
                    auth3['pduData'] = b'    '
                    auth3['sec_trailer'] = sec_trailer
                    auth3['auth_data'] = response.getData()
                    self.__callid = resp['call_id']
                    auth3['call_id'] = self.__callid
                    self._transport.send(auth3.get_packet(), forceWriteAndx=1)
            self.__callid += 1
        return resp

    def _transport_send(self, rpc_packet, forceWriteAndx=0, forceRecv=0):
        if False:
            while True:
                i = 10
        rpc_packet['ctx_id'] = self._ctx
        rpc_packet['sec_trailer'] = b''
        rpc_packet['auth_data'] = b''
        if self.__auth_level in [RPC_C_AUTHN_LEVEL_PKT_INTEGRITY, RPC_C_AUTHN_LEVEL_PKT_PRIVACY]:
            sec_trailer = SEC_TRAILER()
            sec_trailer['auth_type'] = self.__auth_type
            sec_trailer['auth_level'] = self.__auth_level
            sec_trailer['auth_pad_len'] = 0
            sec_trailer['auth_ctx_id'] = self._ctx + 79231
            pad = (4 - len(rpc_packet.get_packet()) % 4) % 4
            if pad != 0:
                rpc_packet['pduData'] += b'\xbb' * pad
                sec_trailer['auth_pad_len'] = pad
            rpc_packet['sec_trailer'] = sec_trailer.getData()
            rpc_packet['auth_data'] = b' ' * 16
            plain_data = rpc_packet['pduData']
            if self.__auth_level == RPC_C_AUTHN_LEVEL_PKT_PRIVACY:
                if self.__auth_type == RPC_C_AUTHN_WINNT:
                    if self.__flags & ntlm.NTLMSSP_NEGOTIATE_EXTENDED_SESSIONSECURITY:
                        (sealedMessage, signature) = ntlm.SEAL(self.__flags, self.__clientSigningKey, self.__clientSealingKey, rpc_packet.get_packet()[:-16], plain_data, self.__sequence, self.__clientSealingHandle)
                    else:
                        (sealedMessage, signature) = ntlm.SEAL(self.__flags, self.__clientSigningKey, self.__clientSealingKey, plain_data, plain_data, self.__sequence, self.__clientSealingHandle)
                elif self.__auth_type == RPC_C_AUTHN_NETLOGON:
                    from impacket.dcerpc.v5 import nrpc
                    (sealedMessage, signature) = nrpc.SEAL(plain_data, self.__confounder, self.__sequence, self.__sessionKey, False)
                elif self.__auth_type == RPC_C_AUTHN_GSS_NEGOTIATE:
                    (sealedMessage, signature) = self.__gss.GSS_Wrap(self.__sessionKey, plain_data, self.__sequence)
                rpc_packet['pduData'] = sealedMessage
            elif self.__auth_level == RPC_C_AUTHN_LEVEL_PKT_INTEGRITY:
                if self.__auth_type == RPC_C_AUTHN_WINNT:
                    if self.__flags & ntlm.NTLMSSP_NEGOTIATE_EXTENDED_SESSIONSECURITY:
                        signature = ntlm.SIGN(self.__flags, self.__clientSigningKey, rpc_packet.get_packet()[:-16], self.__sequence, self.__clientSealingHandle)
                    else:
                        signature = ntlm.SIGN(self.__flags, self.__clientSigningKey, plain_data, self.__sequence, self.__clientSealingHandle)
                elif self.__auth_type == RPC_C_AUTHN_NETLOGON:
                    from impacket.dcerpc.v5 import nrpc
                    signature = nrpc.SIGN(plain_data, self.__confounder, self.__sequence, self.__sessionKey, False)
                elif self.__auth_type == RPC_C_AUTHN_GSS_NEGOTIATE:
                    signature = self.__gss.GSS_GetMIC(self.__sessionKey, plain_data, self.__sequence)
            rpc_packet['sec_trailer'] = sec_trailer.getData()
            rpc_packet['auth_data'] = signature
            self.__sequence += 1
        self._transport.send(rpc_packet.get_packet(), forceWriteAndx=forceWriteAndx, forceRecv=forceRecv)

    def send(self, data):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(data, MSRPCHeader) is not True:
            data = DCERPC_RawCall(data.OP_NUM, data.get_packet())
        try:
            if data['uuid'] != b'':
                data['flags'] |= PFC_OBJECT_UUID
        except:
            pass
        data['ctx_id'] = self._ctx
        data['call_id'] = self.__callid
        data['alloc_hint'] = len(data['pduData'])
        should_fragment = False
        if self._max_user_frag > 0:
            fragment_size = min(self._max_user_frag, self.__max_xmit_size)
        else:
            fragment_size = self.__max_xmit_size
        if self.__auth_level in [RPC_C_AUTHN_LEVEL_PKT_INTEGRITY, RPC_C_AUTHN_LEVEL_PKT_PRIVACY]:
            if fragment_size <= 8:
                fragment_size = 8
        if len(data['pduData']) + 128 > fragment_size:
            should_fragment = True
            if fragment_size + 128 > self.__max_xmit_size:
                fragment_size = self.__max_xmit_size - 128
        if should_fragment:
            packet = data['pduData']
            offset = 0
            while 1:
                toSend = packet[offset:offset + fragment_size]
                if not toSend:
                    break
                if offset == 0:
                    data['flags'] |= PFC_FIRST_FRAG
                else:
                    data['flags'] &= ~PFC_FIRST_FRAG
                offset += len(toSend)
                if offset >= len(packet):
                    data['flags'] |= PFC_LAST_FRAG
                else:
                    data['flags'] &= ~PFC_LAST_FRAG
                data['pduData'] = toSend
                self._transport_send(data, forceWriteAndx=1, forceRecv=data['flags'] & PFC_LAST_FRAG)
        else:
            self._transport_send(data)
        self.__callid += 1

    def recv(self):
        if False:
            for i in range(10):
                print('nop')
        finished = False
        forceRecv = 0
        retAnswer = b''
        while not finished:
            response_data = self._transport.recv(forceRecv, count=MSRPCRespHeader._SIZE)
            response_header = MSRPCRespHeader(response_data)
            while len(response_data) < response_header['frag_len']:
                response_data += self._transport.recv(forceRecv, count=response_header['frag_len'] - len(response_data))
            off = response_header.get_header_size()
            if response_header['type'] == MSRPC_FAULT and response_header['frag_len'] >= off + 4:
                status_code = unpack('<L', response_data[off:off + 4])[0]
                if status_code in rpc_status_codes:
                    raise DCERPCException(rpc_status_codes[status_code])
                elif status_code & 65535 in rpc_status_codes:
                    raise DCERPCException(rpc_status_codes[status_code & 65535])
                elif status_code in hresult_errors.ERROR_MESSAGES:
                    error_msg_short = hresult_errors.ERROR_MESSAGES[status_code][0]
                    error_msg_verbose = hresult_errors.ERROR_MESSAGES[status_code][1]
                    raise DCERPCException('%s - %s' % (error_msg_short, error_msg_verbose))
                else:
                    raise DCERPCException('Unknown DCE RPC fault status code: %.8x' % status_code)
            if response_header['flags'] & PFC_LAST_FRAG:
                finished = True
            else:
                forceRecv = 1
            answer = response_data[off:]
            auth_len = response_header['auth_len']
            if auth_len:
                auth_len += 8
                auth_data = answer[-auth_len:]
                sec_trailer = SEC_TRAILER(data=auth_data)
                answer = answer[:-auth_len]
                if sec_trailer['auth_level'] == RPC_C_AUTHN_LEVEL_PKT_PRIVACY:
                    if self.__auth_type == RPC_C_AUTHN_WINNT:
                        if self.__flags & ntlm.NTLMSSP_NEGOTIATE_EXTENDED_SESSIONSECURITY:
                            (answer, signature) = ntlm.SEAL(self.__flags, self.__serverSigningKey, self.__serverSealingKey, answer, answer, self.__sequence, self.__serverSealingHandle)
                        else:
                            (answer, signature) = ntlm.SEAL(self.__flags, self.__serverSigningKey, self.__serverSealingKey, answer, answer, self.__sequence, self.__serverSealingHandle)
                            self.__sequence += 1
                    elif self.__auth_type == RPC_C_AUTHN_NETLOGON:
                        from impacket.dcerpc.v5 import nrpc
                        (answer, cfounder) = nrpc.UNSEAL(answer, auth_data[len(sec_trailer):], self.__sessionKey, False)
                        self.__sequence += 1
                    elif self.__auth_type == RPC_C_AUTHN_GSS_NEGOTIATE:
                        if self.__sequence > 0:
                            (answer, cfounder) = self.__gss.GSS_Unwrap(self.__sessionKey, answer, self.__sequence, direction='init', authData=auth_data)
                elif sec_trailer['auth_level'] == RPC_C_AUTHN_LEVEL_PKT_INTEGRITY:
                    if self.__auth_type == RPC_C_AUTHN_WINNT:
                        ntlmssp = auth_data[12:]
                        if self.__flags & ntlm.NTLMSSP_NEGOTIATE_EXTENDED_SESSIONSECURITY:
                            signature = ntlm.SIGN(self.__flags, self.__serverSigningKey, answer, self.__sequence, self.__serverSealingHandle)
                        else:
                            signature = ntlm.SIGN(self.__flags, self.__serverSigningKey, ntlmssp, self.__sequence, self.__serverSealingHandle)
                            self.__sequence += 1
                    elif self.__auth_type == RPC_C_AUTHN_NETLOGON:
                        from impacket.dcerpc.v5 import nrpc
                        ntlmssp = auth_data[12:]
                        signature = nrpc.SIGN(ntlmssp, self.__confounder, self.__sequence, self.__sessionKey, False)
                        self.__sequence += 1
                    elif self.__auth_type == RPC_C_AUTHN_GSS_NEGOTIATE:
                        pass
                if sec_trailer['auth_pad_len']:
                    answer = answer[:-sec_trailer['auth_pad_len']]
            retAnswer += answer
        return retAnswer

    def alter_ctx(self, newUID, bogus_binds=0):
        if False:
            for i in range(10):
                print('nop')
        answer = self.__class__(self._transport)
        answer.set_credentials(self.__username, self.__password, self.__domain, self.__lmhash, self.__nthash, self.__aesKey, self.__TGT, self.__TGS)
        answer.set_auth_type(self.__auth_type)
        answer.set_auth_level(self.__auth_level)
        answer.set_ctx_id(self._ctx + 1)
        answer.__callid = self.__callid
        answer.bind(newUID, alter=1, bogus_binds=bogus_binds, transfer_syntax=bin_to_uuidtup(self.transfer_syntax))
        return answer

class DCERPC_RawCall(MSRPCRequestHeader):

    def __init__(self, op_num, data=b'', uuid=None):
        if False:
            i = 10
            return i + 15
        MSRPCRequestHeader.__init__(self)
        self['op_num'] = op_num
        self['pduData'] = data
        if uuid is not None:
            self['flags'] |= PFC_OBJECT_UUID
            self['uuid'] = uuid

    def setData(self, data):
        if False:
            while True:
                i = 10
        self['pduData'] = data

class CommonHeader(NDRSTRUCT):
    structure = (('Version', UCHAR), ('Endianness', UCHAR), ('CommonHeaderLength', USHORT), ('Filler', ULONG))

    def __init__(self, data=None, isNDR64=False):
        if False:
            return 10
        NDRSTRUCT.__init__(self, data, isNDR64)
        if data is None:
            self['Version'] = 1
            self['Endianness'] = 16
            self['CommonHeaderLength'] = 8
            self['Filler'] = 3435973836

class PrivateHeader(NDRSTRUCT):
    structure = (('ObjectBufferLength', ULONG), ('Filler', ULONG))

    def __init__(self, data=None, isNDR64=False):
        if False:
            while True:
                i = 10
        NDRSTRUCT.__init__(self, data, isNDR64)
        if data is None:
            self['Filler'] = 3435973836

class TypeSerialization1(NDRSTRUCT):
    commonHdr = (('CommonHeader', CommonHeader), ('PrivateHeader', PrivateHeader))

    def getData(self, soFar=0):
        if False:
            i = 10
            return i + 15
        self['PrivateHeader']['ObjectBufferLength'] = len(NDRSTRUCT.getData(self, soFar)) + len(NDRSTRUCT.getDataReferents(self, soFar)) - len(self['CommonHeader']) - len(self['PrivateHeader'])
        return NDRSTRUCT.getData(self, soFar)

class DCERPCServer(Thread):
    """
    A minimalistic DCERPC Server, mainly used by the smbserver, for now. Might be useful
    for other purposes in the future, but we should do it way stronger.
    If you want to implement a DCE Interface Server, use this class as the base class
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        Thread.__init__(self)
        self._listenPort = 0
        self._listenAddress = '127.0.0.1'
        self._listenUUIDS = {}
        self._boundUUID = b''
        self._sock = None
        self._clientSock = None
        self._callid = 1
        self._max_frag = None
        self._max_xmit_size = 4280
        self.__log = LOG
        self._sock = socket.socket()
        self._sock.bind((self._listenAddress, self._listenPort))

    def log(self, msg, level=logging.INFO):
        if False:
            print('Hello World!')
        self.__log.log(level, msg)

    def addCallbacks(self, ifaceUUID, secondaryAddr, callbacks):
        if False:
            print('Hello World!')
        '\n        adds a call back to a UUID/opnum call\n        \n        :param uuid ifaceUUID: the interface UUID\n        :param string secondaryAddr: the secondary address to answer as part of the bind request (e.g. \\\\PIPE\\\\srvsvc)\n        :param dict callbacks: the callbacks for each opnum. Format is [opnum] = callback\n        '
        self._listenUUIDS[uuidtup_to_bin(ifaceUUID)] = {}
        self._listenUUIDS[uuidtup_to_bin(ifaceUUID)]['SecondaryAddr'] = secondaryAddr
        self._listenUUIDS[uuidtup_to_bin(ifaceUUID)]['CallBacks'] = callbacks
        self.log('Callback added for UUID %s V:%s' % ifaceUUID)

    def setListenPort(self, portNum):
        if False:
            return 10
        self._listenPort = portNum
        self._sock = socket.socket()
        self._sock.bind((self._listenAddress, self._listenPort))

    def getListenPort(self):
        if False:
            while True:
                i = 10
        return self._sock.getsockname()[1]

    def recv(self):
        if False:
            for i in range(10):
                print('nop')
        finished = False
        retAnswer = b''
        response_data = b''
        while not finished:
            response_data = self._clientSock.recv(MSRPCRespHeader._SIZE)
            if response_data == b'':
                return None
            response_header = MSRPCRespHeader(response_data)
            while len(response_data) < response_header['frag_len']:
                response_data += self._clientSock.recv(response_header['frag_len'] - len(response_data))
            response_header = MSRPCRespHeader(response_data)
            if response_header['flags'] & PFC_LAST_FRAG:
                finished = True
            answer = response_header['pduData']
            auth_len = response_header['auth_len']
            if auth_len:
                auth_len += 8
                auth_data = answer[-auth_len:]
                sec_trailer = SEC_TRAILER(data=auth_data)
                answer = answer[:-auth_len]
                if sec_trailer['auth_pad_len']:
                    answer = answer[:-sec_trailer['auth_pad_len']]
            retAnswer += answer
        return response_data

    def run(self):
        if False:
            i = 10
            return i + 15
        self._sock.listen(10)
        while True:
            (self._clientSock, address) = self._sock.accept()
            try:
                while True:
                    data = self.recv()
                    if data is None:
                        break
                    answer = self.processRequest(data)
                    if answer is not None:
                        self.send(answer)
            except Exception:
                pass
            self._clientSock.close()

    def send(self, data):
        if False:
            return 10
        max_frag = self._max_frag
        if len(data['pduData']) > self._max_xmit_size - 32:
            max_frag = self._max_xmit_size - 32
        if self._max_frag:
            max_frag = min(max_frag, self._max_frag)
        if max_frag and len(data['pduData']) > 0:
            packet = data['pduData']
            offset = 0
            while 1:
                toSend = packet[offset:offset + max_frag]
                if not toSend:
                    break
                flags = 0
                if offset == 0:
                    flags |= PFC_FIRST_FRAG
                offset += len(toSend)
                if offset == len(packet):
                    flags |= PFC_LAST_FRAG
                data['flags'] = flags
                data['pduData'] = toSend
                self._clientSock.send(data.get_packet())
        else:
            self._clientSock.send(data.get_packet())
        self._callid += 1

    def bind(self, packet, bind):
        if False:
            print('Hello World!')
        NDRSyntax = ('8a885d04-1ceb-11c9-9fe8-08002b104860', '2.0')
        resp = MSRPCBindAck()
        resp['type'] = MSRPC_BINDACK
        resp['flags'] = packet['flags']
        resp['frag_len'] = 0
        resp['auth_len'] = 0
        resp['auth_data'] = b''
        resp['call_id'] = packet['call_id']
        resp['max_tfrag'] = bind['max_tfrag']
        resp['max_rfrag'] = bind['max_rfrag']
        resp['assoc_group'] = 4660
        resp['ctx_num'] = 0
        data = bind['ctx_items']
        ctx_items = b''
        resp['SecondaryAddrLen'] = 0
        for i in range(bind['ctx_num']):
            result = MSRPC_CONT_RESULT_USER_REJECT
            item = CtxItem(data)
            data = data[len(item):]
            if item['TransferSyntax'] == uuidtup_to_bin(NDRSyntax):
                reason = 1
                for j in self._listenUUIDS:
                    if item['AbstractSyntax'] == j:
                        resp['SecondaryAddr'] = self._listenUUIDS[item['AbstractSyntax']]['SecondaryAddr']
                        resp['SecondaryAddrLen'] = len(resp['SecondaryAddr']) + 1
                        reason = 0
                        self._boundUUID = j
            else:
                reason = 2
            if reason == 0:
                result = MSRPC_CONT_RESULT_ACCEPT
            if reason == 1:
                LOG.error('Bind request for an unsupported interface %s' % bin_to_uuidtup(item['AbstractSyntax']))
            resp['ctx_num'] += 1
            itemResult = CtxItemResult()
            itemResult['Result'] = result
            itemResult['Reason'] = reason
            itemResult['TransferSyntax'] = uuidtup_to_bin(NDRSyntax)
            ctx_items += itemResult.getData()
        resp['Pad'] = 'A' * ((4 - (resp['SecondaryAddrLen'] + MSRPCBindAck._SIZE) % 4) % 4)
        resp['ctx_items'] = ctx_items
        resp['frag_len'] = len(resp.getData())
        self._clientSock.send(resp.getData())
        return None

    def processRequest(self, data):
        if False:
            return 10
        packet = MSRPCHeader(data)
        if packet['type'] == MSRPC_BIND:
            bind = MSRPCBind(packet['pduData'])
            self.bind(packet, bind)
            packet = None
        elif packet['type'] == MSRPC_REQUEST:
            request = MSRPCRequestHeader(data)
            response = MSRPCRespHeader(data)
            response['type'] = MSRPC_RESPONSE
            if request['op_num'] in self._listenUUIDS[self._boundUUID]['CallBacks']:
                returnData = self._listenUUIDS[self._boundUUID]['CallBacks'][request['op_num']](request['pduData'])
                response['pduData'] = returnData
            else:
                LOG.error('Unsupported DCERPC opnum %d called for interface %s' % (request['op_num'], bin_to_uuidtup(self._boundUUID)))
                response['type'] = MSRPC_FAULT
                response['pduData'] = pack('<L', 1764)
            response['frag_len'] = len(response)
            return response
        else:
            packet = MSRPCRespHeader(data)
            packet['type'] = MSRPC_FAULT
        return packet