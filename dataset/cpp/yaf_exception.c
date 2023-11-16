/*
  +----------------------------------------------------------------------+
  | Yet Another Framework                                                |
  +----------------------------------------------------------------------+
  | This source file is subject to version 3.01 of the PHP license,      |
  | that is bundled with this package in the file LICENSE, and is        |
  | available through the world-wide-web at the following url:           |
  | http://www.php.net/license/3_01.txt                                  |
  | If you did not receive a copy of the PHP license and are unable to   |
  | obtain it through the world-wide-web, please send a note to          |
  | license@php.net so we can mail you a copy immediately.               |
  +----------------------------------------------------------------------+
  | Author: Xinchen Hui  <laruence@php.net>                              |
  +----------------------------------------------------------------------+
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "Zend/zend_exceptions.h"

#include "php_yaf.h"
#include "yaf_application.h"
#include "yaf_namespace.h"
#include "yaf_exception.h"

zend_class_entry *yaf_ce_RuntimeException;
zend_class_entry *yaf_exception_ce;

zend_class_entry *yaf_buildin_exceptions[YAF_MAX_BUILDIN_EXCEPTION];

ZEND_COLD void yaf_trigger_error(int type, char *format, ...) /* {{{ */ {
	va_list args;

	if (yaf_is_throw_exception()) {
		char buf[256];
		va_start(args, format);
		vsnprintf(buf, sizeof(buf), format, args);
		va_end(args);
		yaf_throw_exception(type, buf);
	} else {
		zend_string *msg;
		yaf_application_object *app = yaf_application_instance();

		va_start(args, format);
		msg = vstrpprintf(0, format, args);
		va_end(args);

		if (app) {
			app->err_no = type;
			app->err_msg = msg;
		}
		php_error_docref(NULL, E_RECOVERABLE_ERROR, "%s", ZSTR_VAL(msg));
	}
}
/* }}} */

zend_class_entry * yaf_get_exception_base(int root) /* {{{ */ {
#if can_handle_soft_dependency_on_SPL && defined(HAVE_SPL) && ((PHP_MAJOR_VERSION > 5) || (PHP_MAJOR_VERSION == 5 && PHP_MINOR_VERSION >= 1))
	if (!root) {
		if (!spl_ce_RuntimeException) {
			zend_class_entry **pce;

			if (zend_hash_find(CG(class_table), "runtimeexception", sizeof("RuntimeException"), (void **) &pce) == SUCCESS) {
				spl_ce_RuntimeException = *pce;
				return *pce;
			}
		} else {
			return spl_ce_RuntimeException;
		}
	}
#endif

	return zend_exception_get_default();
}
/* }}} */

ZEND_COLD void yaf_throw_exception(long code, char *message) /* {{{ */ {
	zend_class_entry *base_exception = yaf_exception_ce;

	if ((code & YAF_ERR_BASE) == YAF_ERR_BASE
			&& yaf_buildin_exceptions[YAF_EXCEPTION_OFFSET(code)]) {
		base_exception = yaf_buildin_exceptions[YAF_EXCEPTION_OFFSET(code)];
	}

	zend_throw_exception(base_exception, message, code);
}
/* }}} */

/** {{{ yaf_exception_methods
*/
zend_function_entry yaf_exception_methods[] = {
	{NULL, NULL, NULL}
};
/* }}} */

/** {{{ YAF_STARTUP_FUNCTION
*/
YAF_STARTUP_FUNCTION(exception) {
	zend_class_entry ce;
	zend_class_entry startup_ce;
	zend_class_entry route_ce;
	zend_class_entry dispatch_ce;
	zend_class_entry loader_ce;
	zend_class_entry module_notfound_ce;
	zend_class_entry controller_notfound_ce;
	zend_class_entry action_notfound_ce;
	zend_class_entry view_notfound_ce;
	zend_class_entry type_ce;

	YAF_INIT_CLASS_ENTRY(ce, "Yaf_Exception", "Yaf\\Exception", yaf_exception_methods);
	yaf_exception_ce = zend_register_internal_class_ex(&ce, yaf_get_exception_base(0));
	zend_declare_property_null(yaf_exception_ce, ZEND_STRL("message"), 	ZEND_ACC_PROTECTED);
	zend_declare_property_long(yaf_exception_ce, ZEND_STRL("code"), 0,	ZEND_ACC_PROTECTED);
	zend_declare_property_null(yaf_exception_ce, ZEND_STRL("previous"),  ZEND_ACC_PROTECTED);

	YAF_INIT_CLASS_ENTRY(startup_ce, "Yaf_Exception_StartupError", "Yaf\\Exception\\StartupError", NULL);
	yaf_buildin_exceptions[YAF_EXCEPTION_OFFSET(YAF_ERR_STARTUP_FAILED)] = zend_register_internal_class_ex(&startup_ce, yaf_exception_ce);

	YAF_INIT_CLASS_ENTRY(route_ce, "Yaf_Exception_RouterFailed", "Yaf\\Exception\\RouterFailed", NULL);
	yaf_buildin_exceptions[YAF_EXCEPTION_OFFSET(YAF_ERR_ROUTE_FAILED)] = zend_register_internal_class_ex(&route_ce, yaf_exception_ce);

	YAF_INIT_CLASS_ENTRY(dispatch_ce, "Yaf_Exception_DispatchFailed", "Yaf\\Exception\\DispatchFailed", NULL);
	yaf_buildin_exceptions[YAF_EXCEPTION_OFFSET(YAF_ERR_DISPATCH_FAILED)] = zend_register_internal_class_ex(&dispatch_ce, yaf_exception_ce);

	YAF_INIT_CLASS_ENTRY(loader_ce, "Yaf_Exception_LoadFailed", "Yaf\\Exception\\LoadFailed", NULL);
	yaf_buildin_exceptions[YAF_EXCEPTION_OFFSET(YAF_ERR_AUTOLOAD_FAILED)] = zend_register_internal_class_ex(&loader_ce, yaf_exception_ce);

	YAF_INIT_CLASS_ENTRY(module_notfound_ce, "Yaf_Exception_LoadFailed_Module", "Yaf\\Exception\\LoadFailed\\Module", NULL);
	yaf_buildin_exceptions[YAF_EXCEPTION_OFFSET(YAF_ERR_NOTFOUND_MODULE)] = zend_register_internal_class_ex(&module_notfound_ce, yaf_buildin_exceptions[YAF_EXCEPTION_OFFSET(YAF_ERR_AUTOLOAD_FAILED)]);

	YAF_INIT_CLASS_ENTRY(controller_notfound_ce, "Yaf_Exception_LoadFailed_Controller", "Yaf\\Exception\\LoadFailed\\Controller", NULL);
	yaf_buildin_exceptions[YAF_EXCEPTION_OFFSET(YAF_ERR_NOTFOUND_CONTROLLER)] = zend_register_internal_class_ex(&controller_notfound_ce, yaf_buildin_exceptions[YAF_EXCEPTION_OFFSET(YAF_ERR_AUTOLOAD_FAILED)]);

	YAF_INIT_CLASS_ENTRY(action_notfound_ce, "Yaf_Exception_LoadFailed_Action", "Yaf\\Exception\\LoadFailed\\Action", NULL);
	yaf_buildin_exceptions[YAF_EXCEPTION_OFFSET(YAF_ERR_NOTFOUND_ACTION)] = zend_register_internal_class_ex(&action_notfound_ce, yaf_buildin_exceptions[YAF_EXCEPTION_OFFSET(YAF_ERR_AUTOLOAD_FAILED)]);

	YAF_INIT_CLASS_ENTRY(view_notfound_ce, "Yaf_Exception_LoadFailed_View", "Yaf\\Exception\\LoadFailed\\View", NULL);
	yaf_buildin_exceptions[YAF_EXCEPTION_OFFSET(YAF_ERR_NOTFOUND_VIEW)] = zend_register_internal_class_ex(&view_notfound_ce, yaf_buildin_exceptions[YAF_EXCEPTION_OFFSET(YAF_ERR_AUTOLOAD_FAILED)]);

	YAF_INIT_CLASS_ENTRY(type_ce, "Yaf_Exception_TypeError", "Yaf\\Exception\\TypeError", NULL);
	yaf_buildin_exceptions[YAF_EXCEPTION_OFFSET(YAF_ERR_TYPE_ERROR)] = zend_register_internal_class_ex(&type_ce, yaf_exception_ce);

	return SUCCESS;
}
/* }}} */

/*
 * Local variables:
 * tab-width: 4
 * c-basic-offset: 4
 * End:
 * vim600: noet sw=4 ts=4 fdm=marker
 * vim<600: noet sw=4 ts=4
 */
