/**************************************************************************//**
 *
 *
 *
 *****************************************************************************/
#include <x86_64_accton_as7726_32x/x86_64_accton_as7726_32x_config.h>

#include "x86_64_accton_as7726_32x_log.h"

static int
datatypes_init__(void)
{
#define X86_64_ACCTON_AS7726_32X_ENUMERATION_ENTRY(_enum_name, _desc)     AIM_DATATYPE_MAP_REGISTER(_enum_name, _enum_name##_map, _desc,                               AIM_LOG_INTERNAL);
#include <x86_64_accton_as7726_32x/x86_64_accton_as7726_32x.x>
    return 0;
}

void __x86_64_accton_as7726_32x_module_init__(void)
{
    AIM_LOG_STRUCT_REGISTER();
    datatypes_init__();
}

int __onlp_platform_version__ = 1;
