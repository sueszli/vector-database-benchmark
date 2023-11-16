class VariableAttributes(object):
    EFI_VARIABLE_NON_VOLATILE = 1
    EFI_VARIABLE_BOOTSERVICE_ACCESS = 2
    EFI_VARIABLE_RUNTIME_ACCESS = 4
    VAR_CHECK_VARIABLE_PROPERTY_READ_ONLY = 1
    VarAttributesMap = {'NV': EFI_VARIABLE_NON_VOLATILE, 'BS': EFI_VARIABLE_BOOTSERVICE_ACCESS, 'RT': EFI_VARIABLE_RUNTIME_ACCESS, 'RO': VAR_CHECK_VARIABLE_PROPERTY_READ_ONLY}

    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

    @staticmethod
    def GetVarAttributes(var_attr_str):
        if False:
            i = 10
            return i + 15
        VarAttr = 0
        VarProp = 0
        attr_list = var_attr_str.split(',')
        for attr in attr_list:
            attr = attr.strip()
            if attr == 'RO':
                VarProp = VariableAttributes.VAR_CHECK_VARIABLE_PROPERTY_READ_ONLY
            else:
                VarAttr = VarAttr | VariableAttributes.VarAttributesMap.get(attr, 0)
        return (VarAttr, VarProp)

    @staticmethod
    def ValidateVarAttributes(var_attr_str):
        if False:
            while True:
                i = 10
        if not var_attr_str:
            return (True, '')
        attr_list = var_attr_str.split(',')
        attr_temp = []
        for attr in attr_list:
            attr = attr.strip()
            attr_temp.append(attr)
            if attr not in VariableAttributes.VarAttributesMap:
                return (False, "The variable attribute %s is not support to be specified in dsc file. Supported variable attribute are ['BS','NV','RT','RO'] ")
        if 'RT' in attr_temp and 'BS' not in attr_temp:
            return (False, 'the RT attribute need the BS attribute to be present')
        return (True, '')