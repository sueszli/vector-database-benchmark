class SkuInfoClass(object):

    def __init__(self, SkuIdName='', SkuId='', VariableName='', VariableGuid='', VariableOffset='', HiiDefaultValue='', VpdOffset='', DefaultValue='', VariableGuidValue='', VariableAttribute='', DefaultStore=None):
        if False:
            return 10
        self.SkuIdName = SkuIdName
        self.SkuId = SkuId
        if DefaultStore is None:
            DefaultStore = {}
        self.VariableName = VariableName
        self.VariableGuid = VariableGuid
        self.VariableGuidValue = VariableGuidValue
        self.VariableOffset = VariableOffset
        self.HiiDefaultValue = HiiDefaultValue
        self.VariableAttribute = VariableAttribute
        self.DefaultStoreDict = DefaultStore
        self.VpdOffset = VpdOffset
        self.DefaultValue = DefaultValue

    def __str__(self):
        if False:
            print('Hello World!')
        Rtn = 'SkuId = ' + str(self.SkuId) + ',' + 'SkuIdName = ' + str(self.SkuIdName) + ',' + 'VariableName = ' + str(self.VariableName) + ',' + 'VariableGuid = ' + str(self.VariableGuid) + ',' + 'VariableOffset = ' + str(self.VariableOffset) + ',' + 'HiiDefaultValue = ' + str(self.HiiDefaultValue) + ',' + 'VpdOffset = ' + str(self.VpdOffset) + ',' + 'DefaultValue = ' + str(self.DefaultValue) + ','
        return Rtn

    def __deepcopy__(self, memo):
        if False:
            while True:
                i = 10
        new_sku = SkuInfoClass()
        new_sku.SkuIdName = self.SkuIdName
        new_sku.SkuId = self.SkuId
        new_sku.VariableName = self.VariableName
        new_sku.VariableGuid = self.VariableGuid
        new_sku.VariableGuidValue = self.VariableGuidValue
        new_sku.VariableOffset = self.VariableOffset
        new_sku.HiiDefaultValue = self.HiiDefaultValue
        new_sku.VariableAttribute = self.VariableAttribute
        new_sku.DefaultStoreDict = {key: value for (key, value) in self.DefaultStoreDict.items()}
        new_sku.VpdOffset = self.VpdOffset
        new_sku.DefaultValue = self.DefaultValue
        return new_sku