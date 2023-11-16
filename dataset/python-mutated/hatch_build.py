from hatchling.metadata.plugin.interface import MetadataHookInterface

class CustomMetadataHook(MetadataHookInterface):

    def update(self, metadata):
        if False:
            return 10
        optional_dependencies = self.config['optional-dependencies']
        for (feature, dependencies) in list(optional_dependencies.items()):
            if '-' not in feature:
                continue
            legacy_feature = feature.replace('-', '_')
            optional_dependencies[legacy_feature] = dependencies
        metadata['optional-dependencies'] = optional_dependencies