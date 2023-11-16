from featuretools.primitives.base.primitive_base import PrimitiveBase

class AggregationPrimitive(PrimitiveBase):

    def generate_name(self, base_feature_names, relationship_path_name, parent_dataframe_name, where_str, use_prev_str):
        if False:
            while True:
                i = 10
        base_features_str = ', '.join(base_feature_names)
        return '%s(%s.%s%s%s%s)' % (self.name.upper(), relationship_path_name, base_features_str, where_str, use_prev_str, self.get_args_string())

    def generate_names(self, base_feature_names, relationship_path_name, parent_dataframe_name, where_str, use_prev_str):
        if False:
            print('Hello World!')
        n = self.number_output_features
        base_name = self.generate_name(base_feature_names, relationship_path_name, parent_dataframe_name, where_str, use_prev_str)
        return [base_name + '[%s]' % i for i in range(n)]