"""A function to build an object detection box coder from configuration."""
from object_detection.builders import box_coder_builder
from object_detection.builders import matcher_builder
from object_detection.builders import region_similarity_calculator_builder
from object_detection.core import target_assigner

def build(target_assigner_config):
    if False:
        return 10
    'Builds a TargetAssigner object based on the config.\n\n  Args:\n    target_assigner_config: A target_assigner proto message containing config\n      for the desired target assigner.\n\n  Returns:\n    TargetAssigner object based on the config.\n  '
    matcher_instance = matcher_builder.build(target_assigner_config.matcher)
    similarity_calc_instance = region_similarity_calculator_builder.build(target_assigner_config.similarity_calculator)
    box_coder = box_coder_builder.build(target_assigner_config.box_coder)
    return target_assigner.TargetAssigner(matcher=matcher_instance, similarity_calc=similarity_calc_instance, box_coder_instance=box_coder)