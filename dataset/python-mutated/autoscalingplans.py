def validate_predictivescalingmaxcapacitybehavior(predictivescalingmaxcapacitybehavior):
    if False:
        return 10
    '\n    Validate PredictiveScalingMaxCapacityBehavior for ScalingInstruction\n    Property: ScalingInstruction.PredictiveScalingMaxCapacityBehavior\n    '
    VALID_PREDICTIVESCALINGMAXCAPACITYBEHAVIOR = ('SetForecastCapacityToMaxCapacity', 'SetMaxCapacityToForecastCapacity', 'SetMaxCapacityAboveForecastCapacity')
    if predictivescalingmaxcapacitybehavior not in VALID_PREDICTIVESCALINGMAXCAPACITYBEHAVIOR:
        raise ValueError('ScalingInstruction PredictiveScalingMaxCapacityBehavior must be one of: %s' % ', '.join(VALID_PREDICTIVESCALINGMAXCAPACITYBEHAVIOR))
    return predictivescalingmaxcapacitybehavior

def validate_predictivescalingmode(predictivescalingmode):
    if False:
        return 10
    '\n    Validate PredictiveScalingMode for ScalingInstruction\n    Property: ScalingInstruction.PredictiveScalingMode\n    '
    VALID_PREDICTIVESCALINGMODE = ('ForecastAndScale', 'ForecastOnly')
    if predictivescalingmode not in VALID_PREDICTIVESCALINGMODE:
        raise ValueError('ScalingInstruction PredictiveScalingMode must be one of: %s' % ', '.join(VALID_PREDICTIVESCALINGMODE))
    return predictivescalingmode

def validate_scalingpolicyupdatebehavior(scalingpolicyupdatebehavior):
    if False:
        for i in range(10):
            print('nop')
    '\n    Validate ScalingPolicyUpdateBehavior for ScalingInstruction\n    Property: ScalingInstruction.ScalingPolicyUpdateBehavior\n    '
    VALID_SCALINGPOLICYUPDATEBEHAVIOR = ('KeepExternalPolicies', 'ReplaceExternalPolicies')
    if scalingpolicyupdatebehavior not in VALID_SCALINGPOLICYUPDATEBEHAVIOR:
        raise ValueError('ScalingInstruction ScalingPolicyUpdateBehavior must be one of: %s' % ', '.join(VALID_SCALINGPOLICYUPDATEBEHAVIOR))
    return scalingpolicyupdatebehavior

def scalable_dimension_type(scalable_dimension):
    if False:
        print('Hello World!')
    '\n    Property: ScalingInstruction.ScalableDimension\n    '
    valid_values = ['autoscaling:autoScalingGroup:DesiredCapacity', 'ecs:service:DesiredCount', 'ec2:spot-fleet-request:TargetCapacity', 'rds:cluster:ReadReplicaCount', 'dynamodb:table:ReadCapacityUnits', 'dynamodb:table:WriteCapacityUnits', 'dynamodb:index:ReadCapacityUnits', 'dynamodb:index:WriteCapacityUnits']
    if scalable_dimension not in valid_values:
        raise ValueError('ScalableDimension must be one of: "%s"' % ', '.join(valid_values))
    return scalable_dimension

def service_namespace_type(service_namespace):
    if False:
        return 10
    '\n    Property: ScalingInstruction.ServiceNamespace\n    '
    valid_values = ['autoscaling', 'ecs', 'ec2', 'rds', 'dynamodb']
    if service_namespace not in valid_values:
        raise ValueError('ServiceNamespace must be one of: "%s"' % ', '.join(valid_values))
    return service_namespace

def statistic_type(statistic):
    if False:
        for i in range(10):
            print('nop')
    '\n    Property: CustomizedScalingMetricSpecification.Statistic\n    '
    valid_values = ['Average', 'Minimum', 'Maximum', 'SampleCount', 'Sum']
    if statistic not in valid_values:
        raise ValueError('Statistic must be one of: "%s"' % ', '.join(valid_values))
    return statistic