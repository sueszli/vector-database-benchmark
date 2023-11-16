"""Smoke test of functions in StableHLO Portable APIs."""
from tensorflow.compiler.mlir.stablehlo import stablehlo

def smoketest():
    if False:
        i = 10
        return i + 15
    'Test StableHLO Portable APIs.'
    assert isinstance(stablehlo.get_api_version(), int)
    assembly = '\n    module @jit_f_jax.0 {\n      func.func public @main(%arg0: tensor<ui32>) -> tensor<i1> {\n        %0 = stablehlo.constant dense<1> : tensor<ui32>\n        %1 = "stablehlo.compare"(%arg0, %0) {compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction GE>} : (tensor<ui32>, tensor<ui32>) -> tensor<i1>\n        return %1 : tensor<i1>\n      }\n    }\n  '
    target = stablehlo.get_current_version()
    artifact = stablehlo.serialize_portable_artifact(assembly, target)
    deserialized = stablehlo.deserialize_portable_artifact(artifact)
    rountrip = stablehlo.serialize_portable_artifact(deserialized, target)
    assert artifact == rountrip
if __name__ == '__main__':
    smoketest()