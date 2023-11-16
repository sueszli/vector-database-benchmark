"""Tests for generate_test_hlo_checks."""
from absl.testing import absltest
from xla.service import generate_test_hlo_checks

class GenerateTestHloChecksTest(absltest.TestCase):

    def test_replacement(self):
        if False:
            for i in range(10):
                print('nop')
        input_hlo = "\n%param.0 # Do not replace if it's not CHECK'd.\n// CHECK: %computation { # Do not replace computations\n// CHECK: %param.0 = parameter(0) # Replace\n// CHECK: %param_1 = parameter(1)\n// CHECK-NEXT: %add.1 = add(%param.0, %param_1) # Replace for any CHECK-directive\n// CHECK-NEXT: ROOT %reduce = reduce(%add.1)\n// CHECK-NEXT: }\n// CHECK: %computation.2 { # New computation resets the counter.\n// CHECK-NEXT: %parameter.0 = parameter(0)\n// CHECK-NEXT: %get-tuple-element.1 = get-tuple-element(%parameter.0)\n// CHECK-NEXT: ROOT %bitcast-convert = bitcast-convert(%get-tuple-element.1)\n"
        self.assertEqual(generate_test_hlo_checks.replace_instruction_names(input_hlo), "\n%param.0 # Do not replace if it's not CHECK'd.\n// CHECK: %computation { # Do not replace computations\n// CHECK: [[param_0_0:%[^ ]+]] = parameter(0) # Replace\n// CHECK: [[param_1_1:%[^ ]+]] = parameter(1)\n// CHECK-NEXT: [[add_1_2:%[^ ]+]] = add([[param_0_0]], [[param_1_1]]) # Replace for any CHECK-directive\n// CHECK-NEXT: ROOT [[reduce_3:%[^ ]+]] = reduce([[add_1_2]])\n// CHECK-NEXT: }\n// CHECK: %computation.2 { # New computation resets the counter.\n// CHECK-NEXT: [[parameter_0_0:%[^ ]+]] = parameter(0)\n// CHECK-NEXT: [[get_tuple_element_1_1:%[^ ]+]] = get-tuple-element([[parameter_0_0]])\n// CHECK-NEXT: ROOT [[bitcast_convert_2:%[^ ]+]] = bitcast-convert([[get_tuple_element_1_1]])\n")
if __name__ == '__main__':
    absltest.main()