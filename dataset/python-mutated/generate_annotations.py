import platform
import subprocess
from pathlib import Path
script_path = Path(__file__).parent.resolve()
root_path = script_path.parent.parent.absolute()
flatc_exe = Path('flatc' if not platform.system() == 'Windows' else 'flatc.exe')
if root_path in flatc_exe.parents:
    flatc_exe = flatc_exe.relative_to(root_path)
flatc_path = Path(root_path, flatc_exe)
assert flatc_path.exists(), 'Cannot find the flatc compiler ' + str(flatc_path)
tests_path = Path(script_path, 'tests')

def flatc_annotate(schema, file, cwd=script_path):
    if False:
        for i in range(10):
            print('nop')
    cmd = [str(flatc_path), '--annotate', schema, file]
    result = subprocess.run(cmd, cwd=str(cwd), check=True)
test_files = ['annotated_binary.bin', 'tests/invalid_root_offset.bin', 'tests/invalid_root_table_too_short.bin', 'tests/invalid_root_table_vtable_offset.bin', 'tests/invalid_string_length.bin', 'tests/invalid_string_length_cut_short.bin', 'tests/invalid_struct_array_field_cut_short.bin', 'tests/invalid_struct_field_cut_short.bin', 'tests/invalid_table_field_size.bin', 'tests/invalid_table_field_offset.bin', 'tests/invalid_union_type_value.bin', 'tests/invalid_vector_length_cut_short.bin', 'tests/invalid_vector_scalars_cut_short.bin', 'tests/invalid_vector_strings_cut_short.bin', 'tests/invalid_vector_structs_cut_short.bin', 'tests/invalid_vector_tables_cut_short.bin', 'tests/invalid_vector_unions_cut_short.bin', 'tests/invalid_vector_union_type_value.bin', 'tests/invalid_vtable_ref_table_size_short.bin', 'tests/invalid_vtable_ref_table_size.bin', 'tests/invalid_vtable_size_short.bin', 'tests/invalid_vtable_size.bin', 'tests/invalid_vtable_field_offset.bin']
for test_file in test_files:
    flatc_annotate('annotated_binary.fbs', test_file)