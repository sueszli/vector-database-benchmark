import io
import logging
import sys
import zipfile
from pathlib import Path
from typing import Set
import torch
from test.jit.fixtures_srcs.fixtures_src import *
from torch.jit.mobile import _load_for_lite_interpreter, _export_operator_list
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
'\nThis file is used to generate model for test operator change. Please refer to\nhttps://github.com/pytorch/rfcs/blob/master/RFC-0017-PyTorch-Operator-Versioning.md for more details.\n\nA systematic workflow to change operator is needed to ensure\nBackwards Compatibility (BC) / Forwards Compatibility (FC) for operator changes. For BC-breaking operator change,\nan upgrader is needed. Here is the flow to properly land a BC-breaking operator change.\n\n1. Write an upgrader in caffe2/torch/csrc/jit/operator_upgraders/upgraders_entry.cpp file. The softly enforced\nnaming format is <operator_name>_<operator_overload>_<start>_<end>. For example, the below example means that\ndiv.Tensor at version from 0 to 3 needs to be replaced by this upgrader.\n\n```\n/*\ndiv_Tensor_0_3 is added for a change of operator div in pr xxxxxxx.\nCreate date: 12/02/2021\nExpire date: 06/02/2022\n*/\n     {"div_Tensor_0_3", R"SCRIPT(\ndef div_Tensor_0_3(self: Tensor, other: Tensor) -> Tensor:\n  if (self.is_floating_point() or other.is_floating_point()):\n    return self.true_divide(other)\n  return self.divide(other, rounding_mode=\'trunc\')\n)SCRIPT"},\n```\n\n2. In caffe2/torch/csrc/jit/operator_upgraders/version_map.h, add changes like below.\nYou will need to make sure that the entry is SORTED according to the version bump number.\n```\n    {"div.Tensor",\n      {{4,\n        "div_Tensor_0_3",\n        "aten::div.Tensor(Tensor self, Tensor other) -> Tensor"}}},\n```\n\n3. After rebuild PyTorch, run the following command and it will auto generate a change to\nfbcode/caffe2/torch/csrc/jit/mobile/upgrader_mobile.cpp\n\n```\npython pytorch/torchgen/operator_versions/gen_mobile_upgraders.py\n```\n\n4. Generate the test to cover upgrader.\n\n4.1 Switch the commit before the operator change, and add a module in\n`test/jit/fixtures_srcs/fixtures_src.py`. The reason why switching to commit is that,\nan old model with the old operator before the change is needed to ensure the upgrader\nis working as expected. In `test/jit/fixtures_srcs/generate_models.py`, add the module and\nit\'s corresponding changed operator like following\n```\nALL_MODULES = {\n    TestVersionedDivTensorExampleV7(): "aten::div.Tensor",\n}\n```\nThis module should includes the changed operator. If the operator isn\'t covered in the model,\nthe model export process in step 4.2 will fail.\n\n4.2 Export the model to `test/jit/fixtures` by running\n```\npython /Users/chenlai/pytorch/test/jit/fixtures_src/generate_models.py\n```\n\n4.3 In `test/jit/test_save_load_for_op_version.py`, add a test to cover the old models and\nensure the result is equivalent between current module and old module + upgrader.\n\n4.4 Save all change in 4.1, 4.2 and 4.3, as well as previous changes made in step 1, 2, 3.\nSubmit a pr\n\n'
"\nA map of test modules and it's according changed operator\nkey: test module\nvalue: changed operator\n"
ALL_MODULES = {TestVersionedDivTensorExampleV7(): 'aten::div.Tensor', TestVersionedLinspaceV7(): 'aten::linspace', TestVersionedLinspaceOutV7(): 'aten::linspace.out', TestVersionedLogspaceV8(): 'aten::logspace', TestVersionedLogspaceOutV8(): 'aten::logspace.out', TestVersionedGeluV9(): 'aten::gelu', TestVersionedGeluOutV9(): 'aten::gelu.out', TestVersionedRandomV10(): 'aten::random_.from', TestVersionedRandomFuncV10(): 'aten::random.from', TestVersionedRandomOutV10(): 'aten::random.from_out'}
'\nGet the path to `test/jit/fixtures`, where all test models for operator changes\n(upgrader/downgrader) are stored\n'

def get_fixtures_path() -> Path:
    if False:
        while True:
            i = 10
    pytorch_dir = Path(__file__).resolve().parents[3]
    fixtures_path = pytorch_dir / 'test' / 'jit' / 'fixtures'
    return fixtures_path
"\nGet all models' name in `test/jit/fixtures`\n"

def get_all_models(model_directory_path: Path) -> Set[str]:
    if False:
        i = 10
        return i + 15
    files_in_fixtures = model_directory_path.glob('**/*')
    all_models_from_fixtures = [fixture.stem for fixture in files_in_fixtures if fixture.is_file()]
    return set(all_models_from_fixtures)
'\nCheck if a given model already exist in `test/jit/fixtures`\n'

def model_exist(model_file_name: str, all_models: Set[str]) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return model_file_name in all_models
'\nGet the operator list given a module\n'

def get_operator_list(script_module: torch) -> Set[str]:
    if False:
        i = 10
        return i + 15
    buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
    buffer.seek(0)
    mobile_module = _load_for_lite_interpreter(buffer)
    operator_list = _export_operator_list(mobile_module)
    return operator_list
'\nGet the output model operator version, given a module\n'

def get_output_model_version(script_module: torch.nn.Module) -> int:
    if False:
        for i in range(10):
            print('nop')
    buffer = io.BytesIO()
    torch.jit.save(script_module, buffer)
    buffer.seek(0)
    zipped_model = zipfile.ZipFile(buffer)
    try:
        version = int(zipped_model.read('archive/version').decode('utf-8'))
        return version
    except KeyError:
        version = int(zipped_model.read('archive/.data/version').decode('utf-8'))
        return version
"\nLoop through all test modules. If the corresponding model doesn't exist in\n`test/jit/fixtures`, generate one. For the following reason, a model won't be exported:\n\n1. The test module doens't cover the changed operator. For example, test_versioned_div_tensor_example_v4\nis supposed to test the operator aten::div.Tensor. If the model doesn't include this operator, it will fail.\nThe error message includes the actual operator list from the model.\n\n2. The output model version is not the same as expected version. For example, test_versioned_div_tensor_example_v4\nis used to test an operator change aten::div.Tensor, and the operator version will be bumped to v5. This script is\nsupposed to run before the operator change (before the commit to make the change). If the actual model version is v5,\nlikely this script is running with the commit to make the change.\n\n3. The model already exists in `test/jit/fixtures`.\n\n"

def generate_models(model_directory_path: Path):
    if False:
        i = 10
        return i + 15
    all_models = get_all_models(model_directory_path)
    for (a_module, expect_operator) in ALL_MODULES.items():
        torch_module_name = type(a_module).__name__
        if not isinstance(a_module, torch.nn.Module):
            logger.error("The module %s is not a torch.nn.module instance. Please ensure it's a subclass of torch.nn.module in fixtures_src.pyand it's registered as an instance in ALL_MODULES in generated_models.py", torch_module_name)
        model_name = ''.join(['_' + char.lower() if char.isupper() else char for char in torch_module_name]).lstrip('_')
        logger.info('Processing %s', torch_module_name)
        if model_exist(model_name, all_models):
            logger.info('Model %s already exists, skipping', model_name)
            continue
        script_module = torch.jit.script(a_module)
        actual_model_version = get_output_model_version(script_module)
        current_operator_version = torch._C._get_max_operator_version()
        if actual_model_version >= current_operator_version + 1:
            logger.error('Actual model version %s is equal or larger than %s + 1. Please run the script before the commit to change operator.', actual_model_version, current_operator_version)
            continue
        actual_operator_list = get_operator_list(script_module)
        if expect_operator not in actual_operator_list:
            logger.error("The model includes operator: %s, however it doesn't cover the operator %s.Please ensure the output model includes the tested operator.", actual_operator_list, expect_operator)
            continue
        export_model_path = str(model_directory_path / (str(model_name) + '.ptl'))
        script_module._save_for_lite_interpreter(export_model_path)
        logger.info("Generating model %s and it's save to %s", model_name, export_model_path)

def main() -> None:
    if False:
        i = 10
        return i + 15
    model_directory_path = get_fixtures_path()
    generate_models(model_directory_path)
if __name__ == '__main__':
    main()