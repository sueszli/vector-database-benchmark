from . import Program
_already_patch_program = False
global_prog_seed = 0

def monkey_patch_program():
    if False:
        for i in range(10):
            print('nop')

    def global_seed(self, seed=0):
        if False:
            print('Hello World!')
        global global_prog_seed
        global_prog_seed = seed
        self._seed = global_prog_seed
    global _already_patch_program
    if not _already_patch_program:
        Program.global_seed = global_seed
        global global_prog_seed
        Program._seed = global_prog_seed
        _already_patch_program = True