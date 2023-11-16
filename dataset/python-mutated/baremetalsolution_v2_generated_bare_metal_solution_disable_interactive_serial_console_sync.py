from google.cloud import bare_metal_solution_v2

def sample_disable_interactive_serial_console():
    if False:
        for i in range(10):
            print('nop')
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.DisableInteractiveSerialConsoleRequest(name='name_value')
    operation = client.disable_interactive_serial_console(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)