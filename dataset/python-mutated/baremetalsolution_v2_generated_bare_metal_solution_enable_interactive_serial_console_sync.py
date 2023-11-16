from google.cloud import bare_metal_solution_v2

def sample_enable_interactive_serial_console():
    if False:
        while True:
            i = 10
    client = bare_metal_solution_v2.BareMetalSolutionClient()
    request = bare_metal_solution_v2.EnableInteractiveSerialConsoleRequest(name='name_value')
    operation = client.enable_interactive_serial_console(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)