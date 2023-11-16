import dagger
DAGGER_FIELD_NAME_TO_DOCKERFILE_INSTRUCTION = {'from': lambda field: f"FROM {field.args.get('address')}", 'withExec': lambda field: f"RUN {' '.join(field.args.get('args'))}", 'withEnvVariable': lambda field: f"ENV {field.args.get('name')}={field.args.get('value')}", 'withLabel': lambda field: f"LABEL {field.args.get('name')}={field.args.get('value')}"}

def get_container_dockerfile(container) -> str:
    if False:
        return 10
    'Returns the Dockerfile of the base image container.\n    Disclaimer: THIS IS HIGHLY EXPERIMENTAL, HACKY AND BRITTLE.\n    TODO: CONFIRM WITH THE DAGGER TEAM WHAT CAN GO WRONG HERE.\n    Returns:\n        str: The Dockerfile of the base image container.\n    '
    lineage = [field for field in list(container._ctx.selections) if isinstance(field, dagger.api.base.Field) and field.type_name == 'Container']
    dockerfile = []
    for field in lineage:
        if field.name in DAGGER_FIELD_NAME_TO_DOCKERFILE_INSTRUCTION:
            try:
                dockerfile.append(DAGGER_FIELD_NAME_TO_DOCKERFILE_INSTRUCTION[field.name](field))
            except KeyError:
                raise KeyError(f'Unknown field name: {field.name}, please add it to the DAGGER_FIELD_NAME_TO_DOCKERFILE_INSTRUCTION mapping.')
    return '\n'.join(dockerfile)