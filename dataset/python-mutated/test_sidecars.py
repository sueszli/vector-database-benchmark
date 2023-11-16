def test_default(ecs, instance, launch_run):
    if False:
        for i in range(10):
            print('nop')
    initial_task_definitions = ecs.list_task_definitions()['taskDefinitionArns']
    launch_run(instance)
    task_definitions = ecs.list_task_definitions()['taskDefinitionArns']
    assert len(task_definitions) == len(initial_task_definitions) + 1
    task_definition_arn = next(iter(set(task_definitions).difference(initial_task_definitions)))
    task_definition = ecs.describe_task_definition(taskDefinition=task_definition_arn)
    container_definitions = task_definition['taskDefinition']['containerDefinitions']
    assert len(container_definitions) == 1
    assert not container_definitions[0].get('dependsOn')

def test_include_sidecars_with_depends_on(ecs, instance_cm, launch_run, task_definition):
    if False:
        for i in range(10):
            print('nop')
    with instance_cm({'include_sidecars': True}) as instance:
        initial_task_definitions = ecs.list_task_definitions()['taskDefinitionArns']
        launch_run(instance)
        task_definitions = ecs.list_task_definitions()['taskDefinitionArns']
        assert len(task_definitions) == len(initial_task_definitions) + 1
        task_definition_arn = next(iter(set(task_definitions).difference(initial_task_definitions)))
        task_definition = ecs.describe_task_definition(taskDefinition=task_definition_arn)
        container_definitions = task_definition['taskDefinition']['containerDefinitions']
        assert len(container_definitions) == 2
        for container_definition in container_definitions:
            if container_definition.get('name') == 'run':
                assert container_definition.get('dependsOn')