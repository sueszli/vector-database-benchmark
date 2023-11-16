from autogpt.config.ai_directives import AIDirectives
'\nTest cases for the PromptConfig class, which handles loads the Prompts configuration\nsettings from a YAML file.\n'

def test_prompt_config_loading(tmp_path):
    if False:
        print('Hello World!')
    'Test if the prompt configuration loads correctly'
    yaml_content = '\nconstraints:\n- A test constraint\n- Another test constraint\n- A third test constraint\nresources:\n- A test resource\n- Another test resource\n- A third test resource\nbest_practices:\n- A test best-practice\n- Another test best-practice\n- A third test best-practice\n'
    prompt_settings_file = tmp_path / 'test_prompt_settings.yaml'
    prompt_settings_file.write_text(yaml_content)
    prompt_config = AIDirectives.from_file(prompt_settings_file)
    assert len(prompt_config.constraints) == 3
    assert prompt_config.constraints[0] == 'A test constraint'
    assert prompt_config.constraints[1] == 'Another test constraint'
    assert prompt_config.constraints[2] == 'A third test constraint'
    assert len(prompt_config.resources) == 3
    assert prompt_config.resources[0] == 'A test resource'
    assert prompt_config.resources[1] == 'Another test resource'
    assert prompt_config.resources[2] == 'A third test resource'
    assert len(prompt_config.best_practices) == 3
    assert prompt_config.best_practices[0] == 'A test best-practice'
    assert prompt_config.best_practices[1] == 'Another test best-practice'
    assert prompt_config.best_practices[2] == 'A third test best-practice'