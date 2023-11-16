import sys

def validate_project_name():
    if False:
        i = 10
        return i + 15
    project_name = '{{ cookiecutter.project }}'
    if not project_name.startswith('ckanext-'):
        print("\nERROR: Project name must start with 'ckanext-' > {}".format(project_name))
        sys.exit(1)
if __name__ == '__main__':
    validate_project_name()