from os import path
package_dir = path.dirname(path.abspath(__file__))
template_path = path.join(package_dir)

def get_path():
    if False:
        i = 10
        return i + 15
    return template_path