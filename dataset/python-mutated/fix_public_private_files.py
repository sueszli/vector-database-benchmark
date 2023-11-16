import frappe

def execute():
    if False:
        i = 10
        return i + 15
    files = frappe.get_all('File', fields=['is_private', 'file_url', 'name'], filters={'is_folder': 0})
    for file in files:
        file_url = file.file_url or ''
        if file.is_private:
            if not file_url.startswith('/private/files/'):
                generate_file(file.name)
        elif file_url.startswith('/private/files/'):
            generate_file(file.name)

def generate_file(file_name):
    if False:
        print('Hello World!')
    try:
        file_doc = frappe.get_doc('File', file_name)
        new_doc = frappe.new_doc('File')
        new_doc.is_private = file_doc.is_private
        new_doc.file_name = file_doc.file_name
        new_doc.save_file(content=file_doc.get_content(), ignore_existing_file_check=True)
        file_doc.file_url = new_doc.file_url
        file_doc.save()
    except OSError:
        pass
    except Exception as e:
        print(e)