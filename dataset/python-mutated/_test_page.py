def get_context(context):
    if False:
        for i in range(10):
            print('nop')
    context.base_template_path = 'frappe/templates/test/_test_base_breadcrumbs.html'
    context.add_breadcrumbs = 1