from docs_snippets.guides.dagster.enriching_with_software_defined_assets.sda_graph import defs

def test_sda_graph():
    if False:
        while True:
            i = 10
    assert defs.get_job_def('products_and_categories_job').execute_in_process().success