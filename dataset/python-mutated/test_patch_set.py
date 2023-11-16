from sentry.utils.patch_set import patch_to_file_changes

def test_filename_containing_spaces():
    if False:
        print('Hello World!')
    patch = 'diff --git a/has spaces/t.sql b/has spaces/t.sql\nnew file mode 100644\nindex 0000000..8a9b485\n--- /dev/null\n+++ b/has spaces/t.sql\n@@ -0,0 +1 @@\n+select * FROM t;\n'
    expected = [{'path': 'has spaces/t.sql', 'type': 'A'}]
    assert patch_to_file_changes(patch) == expected