from playwright.sync_api import Page, expect
from e2e_playwright.conftest import ImageCompareFunction, wait_for_app_run

def test_file_uploader_render_correctly(themed_app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        i = 10
        return i + 15
    'Test that the file uploader render as expected via screenshot matching.'
    file_uploaders = themed_app.get_by_test_id('stFileUploader')
    expect(file_uploaders).to_have_count(7)
    assert_snapshot(file_uploaders.nth(0), name='st_single-file-uploader')
    assert_snapshot(file_uploaders.nth(1), name='st_disabled-file-uploader')
    assert_snapshot(file_uploaders.nth(2), name='st_multi-file-uploader')
    assert_snapshot(file_uploaders.nth(4), name='st_hidden-label-file-uploader')
    assert_snapshot(file_uploaders.nth(5), name='st_collapsed-label-file-uploader')

def test_file_uploader_error_message_disallowed_files(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        return 10
    'Test that shows error message for disallowed files.'
    file_name1 = 'example.json'
    file_content1 = b'{}'
    uploader_index = 0
    with app.expect_file_chooser() as fc_info:
        app.get_by_test_id('stFileUploadDropzone').nth(uploader_index).click()
    file_chooser = fc_info.value
    file_chooser.set_files(files=[{'name': file_name1, 'mimeType': 'application/json', 'buffer': file_content1}])
    wait_for_app_run(app)
    app.wait_for_timeout(1000)
    expect(app.get_by_test_id('stUploadedFileErrorMessage').nth(uploader_index)).to_have_text('application/json files are not allowed.', use_inner_text=True)
    file_uploader_in_error_state = app.get_by_test_id('stFileUploader').nth(uploader_index)
    assert_snapshot(file_uploader_in_error_state, name='st_file_uploader-error')

def test_uploads_and_deletes_single_file_only(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        i = 10
        return i + 15
    'Test that uploading a file for single file uploader works as expected.'
    file_name1 = 'file1.txt'
    file_content1 = b'file1content'
    file_name2 = 'file2.txt'
    file_content2 = b'file2content'
    uploader_index = 0
    with app.expect_file_chooser() as fc_info:
        app.get_by_test_id('stFileUploadDropzone').nth(uploader_index).click()
    file_chooser = fc_info.value
    file_chooser.set_files(files=[{'name': file_name1, 'mimeType': 'text/plain', 'buffer': file_content1}])
    wait_for_app_run(app)
    app.wait_for_timeout(1000)
    expect(app.locator('.uploadedFileName')).to_have_text(file_name1, use_inner_text=True)
    expect(app.get_by_test_id('stText').nth(uploader_index)).to_have_text(str(file_content1), use_inner_text=True)
    file_uploader_uploaded_state = app.get_by_test_id('stFileUploader').nth(uploader_index)
    assert_snapshot(file_uploader_uploaded_state, name='st_single_file_uploader-uploaded')
    expect(app.get_by_test_id('stMarkdownContainer').nth(uploader_index + 1)).to_have_text('True', use_inner_text=True)
    with app.expect_file_chooser() as fc_info:
        app.get_by_test_id('stFileUploadDropzone').nth(uploader_index).click()
    file_chooser = fc_info.value
    file_chooser.set_files(files=[{'name': file_name2, 'mimeType': 'text/plain', 'buffer': file_content2}])
    wait_for_app_run(app)
    app.wait_for_timeout(1000)
    expect(app.locator('.uploadedFileName')).to_have_text(file_name2, use_inner_text=True)
    expect(app.get_by_test_id('stText').nth(uploader_index)).to_have_text(str(file_content2), use_inner_text=True)
    expect(app.get_by_test_id('stMarkdownContainer').nth(uploader_index + 1)).to_have_text('True', use_inner_text=True)
    app.get_by_test_id('stHeader').press('r')
    wait_for_app_run(app)
    app.wait_for_timeout(1000)
    expect(app.get_by_test_id('stText').nth(uploader_index)).to_have_text(str(file_content2), use_inner_text=True)
    app.get_by_test_id('fileDeleteBtn').nth(uploader_index).click()
    wait_for_app_run(app)
    app.wait_for_timeout(1000)
    expect(app.get_by_test_id('stText').nth(uploader_index)).to_have_text('No upload', use_inner_text=True)

def test_uploads_and_deletes_multiple_files(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        return 10
    'Test that uploading multiple files at once works correctly.'
    file_name1 = 'file1.txt'
    file_content1 = b'file1content'
    file_name2 = 'file2.txt'
    file_content2 = b'file2content'
    files = [{'name': file_name1, 'mimeType': 'text/plain', 'buffer': file_content1}, {'name': file_name2, 'mimeType': 'text/plain', 'buffer': file_content2}]
    uploader_index = 2
    with app.expect_file_chooser() as fc_info:
        app.get_by_test_id('stFileUploadDropzone').nth(uploader_index).click()
    file_chooser = fc_info.value
    file_chooser.set_files(files=files)
    wait_for_app_run(app)
    app.wait_for_timeout(1000)
    uploaded_file_names = app.locator('.uploadedFileName')
    file_names = [files[1]['name'], files[0]['name']]
    for (i, element) in enumerate(uploaded_file_names.all()):
        expect(element).to_have_text(file_names[i], use_inner_text=True)
    content = '\n'.join([files[0]['buffer'].decode('utf-8'), files[1]['buffer'].decode('utf-8')])
    expect(app.get_by_test_id('stText').nth(uploader_index)).to_have_text(content, use_inner_text=True)
    file_uploader = app.get_by_test_id('stFileUploader').nth(uploader_index)
    assert_snapshot(file_uploader, name='st_multi_file_uploader-uploaded')
    app.get_by_test_id('fileDeleteBtn').first.click()
    wait_for_app_run(app)
    app.wait_for_timeout(1000)
    expect(app.get_by_test_id('stText').nth(uploader_index)).to_have_text(files[0]['buffer'].decode('utf-8'), use_inner_text=True)
    expect(app.get_by_test_id('stMarkdownContainer').nth(5)).to_have_text('True', use_inner_text=True)

def test_does_not_call_callback_when_not_changed(app: Page):
    if False:
        while True:
            i = 10
    'Test that the file uploader does not call a callback when not changed.'
    file_name1 = 'example5.txt'
    file_content1 = b'Hello world!'
    uploader_index = 6
    expect(app.get_by_test_id('stText').nth(uploader_index)).to_have_text('0', use_inner_text=True)
    with app.expect_file_chooser() as fc_info:
        app.get_by_test_id('stFileUploadDropzone').nth(uploader_index).click()
    file_chooser = fc_info.value
    file_chooser.set_files(files=[{'name': file_name1, 'mimeType': 'application/json', 'buffer': file_content1}])
    wait_for_app_run(app)
    app.wait_for_timeout(1000)
    expect(app.get_by_test_id('stText').nth(uploader_index)).to_have_text('1', use_inner_text=True)
    app.get_by_test_id('stHeader').press('r')
    wait_for_app_run(app)
    app.wait_for_timeout(1000)
    expect(app.get_by_test_id('stText').nth(uploader_index)).to_have_text('1', use_inner_text=True)