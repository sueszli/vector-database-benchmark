from organisations.chargebee.metadata import ChargebeeObjMetadata

def test_add_chargebee_object_meta_data():
    if False:
        return 10
    a_obj_metadata = ChargebeeObjMetadata(seats=10, api_calls=100)
    another_obj_metadata = ChargebeeObjMetadata(seats=20, api_calls=200, projects=100)
    added_chargebee_obj_metadata = a_obj_metadata + another_obj_metadata
    assert added_chargebee_obj_metadata.seats == 30
    assert added_chargebee_obj_metadata.api_calls == 300
    assert added_chargebee_obj_metadata.projects == 100

def test_multiply_chargebee_object_metadata():
    if False:
        i = 10
        return i + 15
    metadata = ChargebeeObjMetadata(seats=10, api_calls=100)
    new_metadata = metadata * 3
    assert new_metadata.seats == 30
    assert new_metadata.api_calls == 300
    assert new_metadata.projects is None

def test_multiply_chargebee_object_metadata_works_for_null_values():
    if False:
        for i in range(10):
            print('nop')
    metadata = ChargebeeObjMetadata(seats=10, api_calls=100, projects=None)
    new_metadata = metadata * 3
    assert new_metadata.seats == 30
    assert new_metadata.api_calls == 300
    assert new_metadata.projects is None