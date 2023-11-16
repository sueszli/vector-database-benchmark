from collections.abc import Iterator
import pytest
from sqlalchemy.orm.session import Session

def test_user_favorite_tag(mocker):
    if False:
        print('Hello World!')
    from superset.daos.tag import TagDAO
    mock_session = mocker.patch('superset.daos.tag.db.session')
    mock_TagDAO = mocker.patch('superset.daos.tag.TagDAO')
    mock_TagDAO.find_by_id.return_value = mocker.MagicMock(users_favorited=[])
    mock_g = mocker.patch('superset.daos.tag.g')
    mock_g.user = mocker.MagicMock()
    TagDAO.favorite_tag_by_id_for_current_user(123)
    mock_TagDAO.find_by_id.assert_called_once_with(123)
    assert mock_TagDAO.find_by_id().users_favorited == [mock_g.user]
    mock_session.commit.assert_called_once()

def test_remove_user_favorite_tag(mocker):
    if False:
        while True:
            i = 10
    from superset.daos.tag import TagDAO
    mock_session = mocker.patch('superset.daos.tag.db.session')
    mock_TagDAO = mocker.patch('superset.daos.tag.TagDAO')
    mock_tag = mocker.MagicMock(users_favorited=[])
    mock_TagDAO.find_by_id.return_value = mock_tag
    mock_g = mocker.patch('superset.daos.tag.g')
    mock_user = mocker.MagicMock()
    mock_g.user = mock_user
    mock_tag.users_favorited.append(mock_user)
    TagDAO.remove_user_favorite_tag(123)
    mock_TagDAO.find_by_id.assert_called_once_with(123)
    assert mock_user not in mock_tag.users_favorited
    mock_session.commit.assert_called_once()

def test_remove_user_favorite_tag_no_user(mocker):
    if False:
        return 10
    from superset.daos.tag import TagDAO
    from superset.exceptions import MissingUserContextException
    mock_session = mocker.patch('superset.daos.tag.db.session')
    mock_TagDAO = mocker.patch('superset.daos.tag.TagDAO')
    mock_tag = mocker.MagicMock(users_favorited=[])
    mock_TagDAO.find_by_id.return_value = mock_tag
    mock_g = mocker.patch('superset.daos.tag.g')
    mock_g.user = None
    with pytest.raises(MissingUserContextException):
        TagDAO.remove_user_favorite_tag(1)

def test_remove_user_favorite_tag_exc_raise(mocker):
    if False:
        for i in range(10):
            print('nop')
    from superset.daos.tag import TagDAO
    from superset.exceptions import MissingUserContextException
    mock_session = mocker.patch('superset.daos.tag.db.session')
    mock_TagDAO = mocker.patch('superset.daos.tag.TagDAO')
    mock_tag = mocker.MagicMock(users_favorited=[])
    mock_TagDAO.find_by_id.return_value = mock_tag
    mock_g = mocker.patch('superset.daos.tag.g')
    mock_session.commit.side_effect = Exception('DB Error')
    with pytest.raises(Exception):
        TagDAO.remove_user_favorite_tag(1)

def test_user_favorite_tag_no_user(mocker):
    if False:
        i = 10
        return i + 15
    from superset.daos.tag import TagDAO
    from superset.exceptions import MissingUserContextException
    mock_session = mocker.patch('superset.daos.tag.db.session')
    mock_TagDAO = mocker.patch('superset.daos.tag.TagDAO')
    mock_tag = mocker.MagicMock(users_favorited=[])
    mock_TagDAO.find_by_id.return_value = mock_tag
    mock_g = mocker.patch('superset.daos.tag.g')
    mock_g.user = None
    with pytest.raises(MissingUserContextException):
        TagDAO.favorite_tag_by_id_for_current_user(1)

def test_user_favorite_tag_exc_raise(mocker):
    if False:
        print('Hello World!')
    from superset.daos.tag import TagDAO
    from superset.exceptions import MissingUserContextException
    mock_session = mocker.patch('superset.daos.tag.db.session')
    mock_TagDAO = mocker.patch('superset.daos.tag.TagDAO')
    mock_tag = mocker.MagicMock(users_favorited=[])
    mock_TagDAO.find_by_id.return_value = mock_tag
    mock_g = mocker.patch('superset.daos.tag.g')
    mock_session.commit.side_effect = Exception('DB Error')
    with pytest.raises(Exception):
        TagDAO.remove_user_favorite_tag(1)

def test_create_tag_relationship(mocker):
    if False:
        return 10
    from superset.daos.tag import TagDAO
    from superset.tags.models import ObjectType, TaggedObject
    mock_session = mocker.patch('superset.daos.tag.db.session')
    objects_to_tag = [(ObjectType.query, 1), (ObjectType.chart, 2), (ObjectType.dashboard, 3)]
    tag = TagDAO.get_by_name('test_tag')
    TagDAO.create_tag_relationship(objects_to_tag, tag)
    assert mock_session.add_all.call_count == 1
    assert len(mock_session.add_all.call_args[0][0]) == len(objects_to_tag)