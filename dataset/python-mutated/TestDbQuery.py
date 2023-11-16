import re
from Db.DbQuery import DbQuery

class TestDbQuery:

    def testParse(self):
        if False:
            while True:
                i = 10
        query_text = "\n            SELECT\n             'comment' AS type,\n             date_added, post.title AS title,\n             keyvalue.value || ': ' || comment.body AS body,\n             '?Post:' || comment.post_id || '#Comments' AS url\n            FROM\n             comment\n             LEFT JOIN json USING (json_id)\n             LEFT JOIN json AS json_content ON (json_content.directory = json.directory AND json_content.file_name='content.json')\n             LEFT JOIN keyvalue ON (keyvalue.json_id = json_content.json_id AND key = 'cert_user_id')\n             LEFT JOIN post ON (comment.post_id = post.post_id)\n            WHERE\n             post.date_added > 123\n            ORDER BY\n             date_added DESC\n            LIMIT 20\n        "
        query = DbQuery(query_text)
        assert query.parts['LIMIT'] == '20'
        assert query.fields['body'] == "keyvalue.value || ': ' || comment.body"
        assert re.sub('[ \r\n]', '', str(query)) == re.sub('[ \r\n]', '', query_text)
        query.wheres.append("body LIKE '%hello%'")
        assert "body LIKE '%hello%'" in str(query)