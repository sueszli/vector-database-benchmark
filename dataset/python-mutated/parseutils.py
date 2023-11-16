import re
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Function
from sqlparse.tokens import Keyword, DML, Punctuation
cleanup_regex = {'alphanum_underscore': re.compile('(\\w+)$'), 'many_punctuations': re.compile('([^():,\\s]+)$'), 'most_punctuations': re.compile('([^\\.():,\\s]+)$'), 'all_punctuations': re.compile('([^\\s]+)$')}

def last_word(text, include='alphanum_underscore'):
    if False:
        return 10
    "\n    Find the last word in a sentence.\n\n    >>> last_word('abc')\n    'abc'\n    >>> last_word(' abc')\n    'abc'\n    >>> last_word('')\n    ''\n    >>> last_word(' ')\n    ''\n    >>> last_word('abc ')\n    ''\n    >>> last_word('abc def')\n    'def'\n    >>> last_word('abc def ')\n    ''\n    >>> last_word('abc def;')\n    ''\n    >>> last_word('bac $def')\n    'def'\n    >>> last_word('bac $def', include='most_punctuations')\n    '$def'\n    >>> last_word('bac \\def', include='most_punctuations')\n    '\\\\\\\\def'\n    >>> last_word('bac \\def;', include='most_punctuations')\n    '\\\\\\\\def;'\n    >>> last_word('bac::def', include='most_punctuations')\n    'def'\n    "
    if not text:
        return ''
    if text[-1].isspace():
        return ''
    else:
        regex = cleanup_regex[include]
        matches = regex.search(text)
        if matches:
            return matches.group(0)
        else:
            return ''

def is_subselect(parsed):
    if False:
        while True:
            i = 10
    if not parsed.is_group:
        return False
    for item in parsed.tokens:
        if item.ttype is DML and item.value.upper() in ('SELECT', 'INSERT', 'UPDATE', 'CREATE', 'DELETE'):
            return True
    return False

def extract_from_part(parsed, stop_at_punctuation=True):
    if False:
        return 10
    tbl_prefix_seen = False
    for item in parsed.tokens:
        if tbl_prefix_seen:
            if is_subselect(item):
                for x in extract_from_part(item, stop_at_punctuation):
                    yield x
            elif stop_at_punctuation and item.ttype is Punctuation:
                return
            elif item.ttype is Keyword and item.value.upper() == 'ON':
                tbl_prefix_seen = False
                continue
            elif item.ttype is Keyword and (not item.value.upper() == 'FROM') and (not item.value.upper().endswith('JOIN')):
                return
            else:
                yield item
        elif (item.ttype is Keyword or item.ttype is Keyword.DML) and item.value.upper() in ('COPY', 'FROM', 'INTO', 'UPDATE', 'TABLE', 'JOIN'):
            tbl_prefix_seen = True
        elif isinstance(item, IdentifierList):
            for identifier in item.get_identifiers():
                if identifier.ttype is Keyword and identifier.value.upper() == 'FROM':
                    tbl_prefix_seen = True
                    break

def extract_table_identifiers(token_stream):
    if False:
        i = 10
        return i + 15
    'yields tuples of (schema_name, table_name, table_alias)'
    for item in token_stream:
        if isinstance(item, IdentifierList):
            for identifier in item.get_identifiers():
                try:
                    schema_name = identifier.get_parent_name()
                    real_name = identifier.get_real_name()
                except AttributeError:
                    continue
                if real_name:
                    yield (schema_name, real_name, identifier.get_alias())
        elif isinstance(item, Identifier):
            real_name = item.get_real_name()
            schema_name = item.get_parent_name()
            if real_name:
                yield (schema_name, real_name, item.get_alias())
            else:
                name = item.get_name()
                yield (None, name, item.get_alias() or name)
        elif isinstance(item, Function):
            yield (None, item.get_name(), item.get_name())

def extract_tables(sql):
    if False:
        for i in range(10):
            print('nop')
    'Extract the table names from an SQL statement.\n\n    Returns a list of (schema, table, alias) tuples\n\n    '
    parsed = sqlparse.parse(sql)
    if not parsed:
        return []
    insert_stmt = parsed[0].token_first().value.lower() == 'insert'
    stream = extract_from_part(parsed[0], stop_at_punctuation=insert_stmt)
    return list(extract_table_identifiers(stream))

def find_prev_keyword(sql):
    if False:
        while True:
            i = 10
    ' Find the last sql keyword in an SQL statement\n\n    Returns the value of the last keyword, and the text of the query with\n    everything after the last keyword stripped\n    '
    if not sql.strip():
        return (None, '')
    parsed = sqlparse.parse(sql)[0]
    flattened = list(parsed.flatten())
    logical_operators = ('AND', 'OR', 'NOT', 'BETWEEN')
    for t in reversed(flattened):
        if t.value == '(' or (t.is_keyword and t.value.upper() not in logical_operators):
            idx = flattened.index(t)
            text = ''.join((tok.value for tok in flattened[:idx + 1]))
            return (t, text)
    return (None, '')

def query_starts_with(query, prefixes):
    if False:
        print('Hello World!')
    'Check if the query starts with any item from *prefixes*.'
    prefixes = [prefix.lower() for prefix in prefixes]
    formatted_sql = sqlparse.format(query.lower(), strip_comments=True)
    return bool(formatted_sql) and formatted_sql.split()[0] in prefixes

def queries_start_with(queries, prefixes):
    if False:
        print('Hello World!')
    'Check if any queries start with any item from *prefixes*.'
    for query in sqlparse.split(queries):
        if query and query_starts_with(query, prefixes) is True:
            return True
    return False

def query_has_where_clause(query):
    if False:
        return 10
    'Check if the query contains a where-clause.'
    return any((isinstance(token, sqlparse.sql.Where) for token_list in sqlparse.parse(query) for token in token_list))

def is_destructive(queries):
    if False:
        return 10
    'Returns if any of the queries in *queries* is destructive.'
    keywords = ('drop', 'shutdown', 'delete', 'truncate', 'alter')
    for query in sqlparse.split(queries):
        if query:
            if query_starts_with(query, keywords) is True:
                return True
            elif query_starts_with(query, ['update']) is True and (not query_has_where_clause(query)):
                return True
    return False
if __name__ == '__main__':
    sql = 'select * from (select t. from tabl t'
    print(extract_tables(sql))

def is_dropping_database(queries, dbname):
    if False:
        print('Hello World!')
    'Determine if the query is dropping a specific database.'
    result = False
    if dbname is None:
        return False

    def normalize_db_name(db):
        if False:
            print('Hello World!')
        return db.lower().strip('`"')
    dbname = normalize_db_name(dbname)
    for query in sqlparse.parse(queries):
        keywords = [t for t in query.tokens if t.is_keyword]
        if len(keywords) < 2:
            continue
        if keywords[0].normalized in ('DROP', 'CREATE') and keywords[1].value.lower() in ('database', 'schema'):
            database_token = next((t for t in query.tokens if isinstance(t, Identifier)), None)
            if database_token is not None and normalize_db_name(database_token.get_name()) == dbname:
                result = keywords[0].normalized == 'DROP'
    return result