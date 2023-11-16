import ckan.plugins as p
import ckanext.datastore.interfaces as interfaces

class SampleDataStorePlugin(p.SingletonPlugin):
    p.implements(interfaces.IDatastore, inherit=True)

    def datastore_validate(self, context, data_dict, column_names):
        if False:
            for i in range(10):
                print('nop')
        valid_filters = ('age_between', 'age_not_between', 'insecure_filter')
        filters = data_dict.get('filters', {})
        for key in list(filters.keys()):
            if key in valid_filters:
                del filters[key]
        return data_dict

    def datastore_search(self, context, data_dict, column_names, query_dict):
        if False:
            i = 10
            return i + 15
        query_dict['where'] += self._where(data_dict)
        return query_dict

    def datastore_delete(self, context, data_dict, column_names, query_dict):
        if False:
            i = 10
            return i + 15
        query_dict['where'] += self._where(data_dict)
        return query_dict

    def _where(self, data_dict):
        if False:
            print('Hello World!')
        filters = data_dict.get('filters', {})
        where_clauses = []
        if 'age_between' in filters:
            age_between = filters['age_between']
            clause = ('"age" >= %s AND "age" <= %s', age_between[0], age_between[1])
            where_clauses.append(clause)
        if 'age_not_between' in filters:
            age_not_between = filters['age_not_between']
            clause = ('"age" < %s OR "age" > %s', age_not_between[0], age_not_between[1])
            where_clauses.append(clause)
        if 'insecure_filter' in filters:
            insecure_filter = filters['insecure_filter']
            clause = (insecure_filter,)
            where_clauses.append(clause)
        return where_clauses