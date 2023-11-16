from frappe.model.document import Document

class RecorderQuery(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        duration: DF.Float
        exact_copies: DF.Int
        explain_result: DF.Text | None
        index: DF.Int
        normalized_copies: DF.Int
        normalized_query: DF.Data | None
        parent: DF.Data
        parentfield: DF.Data
        parenttype: DF.Data
        query: DF.Data
        stack: DF.Text | None
    pass

    def db_insert(self, *args, **kwargs):
        if False:
            return 10
        pass

    def load_from_db(self):
        if False:
            return 10
        pass

    def db_update(self):
        if False:
            print('Hello World!')
        pass

    @staticmethod
    def get_list(args):
        if False:
            for i in range(10):
                print('nop')
        pass

    @staticmethod
    def get_count(args):
        if False:
            return 10
        pass

    @staticmethod
    def get_stats(args):
        if False:
            print('Hello World!')
        pass

    def delete(self):
        if False:
            return 10
        pass