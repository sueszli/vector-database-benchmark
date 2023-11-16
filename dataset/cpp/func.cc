void foo();
void foo();
void foo() {}
void foo();

/*
// Note: we always use the latest seen ("most local") definition/declaration.
OUTPUT:
{
  "includes": [],
  "skipped_by_preprocessor": [],
  "types": [],
  "funcs": [{
      "id": 0,
      "usr": 4259594751088586730,
      "detailed_name": "void foo()",
      "short_name": "foo",
      "kind": 12,
      "storage": 1,
      "declarations": [{
          "spell": "1:6-1:9|-1|1|1",
          "param_spellings": []
        }, {
          "spell": "2:6-2:9|-1|1|1",
          "param_spellings": []
        }, {
          "spell": "4:6-4:9|-1|1|1",
          "param_spellings": []
        }],
      "spell": "3:6-3:9|-1|1|2",
      "extent": "3:1-3:14|-1|1|0",
      "bases": [],
      "derived": [],
      "vars": [],
      "uses": [],
      "callees": []
    }],
  "vars": []
}
*/
