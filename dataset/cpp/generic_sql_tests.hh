#pragma once

#include <iostream>

template <typename... T> std::ostream& operator<<(std::ostream& os, std::tuple<T...> t) {
  os << "TUPLE<";
  li::tuple_map(std::forward<std::tuple<T...>>(t), [&os](auto v) { os << v << " ,"; });
  os << ">";
  return os;
}

#define EXPECT_THROW(STM)                                                                          \
  try {                                                                                            \
    STM;                                                                                           \
    std::cerr << "This must have thrown an exception: " << #STM << std::endl;                      \
    assert(0);                                                                                     \
  } catch (...) {                                                                                  \
  }

#define EXPECT_EQUAL(A, B)                                                                         \
  if ((A) != (B)) {                                                                                \
    std::cerr << #A << " (== " << A << ") "                                                        \
              << " != " << #B << " (== " << B << ") " << std::endl;                                \
    assert(0);                                                                                     \
  }

#define EXPECT(A)                                                                                  \
  if (!(A)) {                                                                                      \
    std::cerr << #A << " (== " << (A) << ") must be true" << std::endl;                            \
    assert(0);                                                                                     \
  }

template <typename DB, typename C> void init_test_table(C&& query) {
  query("DROP table if exists users_test;");

  if constexpr (std::is_same_v<typename DB::db_tag, li::pgsql_tag>)
    query("CREATE TABLE users_test (id int,name TEXT,age int);");
  else
    query("CREATE TABLE users_test (id int,name LONGTEXT,age int);");
}

template <typename DB, typename R, typename S> void test_result(R&& query, S&& new_query) {

  // std::cout << " STRING TO INT " << std::endl;
  // std::cout << "query(SELECT 'xxx';).template read<int>() == " << query("SELECT 'xxx';").template
  // read<int>() << std::endl;

  // new_query("SELECTT 1+2").template read<int>();
  // Invalid queries must throw.
  EXPECT_THROW(new_query("SELECTTT 1+2").template read<int>());
  // //   long long int affected_rows();
  // //   template <typename T> T read();
  EXPECT_EQUAL(3, query("SELECT 1+2").template read<int>());
  EXPECT_EQUAL(-1, query("SELECT 1-2").template read<int>());
  EXPECT_EQUAL(query("SELECT 'xxx'").template read<std::string>(), "xxx");
  init_test_table<DB>(query);
  // Reading empty sets must throw.
  EXPECT_THROW(query("SELECT age from users_test;").template read<int>());
  // Invalid read types must throw.
  EXPECT_THROW(query("SELECT 'xxx';").template read<int>());
  // Mismatch in number of fields must throw.
  EXPECT_THROW((query("SELECT 'xxx';").template read<std::string, int>()));

  int i = 0;
  query("SELECT 1+2").read(i);
  EXPECT_EQUAL(i, 3);

  std::string str;
  query("SELECT 'xxx'").read(str);
  EXPECT_EQUAL(str, "xxx");

  std::tuple<int, std::string, int> tp;
  query("SELECT 1,'xxx',3 ").read(tp);
  EXPECT_EQUAL(tp, (std::make_tuple(1, "xxx", 3)));

  int i1 = 0;
  int i2 = 0;
  str = "";
  query("SELECT 1,'xxx',3 ").read(i1, str, i2);
  EXPECT_EQUAL((std::tie(i1, str, i2)), (std::make_tuple(1, "xxx", 3)));

  //   template <typename T> std::optional<T> read_optional();
  EXPECT_EQUAL(0, query("SELECT count(*) from users_test;").template read<int>());

  EXPECT(!query("SELECT age from users_test;").template read_optional<int>().has_value());
  EXPECT_EQUAL(3, query("SELECT 1+2").template read_optional<int>().value());

  //   template <typename T> void read(std::optional<T>& o);
  std::optional<int> opt;
  query("SELECT 1+2").read(opt);
  EXPECT_EQUAL(opt.value(), 3);

  query("INSERT into users_test(id, name, age) values (1,'a',41);");
  query("INSERT into users_test(id, name, age) values (2,'b',42);");
  query("INSERT into users_test(id, name, age) values (3,'c',43);");
  EXPECT_EQUAL(3, query("SELECT count(*) from users_test").template read<int>());

  int index = 1;
  query("SELECT id, name, age from users_test;").map([&](int id, std::string name, int age) {
    EXPECT_EQUAL(id, index);
    EXPECT_EQUAL(age, index + 40);
    EXPECT_EQUAL(name.size(), 1);
    EXPECT_EQUAL(name[0], 'a' + index - 1);
    index++;
  });

  // map with growing string
  init_test_table<DB>(query);
  query("INSERT into users_test(id, name, age) values (1,'aaaaaaaaaaaaaaaaaaa',41);");
  query("INSERT into users_test(id, name, age) values "
        "(2,'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',42);");
  query("INSERT into users_test(id, name, age) values "
        "(3,'"
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
        "cccccccccccccccccccccccc',43);");
  int sizes[] = {19, 57, 114};
  index = 0;
  query("SELECT name from users_test order by id;").map([&](std::string name) {
    EXPECT_EQUAL(name.size(), sizes[index]);
    index++;
  });
}

template <typename D> std::string placeholder(int pos) {
  if constexpr (std::is_same_v<typename D::db_tag, li::pgsql_tag>) {
    std::stringstream ss;
    ss << "$" << pos;
    return ss.str();
  } else
    return "?";
}

template <typename DB, typename Q> void test_long_strings_prepared_statement(Q& query) {
  init_test_table<DB>(query);

  auto insert_user =
      query.prepare(std::string("INSERT into users_test(id, name, age) values (") +
                    placeholder<Q>(1) + "," + placeholder<Q>(2) + "," + placeholder<Q>(3) + ");");

  std::string test_str_patern = "lkjlkjdgfad0875g9f658g8w97f32orjw0r89ofuhq07fy0rjgq3478fyqh03g7y0b"
                                "347fyj08yg034f78yj047yh078fy0fyj40";
  std::string test_str = "";
  for (int k = 0; k < 10; k++) {
    for (int l = 0; l < 100; l++)
      test_str += test_str_patern;
    insert_user(k+1, test_str, 42);
  }

  test_str = "";
  query.prepare("Select id,name from users_test;")().map([&](int id, std::string name) {
    // std::cout << "SQL TEST PREPARED LONG STRING id:" << id << " " << 10000 * id << " " << name.size() << std::endl;
    // std::cout << name.size() / test_str_patern.size() << std::endl;
    EXPECT_EQUAL(name.size(), 10000 * id);
    for (int l = 0; l < 100; l++)
      test_str += test_str_patern;
    EXPECT_EQUAL(name, test_str);
  });

  test_str = "";
  query("Select id,name from users_test;").map([&](int id, std::string name) {
    // std::cout << "SQL TEST LONG STRING " << 10000 * id << std::endl;
    EXPECT_EQUAL(name.size(), 10000 * id);

    for (int l = 0; l < 100; l++)
      test_str += test_str_patern;
    EXPECT_EQUAL(name, test_str);
  });
}

template <typename D> void generic_sql_tests(D& database) {

  auto query = database.connect();

  // try {
  //   // database.connect();
  //   auto fun = [&](std::string q) { return database.connect()(q); };
  //   fun("SELECTT 1+2").template read<int>();
  // } catch(...) {}

  try {
    // query.prepare("SELECT 2+2")().template read<int>();
    // query.prepare()
    // database.connect();
    auto fun = [&](std::string q) { return database.connect().prepare(q)(); };
    fun("SELECTT 1+2").template read<int>();
  } catch (...) {
  }

  // auto fun = [&](std::string q) { return database.connect().prepare(q)(); };
  // fun("SELECTT 1+2").template read<int>();
  // query.prepare("SELECT 1+2")().template read<int>();
  // query.prepare("SELECT 2+2")().template read<int>();
  // test_result([&](std::string q) { return query.prepare(q)(); }, [&](std::string q) { return
  // database.connect().prepare(q)(); }); return;

  // test non prepared statement result.
  test_result<D>(query, [&](std::string q) { return database.connect()(q); });
  // test prepared statement result.
  test_result<D>([&](std::string q) { return query.prepare(q)(); },
                 [&](std::string q) { return database.connect().prepare(q)(); });
  test_long_strings_prepared_statement<D>(query);

  init_test_table<D>(query);

  // Prepared statement.
  auto insert_user =
      query.prepare(std::string("INSERT into users_test(id, name, age) values (") +
                    placeholder<D>(1) + "," + placeholder<D>(2) + "," + placeholder<D>(3) + ");");

  insert_user(1, "John", 42);
  insert_user(2, "Bob", 24);
  EXPECT_EQUAL(
      (std::make_tuple("John", 42)),
      (query("select name, age from users_test where id = 1").template read<std::string, int>()));
  EXPECT_EQUAL(
      (std::make_tuple("Bob", 24)),
      (query("select name, age from users_test where id = 2").template read<std::string, int>()));

  // Large string
  // std::string large(1000000, 'x');
  // query("INSERT into users_test(id, name, age) values (4,'" + large + "',43);");

  //   std::string large_out = query("SELECT name from users_test where id = 4").template
  //   read<std::string>(); std::cout << large_out << std::endl; EXPECT_EQUAL(large_out.size(),
  //   1000000); EXPECT_EQUAL(large_out, large);
}