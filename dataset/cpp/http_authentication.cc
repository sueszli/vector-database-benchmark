#include <cassert>

#include <lithium.hh>

#include "symbols.hh"
#include "test.hh"

int main() {
  using namespace li;

  auto db = sqlite_database("blog_database.sqlite");
  auto users =
      li::sql_orm_schema(db, "blog_users")
          .fields(s::id(s::auto_increment, s::primary_key) = int(), s::login = std::string(),
                  s::password = std::string(), s::secret_key = std::string());

  auto sessions = li::hashmap_http_session("test_cookie", s::user_id = -1);

  users.connect().drop_table_if_exists().create_table_if_not_exists();

  auto auth = li::http_authentication(
      sessions, users, s::login, s::password,
      s::hash_password = [&](auto login, auto password) { return password + "hash"; });

  li::api<li::http_request, li::http_response> my_api;

  my_api.post("/login_invalid") = [&](http_request& request, http_response& response) {
    assert(!auth.login(request, response));
  };

  my_api.post("/login_valid") = [&](http_request& request, http_response& response) {
    assert(auth.login(request, response));
  };

  my_api.get("/logout") = [&](http_request& request, http_response& response) {
    auth.logout(request, response);
  };

  my_api.get("/who_am_i") = [&](http_request& request, http_response& response) {
    response.write(auth.current_user(request, response)->login);
  };

  my_api.get("/logged_out") = [&](http_request& request, http_response& response) {
    assert(!auth.current_user(request, response));
  };

  http_serve(my_api, 12353, s::non_blocking);
  auto c = http_client("http://localhost:12353");

  // Login should fail if no user in db.
  auto r =
      c.post("/login_invalid", s::post_parameters = mmm(s::login = "hello", s::password = "test"));
  std::cout << json_encode(r) << std::endl;
  CHECK_EQUAL(
      "invalid",
      c.post("/login_invalid", s::post_parameters = mmm(s::login = "hello", s::password = "test"))
          .status,
      200);

  users.connect().insert(s::login = "hello", s::password = "testhash");
  CHECK_EQUAL(
      "valid",
      c.post("/login_invalid", s::post_parameters = mmm(s::login = "hello", s::password = "testx"))
          .status,
      200);
  CHECK_EQUAL(
      "valid",
      c.post("/login_valid", s::post_parameters = mmm(s::login = "hello", s::password = "test"))
          .status,
      200);

  CHECK_EQUAL("who_am_i", c.get("/who_am_i").body, "hello");
  CHECK_EQUAL("logout", c.get("/logout").status, 200);
  CHECK_EQUAL("logout", c.get("/logged_out").status, 200);
}
