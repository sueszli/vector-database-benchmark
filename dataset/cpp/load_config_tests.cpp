#include <fstream>
#include <iostream>
#include <string>

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>
#include <boost/system/error_code.hpp>

#include "common/config/config.h"

class LoadConfigTest : public ::testing::Test {
 public:
  virtual void SetUp() { config_.Init(); }

 protected:
  ssf::config::Config config_;
};

TEST_F(LoadConfigTest, LoadNoFileTest) {
  boost::system::error_code ec;

  config_.UpdateFromFile("", ec);

  ASSERT_EQ(ec.value(), 0) << "Success if no file given";
}

TEST_F(LoadConfigTest, LoadNonExistantFileTest) {
  boost::system::error_code ec;

  config_.UpdateFromFile("./config_files/unknown.json", ec);

  ASSERT_NE(ec.value(), 0) << "No success if file not existant";
}

TEST_F(LoadConfigTest, LoadEmptyFileTest) {
  boost::system::error_code ec;

  config_.UpdateFromFile("./config_files/empty.json", ec);

  ASSERT_NE(ec.value(), 0) << "No success if file empty";
}

TEST_F(LoadConfigTest, LoadWrongFormatFileTest) {
  boost::system::error_code ec;

  config_.UpdateFromFile("./config_files/wrong_format.json", ec);

  ASSERT_NE(ec.value(), 0) << "No success if wrong file format";
}

TEST_F(LoadConfigTest, DefaultValueTest) {
  ASSERT_FALSE(config_.tls().ca_cert().IsBuffer());
  ASSERT_EQ(config_.tls().ca_cert().value(), "./certs/trusted/ca.crt");
  ASSERT_FALSE(config_.tls().cert().IsBuffer());
  ASSERT_EQ(config_.tls().cert().value(), "./certs/certificate.crt");
  ASSERT_FALSE(config_.tls().key().IsBuffer());
  ASSERT_EQ(config_.tls().key().value(), "./certs/private.key");
  ASSERT_EQ(config_.tls().key_password(), "");
  ASSERT_FALSE(config_.tls().dh().IsBuffer());
  ASSERT_EQ(config_.tls().dh().value(), "./certs/dh4096.pem");
  ASSERT_EQ(config_.tls().cipher_alg(), "DHE-RSA-AES256-GCM-SHA384");

  ASSERT_EQ(config_.http_proxy().host(), "");
  ASSERT_EQ(config_.http_proxy().port(), "");
  ASSERT_EQ(config_.http_proxy().user_agent(), "");
  ASSERT_EQ(config_.http_proxy().username(), "");
  ASSERT_EQ(config_.http_proxy().domain(), "");
  ASSERT_EQ(config_.http_proxy().password(), "");
  ASSERT_EQ(config_.http_proxy().reuse_kerb(), true);
  ASSERT_EQ(config_.http_proxy().reuse_ntlm(), true);

  ASSERT_EQ(config_.socks_proxy().version(), ssf::network::Socks::Version::kV5);
  ASSERT_EQ(config_.socks_proxy().host(), "");
  ASSERT_EQ(config_.socks_proxy().port(), "1080");

  ASSERT_TRUE(config_.services().datagram_forwarder().enabled());
  ASSERT_TRUE(config_.services().datagram_listener().enabled());
  ASSERT_FALSE(config_.services().datagram_listener().gateway_ports());
  ASSERT_FALSE(config_.services().copy().enabled());
  ASSERT_TRUE(config_.services().socks().enabled());
  ASSERT_TRUE(config_.services().stream_forwarder().enabled());
  ASSERT_TRUE(config_.services().stream_listener().enabled());
  ASSERT_FALSE(config_.services().stream_listener().gateway_ports());
  ASSERT_FALSE(config_.services().process().enabled());

  ASSERT_GT(config_.services().process().path().length(),
            static_cast<std::size_t>(0));
  ASSERT_EQ(config_.services().process().args(), "");
}

TEST_F(LoadConfigTest, LoadTlsPartialFileTest) {
  boost::system::error_code ec;

  config_.UpdateFromFile("./config_files/tls_partial.json", ec);

  ASSERT_EQ(ec.value(), 0) << "Success if partial file format";

  ASSERT_FALSE(config_.tls().ca_cert().IsBuffer());
  ASSERT_EQ(config_.tls().ca_cert().value(), "test_ca_path");
  ASSERT_FALSE(config_.tls().cert().IsBuffer());
  ASSERT_EQ(config_.tls().cert().value(), "./certs/certificate.crt");
  ASSERT_FALSE(config_.tls().key().IsBuffer());
  ASSERT_EQ(config_.tls().key().value(), "test_key_path");
  ASSERT_EQ(config_.tls().key_password(), "");
  ASSERT_FALSE(config_.tls().dh().IsBuffer());
  ASSERT_EQ(config_.tls().dh().value(), "test_dh_path");
  ASSERT_EQ(config_.tls().cipher_alg(), "DHE-RSA-AES256-GCM-SHA384");
}

TEST_F(LoadConfigTest, LoadTlsBufferFileTest) {
  boost::system::error_code ec;

  config_.UpdateFromFile("./config_files/tls_buffer.json", ec);

  ASSERT_EQ(ec.value(), 0) << "Success if buffer file format";

  ASSERT_TRUE(config_.tls().ca_cert().IsBuffer());
  ASSERT_EQ(config_.tls().ca_cert().value(), "test_ca_cert_buffer");
  ASSERT_TRUE(config_.tls().cert().IsBuffer());
  ASSERT_EQ(config_.tls().cert().value(), "test_cert_buffer");
  ASSERT_TRUE(config_.tls().key().IsBuffer());
  ASSERT_EQ(config_.tls().key().value(), "test_key_buffer");
  ASSERT_EQ(config_.tls().key_password(), "test_key_password");
  ASSERT_TRUE(config_.tls().dh().IsBuffer());
  ASSERT_EQ(config_.tls().dh().value(), "test_dh_buffer");
  ASSERT_EQ(config_.tls().cipher_alg(), "test_cipher_alg");
}

TEST_F(LoadConfigTest, LoadTlsCompleteFileTest) {
  boost::system::error_code ec;

  config_.UpdateFromFile("./config_files/tls_complete.json", ec);

  ASSERT_EQ(ec.value(), 0) << "Success if complete file format";

  ASSERT_FALSE(config_.tls().ca_cert().IsBuffer());
  ASSERT_EQ(config_.tls().ca_cert().value(), "test_ca_path");
  ASSERT_FALSE(config_.tls().cert().IsBuffer());
  ASSERT_EQ(config_.tls().cert().value(), "test_cert_path");
  ASSERT_TRUE(config_.tls().key().IsBuffer());
  ASSERT_EQ(config_.tls().key().value(), "test_key_buffer");
  ASSERT_EQ(config_.tls().key_password(), "test_key_password");
  ASSERT_FALSE(config_.tls().dh().IsBuffer());
  ASSERT_EQ(config_.tls().dh().value(), "test_dh_path");
  ASSERT_EQ(config_.tls().cipher_alg(), "test_cipher_alg");

  ASSERT_EQ(config_.http_proxy().host(), "");
  ASSERT_EQ(config_.http_proxy().port(), "");
  ASSERT_EQ(config_.http_proxy().user_agent(), "");
  ASSERT_EQ(config_.http_proxy().username(), "");
  ASSERT_EQ(config_.http_proxy().domain(), "");
  ASSERT_EQ(config_.http_proxy().password(), "");
  ASSERT_EQ(config_.http_proxy().reuse_kerb(), true);
  ASSERT_EQ(config_.http_proxy().reuse_ntlm(), true);

  ASSERT_TRUE(config_.services().datagram_forwarder().enabled());
  ASSERT_TRUE(config_.services().datagram_listener().enabled());
  ASSERT_FALSE(config_.services().datagram_listener().gateway_ports());
  ASSERT_FALSE(config_.services().copy().enabled());
  ASSERT_TRUE(config_.services().socks().enabled());
  ASSERT_TRUE(config_.services().stream_forwarder().enabled());
  ASSERT_TRUE(config_.services().stream_listener().enabled());
  ASSERT_FALSE(config_.services().stream_listener().gateway_ports());
  ASSERT_FALSE(config_.services().process().enabled());

  ASSERT_GT(config_.services().process().path().length(),
            static_cast<std::size_t>(0));
  ASSERT_EQ(config_.services().process().args(), "");
}

TEST_F(LoadConfigTest, LoadProxyFileTest) {
  boost::system::error_code ec;

  config_.UpdateFromFile("./config_files/proxy.json", ec);

  ASSERT_EQ(ec.value(), 0) << "Success if complete file format";

  ASSERT_EQ(config_.http_proxy().host(), "127.0.0.1");
  ASSERT_EQ(config_.http_proxy().port(), "8080");
  ASSERT_EQ(config_.http_proxy().user_agent(), "Mozilla/5.0");
  ASSERT_EQ(config_.http_proxy().username(), "test_user");
  ASSERT_EQ(config_.http_proxy().domain(), "test_domain");
  ASSERT_EQ(config_.http_proxy().password(), "test_password");
  ASSERT_EQ(config_.http_proxy().reuse_ntlm(), false);
  ASSERT_EQ(config_.http_proxy().reuse_kerb(), false);

  ASSERT_EQ(config_.socks_proxy().version(), ssf::network::Socks::Version::kV5);
  ASSERT_EQ(config_.socks_proxy().host(), "127.0.0.2");
  ASSERT_EQ(config_.socks_proxy().port(), "1080");
}

TEST_F(LoadConfigTest, LoadServicesFileTest) {
  boost::system::error_code ec;

  config_.UpdateFromFile("./config_files/services.json", ec);

  ASSERT_EQ(ec.value(), 0) << "Success if complete file format";
  ASSERT_FALSE(config_.services().datagram_forwarder().enabled());
  ASSERT_FALSE(config_.services().datagram_listener().enabled());
  ASSERT_TRUE(config_.services().datagram_listener().gateway_ports());
  ASSERT_TRUE(config_.services().copy().enabled());
  ASSERT_FALSE(config_.services().socks().enabled());
  ASSERT_FALSE(config_.services().stream_forwarder().enabled());
  ASSERT_FALSE(config_.services().stream_listener().enabled());
  ASSERT_TRUE(config_.services().stream_listener().gateway_ports());
  ASSERT_TRUE(config_.services().process().enabled());

  ASSERT_EQ(config_.services().process().path(), "/bin/custom_path");
  ASSERT_EQ(config_.services().process().args(), "-custom args");
}

TEST_F(LoadConfigTest, LoadCircuitFileTest) {
  boost::system::error_code ec;

  config_.UpdateFromFile("./config_files/circuit.json", ec);

  ASSERT_EQ(ec.value(), 0) << "Success if complete file format";
  ASSERT_EQ(config_.circuit().nodes().size(), 3);
  auto node_it = config_.circuit().nodes().begin();
  ASSERT_EQ(node_it->addr(), "127.0.0.1");
  ASSERT_EQ(node_it->port(), "8011");
  ++node_it;
  ASSERT_EQ(node_it->addr(), "127.0.0.2");
  ASSERT_EQ(node_it->port(), "8012");
  ++node_it;
  ASSERT_EQ(node_it->addr(), "127.0.0.3");
  ASSERT_EQ(node_it->port(), "8013");
}

TEST_F(LoadConfigTest, LoadArgumentsFileTest) {
  boost::system::error_code ec;

  config_.UpdateFromFile("./config_files/arguments.json", ec);

  ASSERT_EQ(ec.value(), 0) << "Success if complete file format";
  auto argv = config_.GetArgv();

  ASSERT_EQ(argv.size(), 10);
  ASSERT_STREQ(argv[1], "-c");
  ASSERT_STREQ(argv[2], "a/path to/config_file.json");
  ASSERT_STREQ(argv[3], "-D");
  ASSERT_STREQ(argv[4], "9000");
  ASSERT_STREQ(argv[5], "-X");
  ASSERT_STREQ(argv[6], "10000");
  ASSERT_STREQ(argv[7], "-L");
  ASSERT_STREQ(argv[8], "11000:localhost:12000");
  ASSERT_EQ(argv[9], nullptr);
}

TEST_F(LoadConfigTest, LoadCompleteFileTest) {
  boost::system::error_code ec;

  config_.UpdateFromFile("./config_files/complete.json", ec);

  ASSERT_EQ(ec.value(), 0) << "Success if complete file format";

  ASSERT_FALSE(config_.tls().ca_cert().IsBuffer());
  ASSERT_EQ(config_.tls().ca_cert().value(), "test_ca_path");
  ASSERT_FALSE(config_.tls().cert().IsBuffer());
  ASSERT_EQ(config_.tls().cert().value(), "test_cert_path");
  ASSERT_FALSE(config_.tls().key().IsBuffer());
  ASSERT_EQ(config_.tls().key().value(), "test_key_path");
  ASSERT_EQ(config_.tls().key_password(), "test_key_password");
  ASSERT_FALSE(config_.tls().dh().IsBuffer());
  ASSERT_EQ(config_.tls().dh().value(), "test_dh_path");
  ASSERT_EQ(config_.tls().cipher_alg(), "test_cipher_alg");

  ASSERT_EQ(config_.http_proxy().host(), "127.0.0.1");
  ASSERT_EQ(config_.http_proxy().port(), "8080");
  ASSERT_EQ(config_.http_proxy().user_agent(), "Mozilla/5.0");
  ASSERT_EQ(config_.http_proxy().username(), "test_user");
  ASSERT_EQ(config_.http_proxy().password(), "test_password");
  ASSERT_EQ(config_.http_proxy().domain(), "test_domain");
  ASSERT_EQ(config_.http_proxy().reuse_ntlm(), false);
  ASSERT_EQ(config_.http_proxy().reuse_kerb(), false);

  ASSERT_EQ(config_.socks_proxy().version(), ssf::network::Socks::Version::kV4);
  ASSERT_EQ(config_.socks_proxy().host(), "127.0.0.2");
  ASSERT_EQ(config_.socks_proxy().port(), "1080");

  ASSERT_EQ(config_.services().process().path(), "/bin/custom_path");
  ASSERT_EQ(config_.services().process().args(), "-custom args");
}
