/*************************************************************************
 * libjson-rpc-cpp
 *************************************************************************
 * @file    stubclient.cpp
 * @date    01.05.2013
 * @author  Peter Spiess-Knafl <dev@spiessknafl.at>
 * @license See attached LICENSE.txt
 ************************************************************************/

#include <iostream>

#include "gen/stubclient.h"
#include <jsonrpccpp/client/connectors/httpclient.h>

using namespace jsonrpc;
using namespace std;

int main() {
  HttpClient httpclient("http://localhost:8383");
  // StubClient c(httpclient, JSONRPC_CLIENT_V1); //json-rpc 1.0
  StubClient c(httpclient, JSONRPC_CLIENT_V2); // json-rpc 2.0

  try {
    cout << c.sayHello("Peter Knafl") << endl;
    c.notifyServer();

    cout << " 3 + 5 = " << c.addNumbers(3, 5) << endl;
    cout << " 3.2 + 5.3 = " << c.addNumbers2(3.2, 5.3) << endl;
    Json::Value args;
    args["arg1"] = 1;
    args["arg2"] = 2;
    args["operator"] = "+";
    Json::Value result = c.calculate(args);
    cout << " 1 + 2 = " << result[0].asInt() << endl;
    args["arg1"] = 3;
    args["arg2"] = 4;
    args["operator"] = "*";
    result = c.calculate(args);
    cout << " 3 * 4 = " << result[0].asInt() << endl;
    cout << "Compare: " << c.isEqual("Peter", "peter") << endl;
    cout << "Build object: " << c.buildObject("Peter", 1990) << endl;

    c.sayHello(""); // expects a server error

  } catch (JsonRpcException &e) {
    cerr << e.what() << endl;
  }
}