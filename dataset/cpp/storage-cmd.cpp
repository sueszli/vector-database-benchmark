// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#include <vespa/fnet/frt/rpcrequest.h>
#include <vespa/fnet/frt/supervisor.h>
#include <vespa/fnet/frt/target.h>
#include <vespa/slobrok/sbmirror.h>
#include <vespa/vespalib/util/signalhandler.h>
#include <vespa/vespalib/locale/c.h>
#include <vespa/vespalib/util/time.h>
#include <thread>

#include <vespa/log/log.h>
LOG_SETUP("vespa-storage-cmd");

class RPCClient
{
private:
    static bool addArg(FRT_RPCRequest *req, const char *param) {
        int len = strlen(param);
        if (len < 2 || param[1] != ':') {
            return false;
        }
        const char *value = param + 2;
        switch (param[0]) {
        case 'b':
            req->GetParams()->AddInt8(strtoll(value, nullptr, 0));
            break;
        case 'h':
            req->GetParams()->AddInt16(strtoll(value, nullptr, 0));
            break;
        case 'i':
            req->GetParams()->AddInt32(strtoll(value, nullptr, 0));
            break;
        case 'l':
            req->GetParams()->AddInt64(strtoll(value, nullptr, 0));
            break;
        case 'f':
            req->GetParams()->AddFloat(vespalib::locale::c::strtod(value, nullptr));
            break;
        case 'd':
            req->GetParams()->AddDouble(vespalib::locale::c::strtod(value, nullptr));
            break;
        case 's':
            req->GetParams()->AddString(value);
            break;
        default:
            return false;
        }
        return true;
    }

public:
    int main(int argc, char **argv) {
        if (argc < 3) {
            fprintf(stderr, "usage: vespa-storage-cmd <connectspec> <method> [args]\n");
            fprintf(stderr, "Calls RPC method on a storage/distributor process\n");
            fprintf(stderr, "Call frt.rpc.getMethodList to get available RPC methods\n");
            fprintf(stderr, "    each arg must be on the form <type>:<value>\n");
            fprintf(stderr, "    supported types: {'b','h','i','l','f','d','s'}\n");
            return 1;
        }
        int retCode = 0;
        fnet::frt::StandaloneFRT supervisor;

        slobrok::ConfiguratorFactory sbcfg(config::ConfigUri("client"));
        slobrok::api::MirrorAPI mirror(supervisor.supervisor(), sbcfg);

        while (!mirror.ready()) {
            std::this_thread::sleep_for(10ms);
        }

        slobrok::api::MirrorAPI::SpecList list = mirror.lookup(argv[1]);

        if (list.size() == 0) {
            fprintf(stderr, "No servers found matching %s\n", argv[1]);
        }

        for (size_t j = 0; j < list.size(); j++) {
            FRT_Target *target = supervisor.supervisor().GetTarget(list[j].second.c_str());

            // If not fleet controller, need to connect first.
            if (strstr(argv[1], "fleetcontroller") == nullptr) {
                FRT_RPCRequest *req = supervisor.supervisor().AllocRPCRequest();
                req->SetMethodName("vespa.storage.connect");
                req->GetParams()->AddString(argv[1]);
                target->InvokeSync(req, 10.0);

                if (req->GetErrorCode() != FRTE_NO_ERROR) {
                    fprintf(stderr, "error(%d): %s\n",
                        req->GetErrorCode(),
                        req->GetErrorMessage());
                    continue;
                }
                req->internal_subref();
            }

            FRT_RPCRequest *req = supervisor.supervisor().AllocRPCRequest();
            req->SetMethodName(argv[2]);

            for (int i = 3; i < argc; ++i) {
                if (!addArg(req, argv[i])) {
                    fprintf(stderr, "could not parse parameter: '%s'\n", argv[i]);
                    retCode = 2;
                    break;
                }
            }
            if (retCode == 0) {
                target->InvokeSync(req, 10.0);
                if (req->GetErrorCode() == FRTE_NO_ERROR) {
                    fprintf(stdout, "RETURN VALUES FOR %s:\n", list[j].first.c_str());
                    req->GetReturn()->Print();
                    retCode = 3;
                } else {
                    fprintf(stderr, "error(%d): %s\n",
                        req->GetErrorCode(),
                        req->GetErrorMessage());
                }
            }
            req->internal_subref();
            target->internal_subref();
        }
        return retCode;
    }
};

int main(int argc, char **argv) {
    vespalib::SignalHandler::PIPE.ignore();
    RPCClient myapp;
    return myapp.main(argc, argv);
}
