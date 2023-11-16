import sys, os
from time import gmtime, strftime

_H2O_IP_                      = "127.0.0.1"
_H2O_PORT_                    = 54321
_H2O_EXTRA_CONNECT_ARGS_      = dict()
_ON_HADOOP_                   = False
_HADOOP_NAMENODE_             = None
_IS_IPYNB_                    = False
_IS_PYDEMO_                   = False
_IS_PYUNIT_                   = False
_IS_PYBOOKLET_                = False
_RESULTS_DIR_                 = False
_TEST_NAME_                   = ""
_FORCE_CONNECT_               = False
_LDAP_USER_NAME_              = None
_LDAP_PASSWORD_               = None
_KERB_PRINCIPAL_              = None

def parse_args(args):
    global _H2O_IP_
    global _H2O_PORT_
    global _H2O_EXTRA_CONNECT_ARGS_
    global _ON_HADOOP_
    global _HADOOP_NAMENODE_
    global _IS_IPYNB_
    global _IS_PYDEMO_
    global _IS_PYUNIT_
    global _IS_PYBOOKLET_
    global _RESULTS_DIR_
    global _TEST_NAME_
    global _FORCE_CONNECT_
    global _LDAP_USER_NAME_
    global _LDAP_PASSWORD_
    global _KERB_PRINCIPAL_

    i = 1
    while (i < len(args)):
        s = args[i]
        if ( s == "--usecloud" or s == "--uc" ):
            i = i + 1
            if (i > len(args)): usage()
            param = args[i]
            if param.lower().startswith("https://"):
                _H2O_EXTRA_CONNECT_ARGS_ = {'https': True, 'verify_ssl_certificates': False}
                param = param[8:]
            argsplit = param.split(":")
            _H2O_IP_   = argsplit[0]
            _H2O_PORT_ = int(argsplit[1])
        elif (s == "--hadoopNamenode"):
            i = i + 1
            if (i > len(args)): usage()
            _HADOOP_NAMENODE_ = args[i]
        elif (s == "--onHadoop"):
            _ON_HADOOP_ = True
        elif (s == "--ipynb"):
            _IS_IPYNB_ = True
        elif (s == "--pyDemo"):
            _IS_PYDEMO_ = True
        elif (s == "--pyUnit"):
            _IS_PYUNIT_ = True
        elif (s == "--pyBooklet"):
            _IS_PYBOOKLET_ = True
        elif (s == "--resultsDir"):
            i = i + 1
            if (i > len(args)): usage()
            _RESULTS_DIR_ = args[i]
        elif (s == "--testName"):
            i = i + 1
            if (i > len(args)): usage()
            _TEST_NAME_ = args[i]
        elif (s == "--ldapUsername"):
            i = i + 1
            if (i > len(args)): usage()
            _LDAP_USER_NAME_ = args[i]
        elif (s == "--ldapPassword"):
            i = i + 1
            if (i > len(args)): usage()
            _LDAP_PASSWORD_ = args[i]
        elif (s == "--kerbPrincipal"):
            i = i + 1
            if (i > len(args)): usage()
            _KERB_PRINCIPAL_ = args[i]
        elif (s == "--forceConnect"):
            _FORCE_CONNECT_ = True
        else:
            unknownArg(s)
        i = i + 1

def usage():
    print("")
    print("Usage for:  python pyunit.py [...options...]")
    print("")
    print("    --usecloud        connect to h2o on specified ip and port, where ip and port are specified as follows:")
    print("                      IP:PORT")
    print("")
    print("    --onHadoop        Indication that tests will be run on h2o multinode hadoop clusters.")
    print("                      `locate` and `sandbox` pyunit test utilities use this indication in order to")
    print("                      behave properly. --hadoopNamenode must be specified if --onHadoop option is used.")
    print("    --hadoopNamenode  Specifies that the pyunit tests have access to this hadoop namenode.")
    print("                      `hadoop_namenode` pyunit test utility returns this value.")
    print("")
    print("    --ipynb           test is ipython notebook")
    print("")
    print("    --pyDemo          test is python demo")
    print("")
    print("    --pyUnit          test is python unit test")
    print("")
    print("    --pyBooklet       test is python booklet")
    print("")
    print("    --resultsDir      the results directory.")
    print("")
    print("    --testName        name of the pydemo, pyunit, or pybooklet.")
    print("")
    print("    --ldapUsername    LDAP username.")
    print("")
    print("    --ldapPassword    LDAP password.")
    print("")
    print("    --kerbPrincipal   Kerberos service principal.")
    print("")
    print("    --forceConnect    h2o will attempt to connect to cluster regardless of cluster's health.")
    print("")
    sys.exit(1) #exit with nonzero exit code

def unknownArg(arg):
    print("")
    print("ERROR: Unknown argument: " + arg)
    print("")
    usage()


def h2o_test_setup(sys_args):
    h2o_py_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)),".."))
    h2o_docs_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","..","h2o-docs"))

    parse_args(sys_args)

    sys.path.insert(1, h2o_py_dir)
    import h2o
    from tests import pyunit_utils, pydemo_utils, pybooklet_utils

    for pkg in (pyunit_utils, pybooklet_utils):
        setattr(pkg, '__on_hadoop__', _ON_HADOOP_)
        setattr(pkg, '__hadoop_namenode__', _HADOOP_NAMENODE_)
        setattr(pkg, '__test_name__', _TEST_NAME_)
        setattr(pkg, '__results_dir__', _RESULTS_DIR_)

    if _IS_PYUNIT_ or _IS_IPYNB_ or _IS_PYBOOKLET_ or _IS_PYDEMO_:
        pass
    else:
        raise(EnvironmentError, "Unrecognized test type. Must be of type ipynb, pydemo, pyunit, or pybooklet, but got: "
                                "{0}".format(_TEST_NAME_))

    print("[{0}] {1}\n".format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), "Connect to h2o on IP: {0} PORT: {1}".format(_H2O_IP_, _H2O_PORT_)))
    auth = None
    if _LDAP_USER_NAME_ is not None and _LDAP_PASSWORD_ is not None:
        print("Using basic auth with %s user name" % _LDAP_USER_NAME_)
        auth = (_LDAP_USER_NAME_, _LDAP_PASSWORD_)
    elif _KERB_PRINCIPAL_ is not None:
        print("Using SPNEGO auth with %s principal" % _KERB_PRINCIPAL_)
        from h2o.auth import SpnegoAuth
        auth = SpnegoAuth(service_principal=_KERB_PRINCIPAL_)
    else:
        print("Not using any auth")
    h2o.connect(ip=_H2O_IP_, port=_H2O_PORT_, verbose=False, auth=auth, **_H2O_EXTRA_CONNECT_ARGS_)
    h2o.utils.config.H2OConfigReader.get_config()["general.allow_breaking_changes"] = True

    #rest_log = os.path.join(_RESULTS_DIR_, "rest.log")
    #h2o.start_logging(rest_log)
    #print "[{0}] {1}\n".format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), "Started rest logging in: {0}".format(rest_log))

    h2o.log_and_echo("------------------------------------------------------------")
    h2o.log_and_echo("")
    h2o.log_and_echo("STARTING TEST: " + _TEST_NAME_)
    h2o.log_and_echo("")
    h2o.log_and_echo("------------------------------------------------------------")

    h2o.remove_all()

    if _IS_IPYNB_:       pydemo_utils.ipy_notebook_exec(_TEST_NAME_)
    elif _IS_PYUNIT_:    pyunit_utils.pyunit_exec(_TEST_NAME_)
    elif _IS_PYBOOKLET_: pybooklet_utils.pybooklet_exec(_TEST_NAME_)
    elif _IS_PYDEMO_:    pydemo_utils.pydemo_exec(_TEST_NAME_)

if __name__ == "__main__":
    h2o_test_setup(sys.argv)
