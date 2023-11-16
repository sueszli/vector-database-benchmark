// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/defaults.h>
#include <vespa/log/control-file.h>
#include <vespa/log/internal.h>
#include <vespa/log/component.h>

#include <optional>
#include <cstring>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>

LOG_SETUP("vespa-logctl");


using namespace ns_log;

static void modifyLevels(const char *file, const char *component, const char *levels,
                         bool shouldCreateFile, bool shouldCreateEntry);
static void readLevels(const char *file, const char *component);


static void
usage(const char *name)
{
    fprintf(stderr, "Usage: %s [OPTION] <service>[:component-specification]\n"
            "  or:  %s [OPTION] <service>[:component-spec] <level-mods>\n"
            "Print or modify log levels for a VESPA service.\n\n"
            " -c          Create the control file if it does not exist (implies -n)\n"
            " -a          Update all .logcontrol files in <dir>\n"
            " -r          Reset to default levels\n"
            " -n          Create the component entry if it does not exist\n"
            " -f <file>   Use <file> as the log control file\n"
            " -d <dir>    Look in <dir> for log control files\n\n"
            "<level-mods> are defined as follows:\n"
            "  <level>=<on|off>[,<level>=<on|off>]...\n"
            "<level> is one of:\n"
            "  all, fatal, error, warning, info, event, config, debug or spam\n\n"
            "component-specification specifies which sub-components of the\n"
            "service should be controlled. If it is empty, all components\n"
            "are controlled:\n"
            " x.              : Matches only component x\n"
            " x               : Matches component x and all its sub-components\n\n"
            "Example: %s container:log all=on,spam=off,debug=off  : For service\n"
            "container, set log and all sub-components of log to enable all\n"
            "except spam and debug.\n\n", name, name, name);
}

static std::vector<std::string>
findAllFiles(const char *dir)
{
    std::vector<std::string> rv;
    DIR *d = opendir(dir);
    if (d == NULL) {
        perror(dir);
        return rv;
    }
    LOG(spam, "scanning %s", dir);

    struct dirent *entry;
    while ((entry = readdir(d)) != NULL) {
        if (strcmp(entry->d_name, ".")  == 0) continue;
        if (strcmp(entry->d_name, "..") == 0) continue;

        const char *suffix = ".logcontrol";

        LOG(spam, "check %s", entry->d_name);

        if (strlen(entry->d_name) > strlen(suffix)) {
            char *cmp = entry->d_name + strlen(entry->d_name) - strlen(suffix);
            if (strcmp(suffix, cmp) == 0) {
                std::string fn = dir;
                fn.append("/");
                fn.append(entry->d_name);

                struct stat sb;
                if (stat(fn.c_str(), &sb) == 0) {
                    if (S_ISREG(sb.st_mode)) {
                        *cmp = 0;
                        rv.push_back(entry->d_name);
                    }
                } else {
                    perror(fn.c_str());
                }
            }
        }
    }
    closedir(d);
    return rv;
}


int
main(int argc, char **argv)
{
    vespa::Defaults::bootstrap(argv[0]);

    const char *dir = getenv("VESPA_LOG_CONTROL_DIR");
    const char *file = getenv("VESPA_LOG_CONTROL_FILE");
    const char *root = getenv("ROOT");
    if (!root) {
        root = vespa::Defaults::vespaHome();
    }
    bool shouldCreateFile = false;
    bool shouldCreateEntry = false;
    bool doAllFiles = false;
    bool doOnlyFile = false;
    bool doResetLevels = false;

    while (1) {
        int c = getopt(argc, argv, "acnrf:d:h");
        if (c == -1)
            break;
        switch (c) {
        case 'a':
            doAllFiles = true;
            break;
        case 'r':
            doResetLevels = true;
            [[fallthrough]];
        case 'c':
            shouldCreateFile = true;
            [[fallthrough]];
        case 'n':
            shouldCreateEntry = true;
            break;
        case 'f':
            file = strdup(optarg);
            doOnlyFile = true;
            break;
        case 'd':
            dir = strdup(optarg);
            break;
        case 'h':
            usage(argv[0]);
            return EXIT_SUCCESS;
        }
    }


    char buf[PATH_MAX];
    if (!dir && !file) {
        snprintf(buf, sizeof(buf), "%s/var/db/vespa/logcontrol", root);
        dir = buf;
    }

    typedef std::vector<std::string> strlist_t;

    strlist_t services;

    char nullComponent[] = "default";
    std::string component(nullComponent);

    if (doAllFiles) {
        services = findAllFiles(dir);
        if (doOnlyFile) {
            fprintf(stderr, "-f and -a options cannot be used at the same time\n");
            return EXIT_FAILURE;
        }
        // No log control files exist
        if (services.empty()) {
            return EXIT_SUCCESS;
        }
    } else {
        if (optind >= argc) {
            usage(argv[0]);
            fprintf(stderr, "ERROR: Missing service argument!\n");
            return EXIT_FAILURE;
        }
        std::string service(argv[optind]);
        ++optind;

        auto delim_pos = service.find(':');
        if (delim_pos != std::string::npos) {
            services.push_back(service.substr(0, delim_pos));
            component = '.' + service.substr(delim_pos + 1);
        } else {
            services.push_back(service);
        }
    }

    char defLevels[] = "all=on,debug=off,spam=off";
    std::optional<std::string> levels;

    if (doResetLevels) {
        levels = defLevels;
    } else {
        if (argc > optind) {
            levels = argv[optind];
            ++optind;
        }
    }

    if (argc > optind) {
        usage(argv[0]);
        fprintf(stderr, "ERROR: Too many arguments!\n\n");
        return EXIT_FAILURE;
    }

    bool hadFailure = false;
    bool hadSuccess = false;

    for (const auto & service : services) {
        std::string serviceFile(dir);
        if (! doOnlyFile) {
            serviceFile.append("/").append(service).append(".logcontrol");
            file = serviceFile.c_str();
        }
        // fprintf(stderr, "Log control file %s:\n", file);

        try {
            if (levels.has_value()) {
                modifyLevels(file, component.c_str(), levels.value().c_str(), shouldCreateFile, shouldCreateEntry);
            } else {
                readLevels(file, component.c_str());
            }
            hadSuccess = true;
        } catch (InvalidLogException& x) {
            fprintf(stderr, "Failed: %s\n", x.what());
            hadFailure = true;
        }
    }
    if (hadFailure) return EXIT_FAILURE;
    if (! hadSuccess) {
        fprintf(stderr, "no logcontrol files updates\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

static void
modifyLevels(const char *file, const char *componentPattern, const char *levels,
             bool shouldCreateFile, bool shouldCreateEntry)
{
    ControlFile cf(file, shouldCreateFile
                   ? ControlFile::CREATE : ControlFile::READWRITE);
    Component *c;
    if (shouldCreateEntry) {
        cf.ensureComponent(componentPattern);
    }
    ComponentIterator iter(cf.getComponentIterator());
    while ((c = iter.next()) != NULL) {
        std::unique_ptr<Component> component(c);
        if (component->matches(componentPattern)) {
            component->modifyLevels(levels);
        }
    }
    cf.flush();
}

static void
readLevels(const char *file, const char *componentPattern)
{
    ControlFile cf(file, ControlFile::READONLY);
    Component *c;
    ComponentIterator iter(cf.getComponentIterator());
    while ((c = iter.next()) != NULL) {
        std::unique_ptr<Component> component(c);
        if (c->matches(componentPattern)) {
            c->display();
        }
    }
}

