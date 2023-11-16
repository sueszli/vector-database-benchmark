/* Copyright 2013-2023 Bas van den Berg
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "CGenerator/CGenerator.h"
#include "CGenerator/CCodeGenerator.h"
#include "CGenerator/MakefileGenerator.h"
#include "AST/AST.h"
#include "AST/Decl.h"
#include "AST/Component.h"
#include "FileUtils/FileUtils.h"
#include "Utils/StringBuilder.h"
#include "Utils/ProcessUtils.h"
#include "Utils/color.h"
#include "Utils/GenUtils.h"

#include <stdio.h>

using namespace C2;

static const char* logfile = "build.log";

CGenerator::CGenerator(const Component& component_,
                       const Modules& moduleMap_,
                       const HeaderNamer& namer_,
                       const Options& options_,
                       const TargetInfo& targetInfo_,
                       const BuildFile* buildFile_)
    : component(component_)
    , moduleMap(moduleMap_)
    , includeNamer(namer_)
    , options(options_)
    , targetInfo(targetInfo_)
    , buildFile(buildFile_)
{}

void CGenerator::addSourceLib(const Component& lib) {
    const ModuleList& mods = lib.getModules();

    for (unsigned m=0; m<mods.size(); m++) {
        code_mods.push_back(mods[m]);
    }
}

void CGenerator::generate() {
    std::string outdir = options.outputDir + options.buildDir;
    const ModuleList& mods = component.getModules();

    for (unsigned m=0; m<mods.size(); m++) {
        code_mods.push_back(mods[m]);
    }

    // generate code
    if (options.single_module) {
        CCodeGenerator gen(component.getName(), CCodeGenerator::SINGLE_FILE, moduleMap, code_mods, includeNamer, targetInfo, options.generateChecks, options.generateAsserts);
        gen.generate(options.printC, outdir);
    } else {
        for (unsigned m=0; m<code_mods.size(); m++) {
            Module* M = code_mods[m];
            ModuleList single;
            single.push_back(M);
            CCodeGenerator gen(M->getName(), CCodeGenerator::MULTI_FILE, moduleMap, single, includeNamer, targetInfo, options.generateChecks, options.generateAsserts);
            gen.generate(options.printC, outdir);
        }
    }

    // generate Makefile
    MakefileGenerator makeGen(component, options.single_module, options.generateAsserts, options.fast, targetInfo, buildFile);
    makeGen.write(outdir);

    // generate exports.version
    if (component.isSharedLib()) {
        StringBuilder expmap;
        expmap << "LIB_1.0 {\n";
        expmap << "\tglobal:\n";
        for (unsigned m=0; m<mods.size(); m++) {
            const Module* M = mods[m];
            if (!M->isLoaded()) continue;
            const Module::Symbols& syms = M->getSymbols();
            for (Module::SymbolsConstIter iter = syms.begin(); iter != syms.end(); ++iter) {
                const Decl* D = iter->second;
                if (!D->isExported()) continue;
                if (!isa<FunctionDecl>(D) && !isa<VarDecl>(D)) continue;
                expmap << "\t\t";
                GenUtils::addName(M->getName(), iter->first, expmap);
                expmap << ";\n";
            }
        }
        expmap << "\tlocal:\n\t\t*;\n";
        expmap << "};\n";
        std::string outfile = outdir + "exports.version";
        FileUtils::writeFile(outdir.c_str(), outfile.c_str(), expmap);
    }
}

void CGenerator::build() {
    std::string outdir = options.outputDir + options.buildDir;
    // execute generated makefile
    int retval = ProcessUtils::run(outdir, "/usr/bin/make", logfile);
    if (retval != 0) {
        fprintf(stderr, ANSI_RED"error during external c compilation" ANSI_NORMAL"\n");
        fprintf(stderr, "see %s%s for details\n", outdir.c_str(), logfile);
    }
}

void CGenerator::generateExternalHeaders() {
    for (ModulesConstIter iter = moduleMap.begin(); iter != moduleMap.end(); ++iter) {
        Module* M = iter->second;
        if (!M->isLoaded()) continue;
        if (!M->isInterface()) continue;
        if (options.single_module && M->isPlugin()) {
            // TEMP here until refactor C2 as plugin component
            if (strcmp(M->getName().c_str(),  "c2") != 0) {
                continue;
            }
        }
        ModuleList single;
        single.push_back(M);
        CCodeGenerator gen(M->getName(), CCodeGenerator::MULTI_FILE, moduleMap, single, includeNamer, targetInfo, false, false);
        gen.createLibHeader(options.printC, options.outputDir + options.buildDir);
    }
}

void CGenerator::generateInterfaceFiles() {
    const ModuleList& mods = component.getModules();
    for (unsigned m=0; m<mods.size(); m++) {
        Module* M = mods[m];
        if (!M->isLoaded()) continue;
        if (!M->isExported()) continue;
        ModuleList single;
        single.push_back(M);
        CCodeGenerator gen(M->getName(), CCodeGenerator::MULTI_FILE, moduleMap, single, includeNamer, targetInfo, false, false);
        gen.createLibHeader(options.printC, options.outputDir);
    }
}

