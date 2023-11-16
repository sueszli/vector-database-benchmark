// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "documentdb_config_builder.h"
#include <vespa/config-summary.h>
#include <vespa/config-rank-profiles.h>
#include <vespa/config-attributes.h>
#include <vespa/config-indexschema.h>
#include <vespa/document/config/documenttypes_config_fwd.h>
#include <vespa/document/repo/documenttyperepo.h>
#include <vespa/searchsummary/config/config-juniperrc.h>
#include <vespa/document/config/config-documenttypes.h>
#include <vespa/config-imported-fields.h>
#include <vespa/searchcore/proton/common/alloc_config.h>
#include <vespa/searchcore/proton/server/threading_service_config.h>

using search::TuneFileDocumentDB;
using search::index::Schema;
using vespa::config::search::RankProfilesConfig;
using vespa::config::search::IndexschemaConfig;
using vespa::config::search::AttributesConfig;
using vespa::config::search::SummaryConfig;
using vespa::config::search::summary::JuniperrcConfig;
using vespa::config::search::ImportedFieldsConfig;
using proton::ThreadingServiceConfig;

namespace proton::test {

DocumentDBConfigBuilder::DocumentDBConfigBuilder(int64_t generation,
                                                 const search::index::Schema::SP &schema,
                                                 const vespalib::string &configId,
                                                 const vespalib::string &docTypeName)
    : _generation(generation),
      _rankProfiles(std::make_shared<RankProfilesConfig>()),
      _rankingConstants(std::make_shared<search::fef::RankingConstants>()),
      _rankingExpressions(std::make_shared<search::fef::RankingExpressions>()),
      _onnxModels(std::make_shared<search::fef::OnnxModels>()),
      _indexschema(std::make_shared<IndexschemaConfig>()),
      _attributes(std::make_shared<AttributesConfig>()),
      _summary(std::make_shared<SummaryConfig>()),
      _juniperrc(std::make_shared<JuniperrcConfig>()),
      _documenttypes(std::make_shared<DocumenttypesConfig>()),
      _repo(std::make_shared<document::DocumentTypeRepo>()),
      _importedFields(std::make_shared<ImportedFieldsConfig>()),
      _tuneFileDocumentDB(std::make_shared<TuneFileDocumentDB>()),
      _schema(schema),
      _maintenance(std::make_shared<DocumentDBMaintenanceConfig>()),
      _store(),
      _threading_service_config(ThreadingServiceConfig::make()),
      _alloc_config(AllocConfig::makeDefault()),
      _configId(configId),
      _docTypeName(docTypeName)
{ }


DocumentDBConfigBuilder::DocumentDBConfigBuilder(const DocumentDBConfig &cfg)
    : _generation(cfg.getGeneration()),
      _rankProfiles(cfg.getRankProfilesConfigSP()),
      _rankingConstants(cfg.getRankingConstantsSP()),
      _rankingExpressions(cfg.getRankingExpressionsSP()),
      _onnxModels(cfg.getOnnxModelsSP()),
      _indexschema(cfg.getIndexschemaConfigSP()),
      _attributes(cfg.getAttributesConfigSP()),
      _summary(cfg.getSummaryConfigSP()),
      _juniperrc(cfg.getJuniperrcConfigSP()),
      _documenttypes(cfg.getDocumenttypesConfigSP()),
      _repo(cfg.getDocumentTypeRepoSP()),
      _importedFields(cfg.getImportedFieldsConfigSP()),
      _tuneFileDocumentDB(cfg.getTuneFileDocumentDBSP()),
      _schema(cfg.getSchemaSP()),
      _maintenance(cfg.getMaintenanceConfigSP()),
      _store(cfg.getStoreConfig()),
      _threading_service_config(cfg.get_threading_service_config()),
      _alloc_config(cfg.get_alloc_config()),
      _configId(cfg.getConfigId()),
      _docTypeName(cfg.getDocTypeName())
{}

DocumentDBConfigBuilder::~DocumentDBConfigBuilder() = default;

DocumentDBConfig::SP
DocumentDBConfigBuilder::build()
{
    return std::make_shared<DocumentDBConfig>(
            _generation,
            _rankProfiles,
            _rankingConstants,
            _rankingExpressions,
            _onnxModels,
            _indexschema,
            _attributes,
            _summary,
            _juniperrc,
            _documenttypes,
            _repo,
            _importedFields,
            _tuneFileDocumentDB,
            _schema,
            _maintenance,
            _store,
            _threading_service_config,
            _alloc_config,
            _configId,
            _docTypeName);
}

}
