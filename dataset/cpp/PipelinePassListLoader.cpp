// Copyright (C) 2023 Jérôme "Lynix" Leclercq (lynix680@gmail.com)
// This file is part of the "Nazara Engine - Graphics module"
// For conditions of distribution and use, see copyright notice in Config.hpp

#include <Nazara/Graphics/Formats/PipelinePassListLoader.hpp>
#include <Nazara/Core/ParameterFile.hpp>
#include <Nazara/Graphics/Graphics.hpp>
#include <optional>
#include <Nazara/Graphics/Debug.hpp>

namespace Nz::Loaders
{
	namespace NAZARA_ANONYMOUS_NAMESPACE
	{
		class PassListLoader
		{
			public:
				PassListLoader(Stream& stream, const PipelinePassListParams& /*parameters*/) :
				m_paramFile(stream)
				{
				}

				Result<std::shared_ptr<PipelinePassList>, ResourceLoadingError> Parse()
				{
					try
					{
						std::string finalOutputAttachment;

						m_paramFile.Handle(
							"passlist", [&](std::string passListName)
							{
								m_current.emplace();
								m_current->passList = std::make_shared<PipelinePassList>();
								m_paramFile.Block(
									"attachment", [this](std::string attachmentName)
									{
										HandleAttachment(std::move(attachmentName));
									},
									"attachmentproxy", [this](std::string proxyName, std::string targetName)
									{
										HandleAttachmentProxy(std::move(proxyName), std::move(targetName));
									},
									"pass", [this](std::string passName)
									{
										HandlePass(std::move(passName));
									},
									"output", &finalOutputAttachment
								);
							}
						);

						if (finalOutputAttachment.empty())
						{
							NazaraError("missing passlist output attachment");
							throw ResourceLoadingError::DecodingError;
						}

						auto it = m_current->attachmentsByName.find(finalOutputAttachment);
						if (it == m_current->attachmentsByName.end())
						{
							NazaraErrorFmt("unknown attachment {}", finalOutputAttachment);
							throw ResourceLoadingError::DecodingError;
						}

						m_current->passList->SetFinalOutput(it->second);

						return Ok(std::move(m_current->passList));
					}
					catch (ResourceLoadingError error)
					{
						return Err(error);
					}
				}

			private:
				void HandleAttachment(std::string attachmentName)
				{
					std::string format;

					m_paramFile.Block(
						"format", &format
					);

					if (format.empty())
					{
						NazaraErrorFmt("missing mandatory format in attachment {}", attachmentName);
						throw ResourceLoadingError::DecodingError;
					}

					PixelFormat attachmentFormat = PixelFormat::Undefined;
					if (format == "PreferredDepth")
						attachmentFormat = Graphics::Instance()->GetPreferredDepthFormat();
					else if (format == "PreferredDepthStencil")
						attachmentFormat = Graphics::Instance()->GetPreferredDepthStencilFormat();
					else
						attachmentFormat = PixelFormatInfo::IdentifyFormat(format);

					if (attachmentFormat == PixelFormat::Undefined)
					{
						NazaraErrorFmt("unknown format {}", format);
						throw ResourceLoadingError::DecodingError;
					}

					if (m_current->attachmentsByName.find(attachmentName) != m_current->attachmentsByName.end())
					{
						NazaraErrorFmt("attachment {} already exists", attachmentName);
						throw ResourceLoadingError::DecodingError;
					}

					std::size_t attachmentId = m_current->passList->AddAttachment({
						attachmentName,
						attachmentFormat
					});

					m_current->attachmentsByName.emplace(attachmentName, attachmentId);
				}

				void HandleAttachmentProxy(std::string proxyName, std::string targetName)
				{
					auto it = m_current->attachmentsByName.find(targetName);
					if (it == m_current->attachmentsByName.end())
					{
						NazaraErrorFmt("unknown attachment {}", targetName);
						throw ResourceLoadingError::DecodingError;
					}

					if (m_current->attachmentsByName.find(proxyName) != m_current->attachmentsByName.end())
					{
						NazaraErrorFmt("attachment {} already exists", proxyName);
						throw ResourceLoadingError::DecodingError;
					}

					std::size_t proxyId = m_current->passList->AddAttachmentProxy(proxyName, it->second);
					m_current->attachmentsByName.emplace(std::move(proxyName), proxyId);
				}
				
				void HandlePass(std::string passName)
				{
					struct InputOutput
					{
						std::string name;
						std::string attachmentName;
					};

					ParameterList implConfig;
					std::string impl;
					std::string depthstencilInput;
					std::string depthstencilOutput;
					std::vector<InputOutput> inputs;
					std::vector<InputOutput> outputs;
					std::vector<std::string> flags;

					m_paramFile.Block(
						"depthstencilinput", &depthstencilInput,
						"depthstenciloutput", &depthstencilOutput,
						"impl", [&](std::string passImpl)
						{
							impl = std::move(passImpl);
							m_paramFile.Block(ParameterFile::OptionalBlock,
							ParameterFile::Array, [&](ParameterFile::Keyword key, std::string value)
							{
								implConfig.SetParameter(std::move(key.str), std::move(value));
							});
						},
						"input", [&](std::string name, std::string attachment)
						{
							inputs.push_back({
								std::move(name),
								std::move(attachment),
							});
						},
						"output", [&](std::string name, std::string attachment)
						{
							outputs.push_back({
								std::move(name),
								std::move(attachment),
							});
						},
						"flag", [&](std::string flag)
						{
							flags.push_back(std::move(flag));
						}
					);

					FramePipelinePassRegistry& passRegistry = Graphics::Instance()->GetFramePipelinePassRegistry();

					std::size_t implIndex = passRegistry.GetPassIndex(impl);
					if (implIndex == FramePipelinePassRegistry::InvalidIndex)
					{
						NazaraErrorFmt("unknown pass {}", impl);
						throw ResourceLoadingError::DecodingError;
					}

					std::size_t passId = m_current->passList->AddPass(passName, implIndex, std::move(implConfig));

					for (auto&& [inputName, attachmentName] : inputs)
					{
						std::size_t inputIndex = passRegistry.GetPassInputIndex(implIndex, inputName);
						if (inputIndex == FramePipelinePassRegistry::InvalidIndex)
						{
							NazaraErrorFmt("pass {} has no input {}", impl, inputName);
							throw ResourceLoadingError::DecodingError;
						}

						auto it = m_current->attachmentsByName.find(attachmentName);
						if (it == m_current->attachmentsByName.end())
						{
							NazaraErrorFmt("unknown attachment {}", attachmentName);
							throw ResourceLoadingError::DecodingError;
						}

						m_current->passList->SetPassInput(passId, inputIndex, it->second);
					}

					for (auto&& [outputName, attachmentName] : outputs)
					{
						std::size_t inputIndex = passRegistry.GetPassOutputIndex(implIndex, outputName);
						if (inputIndex == FramePipelinePassRegistry::InvalidIndex)
						{
							NazaraErrorFmt("pass {} has no output {}", impl, outputName);
							throw ResourceLoadingError::DecodingError;
						}

						auto it = m_current->attachmentsByName.find(attachmentName);
						if (it == m_current->attachmentsByName.end())
						{
							NazaraErrorFmt("unknown attachment {}", attachmentName);
							throw ResourceLoadingError::DecodingError;
						}

						m_current->passList->SetPassOutput(passId, inputIndex, it->second);
					}

					if (!depthstencilInput.empty())
					{
						auto it = m_current->attachmentsByName.find(depthstencilInput);
						if (it == m_current->attachmentsByName.end())
						{
							NazaraErrorFmt("unknown attachment {}", depthstencilInput);
							throw ResourceLoadingError::DecodingError;
						}

						m_current->passList->SetPassDepthStencilInput(passId, it->second);
					}

					if (!depthstencilOutput.empty())
					{
						auto it = m_current->attachmentsByName.find(depthstencilOutput);
						if (it == m_current->attachmentsByName.end())
						{
							NazaraErrorFmt("unknown attachment {}", depthstencilOutput);
							throw ResourceLoadingError::DecodingError;
						}

						m_current->passList->SetPassDepthStencilOutput(passId, it->second);
					}

					for (const auto& flagStr : flags)
					{
						if (flagStr == "LightShadowing")
							m_current->passList->EnablePassFlags(passId, FramePipelinePassFlag::LightShadowing);
						else
						{
							NazaraErrorFmt("unknown pass flag {}", flagStr);
							throw ResourceLoadingError::DecodingError;
						}
					}
				}

				struct CurrentPassList
				{
					std::shared_ptr<PipelinePassList> passList;
					std::unordered_map<std::string /*attachmentName*/, std::size_t /*attachmentId*/> attachmentsByName;
				};

				std::optional<CurrentPassList> m_current;
				std::string m_currentLine;
				ParameterFile m_paramFile;
		};
	}

	PipelinePassListLoader::Entry GetPipelinePassListLoader()
	{
		NAZARA_USE_ANONYMOUS_NAMESPACE

		PipelinePassListLoader::Entry entry;
		entry.extensionSupport = [](std::string_view ext) { return ext == ".passlist"; };
		entry.streamLoader = [](Stream& stream, const PipelinePassListParams& parameters)
		{
			PassListLoader passListLoader(stream, parameters);
			return passListLoader.Parse();
		};

		return entry;
	}
}
