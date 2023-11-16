#pragma once
#if COMPILE_VULKAN

#include "Graphics/RendererTypes.hpp"
#include "Graphics/VertexBufferData.hpp"
#include "VDeleter.hpp"
#include "VulkanBuffer.hpp"
#include "VulkanRenderPass.hpp"

namespace flex
{
	enum class ImageFormat;
	class ParticleSystem;

	namespace vk
	{
		struct VulkanDevice;

		void VK_CHECK_RESULT(VkResult result);

		VkResult vkAllocateMemory(VkDevice device, const VkMemoryAllocateInfo* pAllocateInfo, const VkAllocationCallbacks* pAllocator, VkDeviceMemory* pMemory);
		void vkFreeMemory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* pAllocator);

		VkResult deviceAllocateMemory(const char* debugName, VkDevice device, const VkMemoryAllocateInfo* pAllocateInfo, const VkAllocationCallbacks* pAllocator, VkDeviceMemory* pMemory);
		void deviceFreeMemory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* pAllocator);

		struct VkAllocInfo
		{
			std::string debugName;
			VkDeviceMemory memory;
			u64 size;
		};

		void GetVertexAttributeDescriptions(VertexAttributes vertexAttributes,
			std::vector<VkVertexInputAttributeDescription>& attributeDescriptions);

		// Framebuffer for offscreen rendering
		struct FrameBufferAttachment final
		{
			struct CreateInfo
			{
				bool bIsDepth = false;
				bool bIsSampled = false;
				bool bIsCubemap = false;
				bool bIsTransferedSrc = false;
				bool bIsTransferedDst = false;
				u32 width = 0;
				u32 height = 0;
				VkFormat format = VK_FORMAT_UNDEFINED;
				VkImageLayout initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			};

			FrameBufferAttachment(VulkanDevice* device, const CreateInfo& createInfo);
			~FrameBufferAttachment();

			FrameBufferAttachment(const FrameBufferAttachment&) = delete;
			FrameBufferAttachment(FrameBufferAttachment&&) = delete;
			FrameBufferAttachment& operator=(const FrameBufferAttachment&) = delete;
			FrameBufferAttachment& operator=(FrameBufferAttachment&&) = delete;

			void CreateImage(u32 inWidth = 0, u32 inHeight = 0, const char* optDBGName = nullptr);
			void CreateImageView(const char* optDBGName = nullptr);

			void TransitionToLayout(VkImageLayout newLayout, VkQueue queue, VkCommandBuffer optCmdBuf = VK_NULL_HANDLE);

			VulkanDevice* device = nullptr;

			FrameBufferAttachmentID ID;

			// TODO: Store data in VulkanTexture
			VkImage image = VK_NULL_HANDLE;
			VkDeviceMemory mem = VK_NULL_HANDLE;
			VkImageView view = VK_NULL_HANDLE;
			u32 width = 0;
			u32 height = 0;
			bool bIsDepth = false;
			bool bIsSampled = false;
			bool bIsCubemap = false;
			bool bIsTransferedSrc = false;
			bool bIsTransferedDst = false;
			bool bOwnsResources = true;
			VkFormat format = VK_FORMAT_UNDEFINED;
			VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
		};

		struct FrameBuffer
		{
			FrameBuffer(VulkanDevice* device);
			~FrameBuffer();

			FrameBuffer(const FrameBuffer&) = delete;
			FrameBuffer(FrameBuffer&&) = delete;
			FrameBuffer& operator=(const FrameBuffer&) = delete;
			FrameBuffer& operator=(FrameBuffer&&) = delete;

			void Create(VkFramebufferCreateInfo* createInfo, VulkanRenderPass* inRenderPass, const char* debugName);

			VkFramebuffer* Replace();

			operator VkFramebuffer()
			{
				return frameBuffer;
			}

			VulkanDevice* m_VulkanDevice = nullptr;
			u32 width = 0;
			u32 height = 0;
			VkFramebuffer frameBuffer = VK_NULL_HANDLE;
			std::vector<std::pair<std::string, FrameBufferAttachment>> frameBufferAttachments;
			VulkanRenderPass* renderPass = nullptr;
		};

		struct Cascade
		{
			Cascade(VulkanDevice* device);
			~Cascade();

			Cascade(const Cascade&) = delete;
			Cascade(Cascade&&) = delete;
			Cascade& operator=(const Cascade&) = delete;
			Cascade& operator=(Cascade&&) = delete;

			FrameBuffer frameBuffer;
			FrameBufferAttachment* attachment = nullptr;
			VDeleter<VkImageView> imageView;
			VkDescriptorSet descSet;
		};

		struct VulkanQueueFamilyIndices
		{
			i32 graphicsFamily = -1;
			i32 computeFamily = -1;
			i32 presentFamily = -1;

			bool IsComplete()
			{
				return graphicsFamily >= 0 && presentFamily >= 0 && computeFamily >= 0;
			}
		};

		struct VulkanSwapChainSupportDetails
		{
			VkSurfaceCapabilitiesKHR capabilities;
			std::vector<VkSurfaceFormatKHR> formats;
			std::vector<VkPresentModeKHR> presentModes;
		};

		struct VulkanGPUBuffer final : public GPUBuffer
		{
			VulkanGPUBuffer(VulkanDevice* device, GPUBufferType type, const std::string& debugName);
			~VulkanGPUBuffer();

			VulkanGPUBuffer(const VulkanGPUBuffer&) = delete;
			VulkanGPUBuffer(VulkanGPUBuffer&& other) = delete;
			VulkanGPUBuffer& operator=(const VulkanGPUBuffer&) = delete;
			VulkanGPUBuffer& operator=(VulkanGPUBuffer&&) = delete;

			VulkanBuffer buffer;
		};

		struct VertexIndexBufferPair final
		{
			VertexIndexBufferPair(VulkanBuffer* vertexBuffer, VulkanBuffer* indexBuffer) :
				vertexBuffer(vertexBuffer),
				indexBuffer(indexBuffer)
			{}

			VertexIndexBufferPair(const VertexIndexBufferPair&) = delete;
			VertexIndexBufferPair(VertexIndexBufferPair&&) = delete;
			VertexIndexBufferPair& operator=(const VertexIndexBufferPair&) = delete;
			VertexIndexBufferPair& operator=(VertexIndexBufferPair&&) = delete;

			void Destroy();
			void Clear();

			VulkanBuffer* vertexBuffer = nullptr;
			VulkanBuffer* indexBuffer = nullptr;
			bool bUseStagingBuffer = true; // Set to false for vertex buffers that need to be updated very frequently (e.g. ImGui vertex buffer)
		};

		struct VulkanTexture final : public Texture
		{
			VulkanTexture(VulkanDevice* device, VkQueue queue);
			VulkanTexture(VulkanDevice* device, VkQueue queue, const std::string& name);

			virtual ~VulkanTexture() {}

			VulkanTexture(const VulkanTexture&) = delete;
			VulkanTexture(VulkanTexture&&) = delete;
			VulkanTexture& operator=(const VulkanTexture&) = delete;
			VulkanTexture& operator=(VulkanTexture&&) = delete;

			virtual void Reload() override;

			struct ImageCreateInfo
			{
				VkImage* image = nullptr;
				VkDeviceMemory* imageMemory = nullptr;

				bool bHDR = false;
				u32 width = 0;
				u32 height = 0;
				VkFormat format = VK_FORMAT_UNDEFINED;
				VkImageUsageFlags usage = 0;
				VkMemoryPropertyFlags properties = 0;
				u32 mipLevels = 1;

				VkImageLayout initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
				VkImageType imageType = VK_IMAGE_TYPE_2D;
				VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
				u32 arrayLayers = 1;
				VkImageCreateFlags flags = 0;

				const char* DBG_Name = nullptr;
			};

			struct ImageViewCreateInfo
			{
				VkImageView* imageView = nullptr;
				VkImage* image = nullptr;

				VkFormat format = VK_FORMAT_UNDEFINED;
				u32 mipLevels = 1;
				u32 layerCount = 1;
				VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D;
				VkImageAspectFlags aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT;

				const char* DBG_Name = nullptr;
			};

			struct SamplerCreateInfo
			{
				HTextureSampler* sampler = nullptr;

				real maxAnisotropy = 16.0f;
				real minLod = 0.0f;
				real maxLod = 1.0f;
				VkFilter magFilter = VK_FILTER_LINEAR;
				VkFilter minFilter = VK_FILTER_LINEAR;
				VkSamplerMipmapMode mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
				VkSamplerAddressMode samplerAddressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT;
				VkBorderColor borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;

				const char* DBG_Name = nullptr;
			};

			struct CubemapCreateInfo
			{
				VkImage* image = nullptr;
				VkDeviceMemory* imageMemory = nullptr;
				VkImageView* imageView = nullptr;

				VkImageLayout imageLayoutOut = VK_IMAGE_LAYOUT_UNDEFINED; // Will be set upon successful creation

				VkFormat format = VK_FORMAT_UNDEFINED;
				u32 width = 0;
				u32 height = 0;
				u32 channels = 0;
				u32 totalSize = 0;
				u32 mipLevels = 1;
				bool bGenerateMipMaps = false;
				bool bEnableTrilinearFiltering = true;

				const char* DBG_Name = nullptr;
			};

			// Static, globally usable functions

			/* Returns the size of the generated image */
			static VkDeviceSize CreateImage(VulkanDevice* device, ImageCreateInfo& createInfo);

			static void CreateImageView(VulkanDevice* device, ImageViewCreateInfo& createInfo);

			static void CreateSampler(VulkanDevice* device, SamplerCreateInfo& createInfo);

			// Expects *texture == nullptr
			static VkDeviceSize CreateCubemap(VulkanDevice* device, CubemapCreateInfo& createInfo);

			u32 CreateFromMemory(void* buffer, u32 bufferSize, u32 inWidth, u32 inHeight, u32 inChannelCount,
				TextureFormat inFormat, i32 inMipLevels, HTextureSampler inSampler, i32 layerCount = 1) override;

			void TransitionToLayout(VkImageLayout newLayout, VkCommandBuffer optCommandBuffer = VK_NULL_HANDLE);
			void CopyFromBuffer(VkBuffer buffer, u32 inWidth, u32 inHeight, VkCommandBuffer optCommandBuffer = 0);

			bool SaveToFile(VulkanDevice* device, const std::string& absoluteFilePath, ImageFormat saveFormat);

			/*
			 * Creates image, image view, and sampler based on the texture at relativeFilePath
			 * Returns true if load completed successfully
			 */
			bool LoadFromFile(const std::string& inRelativeFilePath, HTextureSampler inSampler, TextureFormat inFormat = TextureFormat::UNDEFINED) override;

			/*
			* Creates this texture's rendering resources
			* Requires data to have been loaded already
			* Returns the size of the image
			*/
			u64 Create(bool bGenerateFullMipChain = false) override;

			/*
			 * Creates image, image view, and sampler
			 * Returns the size of the image
			*/
			VkDeviceSize CreateEmpty(u32 inWidth, u32 inHeight, u32 inChannelCount, VkFormat inFormat, HTextureSampler inSampler, u32 inMipLevels = 1, VkImageUsageFlags inUsage = VK_IMAGE_USAGE_SAMPLED_BIT);

			/*
			 * Creates an empty cubemap and returns the size of the generated image
			 * Returns the size of the image
			*/
			VkDeviceSize CreateCubemapEmpty(u32 inWidth, u32 inHeight, u32 inChannelCount, VkFormat inFormat, HTextureSampler inSampler, u32 inMipLevels, bool bEnableTrilinearFiltering);

			/*
			 * Creates a cubemap from the given 6 textures
			 * Returns true if load completed successfully
			 */
			bool LoadCubemapFromTextures(const std::array<std::string, 6>& filePaths, HTextureSampler inSampler);

			/*
			 * Creates a cubemap from the given 6 textures
			 * Requires data to have been loaded already
			 * Returns the size of the image
			 */
			VkDeviceSize CreateCubemap(VkFormat inFormat, bool bEnableTrilinearFiltering);

			void GenerateMipmaps();

			VkFormat CalculateFormat();

			VDeleter<VkImage> image;
			VDeleter<VkDeviceMemory> imageMemory;
			VDeleter<VkImageView> imageView;

			// References sampler being used (owned by renderer)
			HTextureSampler sampler;

			VkImageLayout imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			VkFormat imageFormat = VK_FORMAT_UNDEFINED;

		private:
			// TODO: Make parameters to functions that need it?
			VulkanDevice* m_VulkanDevice = nullptr;
			VkQueue m_Queue = VK_NULL_HANDLE;

		};

		void SetImageLayout(
			VkCommandBuffer cmdbuffer,
			VkImage image,
			VkImageLayout oldImageLayout,
			VkImageLayout newImageLayout,
			const VkImageSubresourceRange& subresourceRange,
			VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

		void SetImageLayout(
			VkCommandBuffer cmdbuffer,
			VulkanTexture* texture,
			VkImageLayout oldImageLayout,
			VkImageLayout newImageLayout,
			const VkImageSubresourceRange& subresourceRange,
			VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

		void SetImageLayout(
			VkCommandBuffer cmdbuffer,
			VkImage image,
			VkImageAspectFlags aspectMask,
			VkImageLayout oldImageLayout,
			VkImageLayout newImageLayout,
			VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

		void SetImageLayout(
			VkCommandBuffer cmdbuffer,
			VulkanTexture* texture,
			VkImageAspectFlags aspectMask,
			VkImageLayout oldImageLayout,
			VkImageLayout newImageLayout,
			VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

		void InsertImageMemoryBarrier(
			VkCommandBuffer cmdbuffer,
			VkImage image,
			VkAccessFlags srcAccessMask,
			VkAccessFlags dstAccessMask,
			VkImageLayout oldImageLayout,
			VkImageLayout newImageLayout,
			VkPipelineStageFlags srcStageMask,
			VkPipelineStageFlags dstStageMask,
			VkImageSubresourceRange subresourceRange);

		void CreateAttachment(
			VulkanDevice* device,
			VkFormat format,
			VkImageUsageFlags usage,
			u32 width,
			u32 height,
			u32 arrayLayers,
			VkImageViewType imageViewType,
			VkImageCreateFlags imageFlags,
			VkImage* image,
			VkDeviceMemory* memory,
			VkImageView* imageView,
			const char* DBG_ImageName = nullptr,
			const char* DBG_ImageViewName = nullptr);

		void CreateAttachment(VulkanDevice* device, FrameBufferAttachment* frameBufferAttachment, const char* DBG_ImageName = nullptr, const char* DBG_ImageViewName = nullptr);

		template<class T>
		void CopyPixels(const T* srcData, T* dstData, u32 dstOffset, u32 width, u32 height, u32 channelCount, u32 pitch, bool bColourSwizzle);

		VkBool32 GetSupportedDepthFormat(VkPhysicalDevice physicalDevice, VkFormat* depthFormat);

		VkFormat FindSupportedFormat(VulkanDevice* device, const std::vector<VkFormat>& candidates, VkImageTiling tiling,
			VkFormatFeatureFlags features);
		bool HasStencilComponent(VkFormat format);
		u32 FindMemoryType(VulkanDevice* device, u32 typeFilter, VkMemoryPropertyFlags properties);
		void TransitionImageLayout(VulkanDevice* device, VkQueue queue, VkImage image, VkFormat format, VkImageLayout oldLayout,
			VkImageLayout newLayout, u32 mipLevels, VkCommandBuffer optCmdBuf = VK_NULL_HANDLE, bool bIsDepthTexture = false);

		void CopyImage(VulkanDevice* device, VkQueue queue, VkImage srcImage, VkImage dstImage, u32 width, u32 height,
			VkCommandBuffer optCmdBuf = VK_NULL_HANDLE, VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT);
		void CopyBufferToImage(VulkanDevice* device, VkQueue queue, VkBuffer buffer, VkImage image,
			u32 width, u32 height, VkCommandBuffer optCommandBuffer = VK_NULL_HANDLE);
		void CopyBuffer(VulkanDevice* device, VkQueue queue, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size,
			VkDeviceSize srcOffset = 0, VkDeviceSize dstOffset = 0);

		VkCommandBuffer BeginSingleTimeCommands(VulkanDevice* device);
		void EndSingleTimeCommands(VulkanDevice* device, VkQueue graqueuephicsQueue, VkCommandBuffer commandBuffer);

		VulkanQueueFamilyIndices FindQueueFamilies(VkSurfaceKHR surface, VkPhysicalDevice device);

		struct VulkanShader final : public Shader
		{
			VulkanShader(const VDeleter<VkDevice>& device, ShaderInfo shaderInfo);
			virtual ~VulkanShader();

			VulkanShader(const VulkanShader&) = delete;
			VulkanShader(VulkanShader&&) = delete;
			VulkanShader& operator=(const VulkanShader&) = delete;
			VulkanShader& operator=(VulkanShader&&) = delete;

			VkRenderPass renderPass = VK_NULL_HANDLE;

			VDeleter<VkShaderModule> vertShaderModule;
			VDeleter<VkShaderModule> fragShaderModule;
			VDeleter<VkShaderModule> geomShaderModule;
			VDeleter<VkShaderModule> computeShaderModule;

			VkSpecializationInfo* fragSpecializationInfo = nullptr;
		};

		struct VulkanMaterial final : public Material
		{
			VulkanMaterial() = default;
			virtual ~VulkanMaterial() = default;

			VulkanMaterial(const VulkanMaterial&) = delete;
			VulkanMaterial(VulkanMaterial&&) = delete;
			VulkanMaterial& operator=(const VulkanMaterial&) = delete;
			VulkanMaterial& operator=(VulkanMaterial&&) = delete;

			VkFramebuffer hdrCubemapFramebuffer = VK_NULL_HANDLE;

			u32 cubemapSamplerID = 0;
			u32 cubemapDepthSamplerID = 0;
		};

		struct SpecializationConstantCreateInfo
		{
			SpecializationConstantID constantID = InvalidSpecializationConstantID;
			u32 size = 0;
			void* data = nullptr;
		};

		struct GraphicsPipeline
		{
			GraphicsPipeline(const VDeleter<VkDevice>& vulkanDevice);

			VDeleter<VkPipeline> pipeline;
			VDeleter<VkPipelineLayout> layout;
			// TODO: Store pipeline cache here
		};

		struct GraphicsPipelineConfiguration
		{
			GraphicsPipelineConfiguration(GraphicsPipelineID pipelineID, GraphicsPipeline* pipeline, bool bPersistent, const char* DBG_Name) :
				pipelineID(pipelineID),
				pipeline(pipeline),
				usageCount(1),
				bPersistent(bPersistent)
			{
				strncpy(this->DBG_Name, DBG_Name, ARRAY_LENGTH(this->DBG_Name));
			}

			~GraphicsPipelineConfiguration()
			{
				delete pipeline;
			}

			GraphicsPipelineID pipelineID = InvalidGraphicsPipelineID;
			GraphicsPipeline* pipeline = nullptr;
			u32 usageCount = 0;
			bool bPersistent;

			// Debug-only
			char DBG_Name[256];
		};

		struct VulkanRenderObject final
		{
			VulkanRenderObject(RenderID renderID);

			VulkanRenderObject(const VulkanRenderObject&) = delete;
			VulkanRenderObject(VulkanRenderObject&&) = delete;
			VulkanRenderObject& operator=(const VulkanRenderObject&) = delete;
			VulkanRenderObject& operator=(VulkanRenderObject&&) = delete;

			VkPrimitiveTopology topology = VkPrimitiveTopology::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

			RenderID renderID = InvalidRenderID;
			MaterialID materialID = InvalidMaterialID;

			GameObject* gameObject = nullptr;

			VertexBufferData* vertexBufferData = nullptr;
			u32 vertexOffset = 0;

			bool bIndexed = false;
			std::vector<u32>* indices = nullptr;
			u32 indexOffset = 0;

			u32 shadowVertexOffset = 0;
			u32 shadowIndexOffset = 0;

			VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT;
			VkCompareOp depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;

			bool bEditorObject = false;
			bool bSetDynamicStates = false;
			bool bAllowDynamicBufferShrinking = true;

			u32 dynamicUBOOffset = 0;
			u32 dynamicShadowUBOOffset = 0;

			u64 dynamicVertexBufferOffset = InvalidBufferID;
			u64 dynamicIndexBufferOffset = InvalidBufferID;

			GraphicsPipelineID graphicsPipelineID = InvalidGraphicsPipelineID;

			RenderPassType renderPassOverride = RenderPassType::_NONE;

			UniformOverrides uniformOverrides;
		};

		struct GraphicsPipelineCreateInfo
		{
			u64 Hash();

			ShaderID shaderID = InvalidShaderID;
			VertexAttributes vertexAttributes = 0;

			VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
			VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT;

			VkRenderPass renderPass = VK_NULL_HANDLE;
			u32 subpass = 0;

			VkPushConstantRange* pushConstants = nullptr;
			u32 pushConstantRangeCount = 0;

			bool bSetDynamicStates = false;
			bool bEnableColourBlending = false;
			bool bEnableAdditiveColourBlending = false;
			bool bPersistent = false;

			VkBool32 depthTestEnable = VK_TRUE;
			VkBool32 depthWriteEnable = VK_TRUE;
			VkCompareOp depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
			VkBool32 stencilTestEnable = VK_FALSE;

			VkSpecializationInfo* fragSpecializationInfo = nullptr;

			const char* DBG_Name = nullptr;
		};

		struct BufferDescriptorInfo
		{
			VkBuffer buffer;
			VkDeviceSize bufferSize;
			GPUBufferType type;
		};

		struct ImageDescriptorInfo
		{
			VkImageView imageView = VK_NULL_HANDLE;
			HTextureSampler imageSampler = nullptr;
		};

		struct DescriptorSetCreateInfo
		{
			VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
			VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
			ShaderID shaderID = InvalidShaderID;
			GPUBufferList const* gpuBufferList = nullptr;

			ShaderUniformContainer<BufferDescriptorInfo> bufferDescriptors;
			ShaderUniformContainer<ImageDescriptorInfo> imageDescriptors;

			const char* DBG_Name = nullptr;
		};

		struct ImGui_PushConstBlock
		{
			glm::vec2 scale;
			glm::vec2 translate;
		};

		struct VulkanParticleSystem final
		{
			explicit VulkanParticleSystem(VulkanDevice* device);

			VulkanParticleSystem(const VulkanParticleSystem&) = delete;
			VulkanParticleSystem(VulkanParticleSystem&&) = delete;
			VulkanParticleSystem& operator=(const VulkanParticleSystem&) = delete;
			VulkanParticleSystem& operator=(VulkanParticleSystem&&) = delete;

			ParticleSystemID ID = InvalidParticleSystemID;
			VkDescriptorSet computeDescriptorSet = VK_NULL_HANDLE;
			VkDescriptorSet renderingDescriptorSet = VK_NULL_HANDLE;
			GraphicsPipelineID graphicsPipelineID = InvalidGraphicsPipelineID;
			VDeleter<VkPipeline> computePipeline;
			ParticleSystem* system = nullptr;
		};

		enum class GPUVendor : u32
		{
			UNKNOWN,
			ARM,
			AMD,
			BROADCOM,
			IMAGINATION,
			INTEL,
			NVIDIA,
			QUALCOMM,
			VERISILICON,
			SOFTWARE,
		};

		constexpr GPUVendor GPUVendorFromPCIVendor(u32 vendorID)
		{
			return vendorID == 0x13B5 ? GPUVendor::ARM
				: vendorID == 0x1002 ? GPUVendor::AMD
				: vendorID == 0x1010 ? GPUVendor::IMAGINATION
				: vendorID == 0x8086 ? GPUVendor::INTEL
				: vendorID == 0x10DE ? GPUVendor::NVIDIA
				: vendorID == 0x5143 ? GPUVendor::QUALCOMM
				: vendorID == 0x1AE0 ? GPUVendor::SOFTWARE  // Google Swiftshader
				: vendorID == 0x1414 ? GPUVendor::SOFTWARE  // Microsoft WARP
				: GPUVendor::UNKNOWN;
		}

		struct VulkanDescriptorPool final
		{
			VulkanDescriptorPool();
			VulkanDescriptorPool(VulkanDevice* device, const char* name);
			~VulkanDescriptorPool();

			VulkanDescriptorPool(const VulkanDescriptorPool& other) = delete;
			VulkanDescriptorPool(VulkanDescriptorPool&& other) = delete;
			VulkanDescriptorPool operator=(const VulkanDescriptorPool& other) = delete;
			VulkanDescriptorPool operator=(VulkanDescriptorPool&& other) = delete;

			VkDescriptorSet CreateDescriptorSet(DescriptorSetCreateInfo* createInfo);
			VkDescriptorSet CreateDescriptorSet(MaterialID materialID, const char* DBG_Name = nullptr);
			VkDescriptorSet GetSet(MaterialID materialID);
			VkDescriptorSet GetOrCreateSet(MaterialID materialID, const char* DBG_Name = nullptr);
			VkDescriptorSetLayout CreateDescriptorSetLayout(ShaderID shaderID);
			VkDescriptorSetLayout GetOrCreateLayout(ShaderID shaderID);
			void Replace();
			void Reset();
			void FreeSet(VkDescriptorSet descSet);

			void DrawImGui();

			VkDescriptorPool GetPool() const;

			// TODO: Monitor number of used desc sets to set this value intelligently
			u32 maxNumDescSets = 4096;
			static const u32 MAX_NUM_DESC_COMBINED_IMAGE_SAMPLERS = 8;
			static const u32 MAX_NUM_DESC_UNIFORM_BUFFERS = 2;
			static const u32 MAX_NUM_DESC_DYNAMIC_UNIFORM_BUFFERS = 2;
			static const u32 MAX_NUM_DESC_DYNAMIC_STORAGE_BUFFERS = 1;
			static const u32 MAX_NUM_DESC_STORAGE_BUFFERS = 4;

		private:
			std::map<ShaderID, VkDescriptorSetLayout> descriptorSetLayouts;
			std::vector<VkDescriptorSet> descriptorSets;
			u32 allocatedSetCount = 0;

			VulkanDevice* device = nullptr;
			const char* name = "";
			VkDescriptorPool pool = VK_NULL_HANDLE;
			u32 size = 0;

			std::map<VkDescriptorSetLayout, std::vector<Uniform const*>> layoutUniforms;
			std::map<VkDescriptorSetLayout, u32> layoutUsageCounts;
			std::map<VkDescriptorType, u32> descriptorTypeCounts;
		};

		VkPrimitiveTopology TopologyModeToVkPrimitiveTopology(TopologyMode mode);
		VkCullModeFlagBits CullFaceToVkCullMode(CullFace cullFace);

		TopologyMode VkPrimitiveTopologyToTopologyMode(VkPrimitiveTopology primitiveTopology);
		CullFace VkCullModeToCullFace(VkCullModeFlags cullMode);

		VkCompareOp DepthTestFuncToVkCompareOp(DepthTestFunc func);

		std::string DeviceTypeToString(VkPhysicalDeviceType type);

		VkResult CreateDebugReportCallbackEXT(VkInstance instance, const VkDebugReportCallbackCreateInfoEXT* createInfo,
			const VkAllocationCallbacks* allocator, VkDebugReportCallbackEXT* callback);
	} // namespace vk
} // namespace flex

#endif // COMPILE_VULKAN