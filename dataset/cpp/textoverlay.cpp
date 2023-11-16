/*
* Vulkan Example - Text overlay rendering on-top of an existing scene using a separate render pass
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanExampleBase.h"
#include "../external/stb/stb_font_consolas_24_latin1.inl"

// Vertex layout for this example
vks::model::VertexLayout vertexLayout =
{
    vks::model::VERTEX_COMPONENT_POSITION,
    vks::model::VERTEX_COMPONENT_NORMAL,
    vks::model::VERTEX_COMPONENT_UV,
    vks::model::VERTEX_COMPONENT_COLOR,
};



class VulkanExample : public vkx::ExampleBase {
public:

    struct {
        vkx::Texture background;
        vkx::Texture cube;
    } textures;

    struct {
        vks::model::Model cube;
    } meshes;

    struct {
        vks::Buffer vsScene;
    } uniformData;

    struct {
        glm::mat4 projection;
        glm::mat4 model;
        glm::vec4 lightPos = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    } uboVS;

    struct {
        vk::Pipeline solid;
        vk::Pipeline background;
    } pipelines;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSetLayout descriptorSetLayout;

    struct {
        vk::DescriptorSet background;
        vk::DescriptorSet cube;
    } descriptorSets;


    VulkanExample() {
        zoom = -4.5f;
        zoomSpeed = 2.5f;
        camera.setRotation({ -25.0f, 0.0f, 0.0f });
        title = "Vulkan Example - Text overlay";
        // Disable text overlay of the example base class
        enableTextOverlay = true;
    }

    ~VulkanExample() {
        device.destroyPipeline(pipelines.solid, nullptr);
        device.destroyPipeline(pipelines.background, nullptr);
        device.destroyPipelineLayout(pipelineLayout, nullptr);
        device.destroyDescriptorSetLayout(descriptorSetLayout, nullptr);
        meshes.cube.destroy();
        textures.background.destroy();
        textures.cube.destroy();
        uniformData.vsScene.destroy();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        cmdBuffer.bindVertexBuffers(0, meshes.cube.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.cube.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.background, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.background);
        cmdBuffer.draw(4, 1, 0, 0);

        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.solid);
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.cube, nullptr);
        cmdBuffer.drawIndexed(meshes.cube.indexCount, 1, 0, 0, 0);
    }

    // Update the text buffer displayed by the text overlay
    void updateTextOverlay(void) {
        textOverlay->beginTextUpdate();

        textOverlay->addText(title, 5.0f, 5.0f, TextOverlay::alignLeft);

        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << (frameTimer * 1000.0f) << "ms (" << lastFPS << " fps)";
        textOverlay->addText(ss.str(), 5.0f, 25.0f, TextOverlay::alignLeft);

        textOverlay->addText(context.deviceProperties.deviceName, 5.0f, 45.0f, TextOverlay::alignLeft);

        textOverlay->addText("Press \"space\" to toggle text overlay", 5.0f, size.height - 20.0f, TextOverlay::alignLeft);

        // Display projected cube vertices
        for (int32_t x = -1; x <= 1; x += 2) {
            for (int32_t y = -1; y <= 1; y += 2) {
                for (int32_t z = -1; z <= 1; z += 2) {
                    std::stringstream vpos;
                    vpos << std::showpos << x << "/" << y << "/" << z;
                    glm::vec3 projected = glm::project(glm::vec3((float)x, (float)y, (float)z), uboVS.model, uboVS.projection, glm::vec4(0, 0, (float)size.width, (float)size.height));
                    textOverlay->addText(vpos.str(), projected.x, projected.y + (y > -1 ? 5.0f : -20.0f), TextOverlay::alignCenter);
                }
            }
        }

        // Display current model view matrix
        textOverlay->addText("model view matrix", size.width, 5.0f, TextOverlay::alignRight);

        for (uint32_t i = 0; i < 4; i++) {
            ss.str("");
            ss << std::fixed << std::setprecision(2) << std::showpos;
            ss << uboVS.model[0][i] << " " << uboVS.model[1][i] << " " << uboVS.model[2][i] << " " << uboVS.model[3][i];
            textOverlay->addText(ss.str(), size.width, 25.0f + (float)i * 20.0f, TextOverlay::alignRight);
        }

        glm::vec3 projected = glm::project(glm::vec3(0.0f), uboVS.model, uboVS.projection, glm::vec4(0, 0, (float)size.width, (float)size.height));
        textOverlay->addText("Uniform cube", projected.x, projected.y, TextOverlay::alignCenter);

#if defined(__ANDROID__)
        // toto
#else
        textOverlay->addText("Hold middle mouse button and drag to move", 5.0f, size.height - 40.0f, TextOverlay::alignLeft);
#endif
        textOverlay->endTextUpdate();
    }

    void draw() override {
        prepareFrame();
        drawCurrentCommandBuffer();
        submitFrame();
    }

    void loadTextures() {
        textures.background = textureLoader->loadTexture(getAssetPath() + "textures/skysphere_bc3.ktx", vk::Format::eBc3UnormBlock);
        textures.cube = textureLoader->loadTexture(getAssetPath() + "textures/round_window_bc3.ktx", vk::Format::eBc3UnormBlock);
    }

    void loadMeshes() {
        meshes.cube = loadMesh(getAssetPath() + "models/cube.dae", vertexLayout, 1.0f);
    }

    void setupVertexDescriptions() {
        // Binding description
        vertices.bindingDescriptions.resize(1);
        vertices.bindingDescriptions[0] =
            vkx::vertexInputBindingDescription(0, vkx::vertexSize(vertexLayout), vk::VertexInputRate::eVertex);

        // Attribute descriptions
        vertices.attributeDescriptions.resize(4);
        // Location 0 : Position
        vertices.attributeDescriptions[0] =
            vkx::vertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, 0);
        // Location 1 : Normal
        vertices.attributeDescriptions[1] =
            vkx::vertexInputAttributeDescription(0, 1, vk::Format::eR32G32B32Sfloat, sizeof(float) * 3);
        // Location 2 : Texture coordinates
        vertices.attributeDescriptions[2] =
            vkx::vertexInputAttributeDescription(0, 2, vk::Format::eR32G32Sfloat, sizeof(float) * 6);
        // Location 3 : Color
        vertices.attributeDescriptions[3] =
            vkx::vertexInputAttributeDescription(0, 3, vk::Format::eR32G32B32Sfloat, sizeof(float) * 8);

        vertices.inputState = vk::PipelineVertexInputStateCreateInfo();
        vertices.inputState.vertexBindingDescriptionCount = vertices.bindingDescriptions.size();
        vertices.inputState.pVertexBindingDescriptions = vertices.bindingDescriptions.data();
        vertices.inputState.vertexAttributeDescriptionCount = vertices.attributeDescriptions.size();
        vertices.inputState.pVertexAttributeDescriptions = vertices.attributeDescriptions.data();
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes =
        {
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 2),
            vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 2),
        };

        vk::DescriptorPoolCreateInfo descriptorPoolInfo =
            vk::DescriptorPoolCreateInfo(poolSizes.size(), poolSizes.data(), 2);

        descriptorPool = device.createDescriptorPool(descriptorPoolInfo, nullptr);
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings =
        {
            // Binding 0 : Vertex shader uniform buffer
            vk::DescriptorSetLayoutBinding(
            vk::DescriptorType::eUniformBuffer,
                vk::ShaderStageFlagBits::eVertex,
                0),
            // Binding 1 : Fragment shader combined sampler
            vk::DescriptorSetLayoutBinding(
                vk::DescriptorType::eCombinedImageSampler,
                vk::ShaderStageFlagBits::eFragment,
                1),
        };

        vk::DescriptorSetLayoutCreateInfo descriptorLayout =
            vk::DescriptorSetLayoutCreateInfo(setLayoutBindings.data(), setLayoutBindings.size());

        descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayout, nullptr);

        vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
            vkx::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);

        pipelineLayout = device.createPipelineLayout(pPipelineLayoutCreateInfo, nullptr);
    }

    void setupDescriptorSet() {
        vk::DescriptorSetAllocateInfo allocInfo =
            vk::DescriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);

        // Background
        descriptorSets.background = device.allocateDescriptorSets(allocInfo)[0];

        vk::DescriptorImageInfo texDescriptor =
            vk::DescriptorImageInfo(textures.background.sampler, textures.background.view, vk::ImageLayout::eGeneral);

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets;

        // Binding 0 : Vertex shader uniform buffer
        writeDescriptorSets.push_back(
            vk::WriteDescriptorSet(descriptorSets.background, vk::DescriptorType::eUniformBuffer, 0, &uniformData.vsScene.descriptor));

        // Binding 1 : Color map 
        writeDescriptorSets.push_back(
            vk::WriteDescriptorSet(descriptorSets.background, vk::DescriptorType::eCombinedImageSampler, 1, &texDescriptor));

        device.updateDescriptorSets(writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);

        // Cube
        descriptorSets.cube = device.allocateDescriptorSets(allocInfo)[0];
        texDescriptor.sampler = textures.cube.sampler;
        texDescriptor.imageView = textures.cube.view;
        writeDescriptorSets[0].dstSet = descriptorSets.cube;
        writeDescriptorSets[1].dstSet = descriptorSets.cube;
        device.updateDescriptorSets(writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);
    }

    void preparePipelines() {
        vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState =
            vkx::pipelineInputAssemblyStateCreateInfo(vk::PrimitiveTopology::eTriangleList, vk::PipelineInputAssemblyStateCreateFlags(), VK_FALSE);

        vk::PipelineRasterizationStateCreateInfo rasterizationState =
            vkx::pipelineRasterizationStateCreateInfo(vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack, vk::FrontFace::eClockwise);

        vk::PipelineColorBlendAttachmentState blendAttachmentState =
            vkx::pipelineColorBlendAttachmentState();

        vk::PipelineColorBlendStateCreateInfo colorBlendState =
            vkx::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);

        vk::PipelineDepthStencilStateCreateInfo depthStencilState =
            vkx::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, vk::CompareOp::eLessOrEqual);

        vk::PipelineViewportStateCreateInfo viewportState =
            vkx::pipelineViewportStateCreateInfo(1, 1);

        vk::PipelineMultisampleStateCreateInfo multisampleState =
            vkx::pipelineMultisampleStateCreateInfo(vk::SampleCountFlagBits::e1);

        std::vector<vk::DynamicState> dynamicStateEnables = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };
        vk::PipelineDynamicStateCreateInfo dynamicState =
            vkx::pipelineDynamicStateCreateInfo(dynamicStateEnables.data(), dynamicStateEnables.size());

        // Wire frame rendering pipeline
        std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

        shaderStages[0] = context.loadShader(getAssetPath() + "shaders/textoverlay/mesh.vert.spv", vk::ShaderStageFlagBits::eVertex);
        shaderStages[1] = context.loadShader(getAssetPath() + "shaders/textoverlay/mesh.frag.spv", vk::ShaderStageFlagBits::eFragment);

        vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
            vkx::pipelineCreateInfo(pipelineLayout, renderPass);

        pipelineCreateInfo.pVertexInputState = &vertices.inputState;
        pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
        pipelineCreateInfo.pRasterizationState = &rasterizationState;
        pipelineCreateInfo.pColorBlendState = &colorBlendState;
        pipelineCreateInfo.pMultisampleState = &multisampleState;
        pipelineCreateInfo.pViewportState = &viewportState;
        pipelineCreateInfo.pDepthStencilState = &depthStencilState;
        pipelineCreateInfo.pDynamicState = &dynamicState;
        pipelineCreateInfo.stageCount = shaderStages.size();
        pipelineCreateInfo.pStages = shaderStages.data();

        pipelines.solid = device.createGraphicsPipelines(context.pipelineCache, pipelineCreateInfo, nullptr)[0];

        // Background rendering pipeline
        depthStencilState.depthTestEnable = VK_FALSE;
        depthStencilState.depthWriteEnable = VK_FALSE;

        rasterizationState.polygonMode = vk::PolygonMode::eFill;

        shaderStages[0] = context.loadShader(getAssetPath() + "shaders/textoverlay/background.vert.spv", vk::ShaderStageFlagBits::eVertex);
        shaderStages[1] = context.loadShader(getAssetPath() + "shaders/textoverlay/background.frag.spv", vk::ShaderStageFlagBits::eFragment);

        pipelines.background = device.createGraphicsPipelines(context.pipelineCache, pipelineCreateInfo, nullptr)[0];
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Vertex shader uniform buffer block
        uniformData.vsScene= context.createUniformBuffer(uboVS);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        // Vertex shader
        uboVS.projection = camera.matrices.perspective;
        uboVS.model = camera.matrices.view;
        uniformData.vsScene.copy(uboVS);
    }

    void prepare() override {
        ExampleBase::prepare();
        loadTextures();
        loadMeshes();
        setupVertexDescriptions();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        prepared = true;
    }

    virtual void render() override {
        if (!prepared)
            return;
        draw();

        if (frameCounter == 0) {
            updateTextOverlay();
        }
    }

    void viewChanged() override {
        updateUniformBuffers();
        updateTextOverlay();
    }

    void windowResized() override {
        updateTextOverlay();
    }

    void keyPressed(uint32_t keyCode) override {
        switch (keyCode) {
        case GLFW_KEY_KP_ADD:
        case GLFW_KEY_SPACE:
            textOverlay->visible = !textOverlay->visible;
        }
    }
};

RUN_EXAMPLE(VulkanExample)
