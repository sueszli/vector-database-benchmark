/*
* Vulkan Example - Animated gears using multiple uniform buffers
*
* See readme.md for details
*
* Copyright (C) 2015 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include "vks/offscreen.hpp"
#include "vks/pipelines.hpp"
#include "shapes.h"
#include "easings.hpp"
#include "utils.hpp"

namespace vkx {
class ShapesRenderer : public OffscreenRenderer {
    using Parent = vkx::OffscreenRenderer;

public:
    static const uint32_t SHAPES_COUNT{ 5 };
    static const uint32_t INSTANCES_PER_SHAPE{ 4000 };
    static const uint32_t INSTANCE_COUNT{ (INSTANCES_PER_SHAPE * SHAPES_COUNT) };

    const bool stereo;
    vks::Buffer meshes;

    // Per-instance data block
    struct InstanceData {
        glm::vec3 pos;
        glm::vec3 rot;
        float scale;
    };

    struct ShapeVertexData {
        size_t baseVertex;
        size_t vertices;
    };

    struct Vertex {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec3 color;
    };

    // Contains the instanced data
    vks::Buffer instanceBuffer;
    // Contains the instanced data
    vks::Buffer indirectBuffer;

    struct UboVS {
        glm::mat4 projection;
        glm::mat4 view;
        float time = 0.0f;
    } uboVS;

    struct {
        vks::Buffer vsScene;
    } uniformData;

    struct {
        vk::Pipeline solid;
    } pipelines;

    std::vector<ShapeVertexData> shapes;
    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::CommandBuffer cmdBuffer;
    const vk::DescriptorType uniformType{ stereo ? vk::DescriptorType::eUniformBufferDynamic : vk::DescriptorType::eUniformBuffer };
    const float duration = 4.0f;
    const float interval = 6.0f;

    std::array<glm::mat4, 2> eyePoses;
#if 0
        float zoom{ -1.0f };
        float zoomDelta{ 135 };
        float zoomStart{ 0 };
        float accumulator{ FLT_MAX };
        float frameTimer{ 0 };
        bool paused{ false };
        glm::quat orientation;
#endif

    ShapesRenderer(const vks::Context& context, bool stereo = false)
        : Parent(context)
        , stereo(stereo) {
        srand((uint32_t)time(NULL));
    }

    ~ShapesRenderer() {
        queue.waitIdle();
        device.waitIdle();
        context.device.freeCommandBuffers(cmdPool, cmdBuffer);
        context.device.destroyPipeline(pipelines.solid);
        context.device.destroyPipelineLayout(pipelineLayout);
        context.device.destroyDescriptorSetLayout(descriptorSetLayout);
        uniformData.vsScene.destroy();
    }

    void buildCommandBuffer() {
        if (!cmdBuffer) {
            vk::CommandBufferAllocateInfo cmdBufAllocateInfo;
            cmdBufAllocateInfo.commandPool = cmdPool;
            cmdBufAllocateInfo.commandBufferCount = 1;
            cmdBufAllocateInfo.level = vk::CommandBufferLevel::ePrimary;
            cmdBuffer = context.device.allocateCommandBuffers(cmdBufAllocateInfo)[0];
        }

        cmdBuffer.reset(vk::CommandBufferResetFlagBits::eReleaseResources);

        vk::CommandBufferBeginInfo cmdBufInfo;
        cmdBufInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
        cmdBuffer.begin(cmdBufInfo);

        vk::ClearValue clearValues[2];
        clearValues[0].color = vks::util::clearColor({ 0.2f, 0.2f, 0.2f, 1 });
        clearValues[1].depthStencil = { 1.0f, 0 };

        context.setImageLayout(cmdBuffer, framebuffer.colors[0].image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined,
                               vk::ImageLayout::eColorAttachmentOptimal);
        context.setImageLayout(cmdBuffer, framebuffer.depth.image, vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil,
                               vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);

        vk::RenderPassBeginInfo renderPassBeginInfo;
        renderPassBeginInfo.renderPass = renderPass;
        renderPassBeginInfo.renderArea.extent.width = framebufferSize.x;
        renderPassBeginInfo.renderArea.extent.height = framebufferSize.y;
        renderPassBeginInfo.clearValueCount = 2;
        renderPassBeginInfo.pClearValues = clearValues;
        renderPassBeginInfo.framebuffer = framebuffer.framebuffer;
        cmdBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
        cmdBuffer.setScissor(0, vks::util::rect2D(framebufferSize));
        if (stereo) {
            auto viewport = vks::util::viewport(framebufferSize);
            viewport.width /= 2.0f;
            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.solid);
            // Binding point 0 : Mesh vertex buffer
            cmdBuffer.bindVertexBuffers(0, meshes.buffer, { 0 });
            // Binding point 1 : Instance data buffer
            cmdBuffer.bindVertexBuffers(1, instanceBuffer.buffer, { 0 });
            for (uint32_t i = 0; i < 2; ++i) {
                cmdBuffer.setViewport(0, viewport);
                cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet,
                                             { (uint32_t)i * (uint32_t)uniformData.vsScene.alignment });
                cmdBuffer.drawIndirect(indirectBuffer.buffer, 0, SHAPES_COUNT, sizeof(vk::DrawIndirectCommand));
                viewport.x += viewport.width;
            }
        } else {
            cmdBuffer.setViewport(0, vks::util::viewport(framebufferSize));
            cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.solid);
            // Binding point 0 : Mesh vertex buffer
            cmdBuffer.bindVertexBuffers(0, meshes.buffer, { 0 });
            // Binding point 1 : Instance data buffer
            cmdBuffer.bindVertexBuffers(1, instanceBuffer.buffer, { 0 });
            cmdBuffer.drawIndirect(indirectBuffer.buffer, 0, SHAPES_COUNT, sizeof(vk::DrawIndirectCommand));
        }
        cmdBuffer.endRenderPass();
        cmdBuffer.end();
    }

    template <size_t N>
    void appendShape(const geometry::Solid<N>& solid, std::vector<Vertex>& vertices) {
        using namespace geometry;
        using namespace glm;
        using namespace std;
        ShapeVertexData shape;
        shape.baseVertex = vertices.size();

        auto faceCount = solid.faces.size();
        // FIXME triangulate the faces
        auto faceTriangles = triangulatedFaceTriangleCount<N>();
        vertices.reserve(vertices.size() + 3 * faceTriangles);

        vec3 color = vec3(rand(), rand(), rand()) / (float)RAND_MAX;
        color = vec3(0.3f) + (0.7f * color);
        for (size_t f = 0; f < faceCount; ++f) {
            const Face<N>& face = solid.faces[f];
            vec3 normal = solid.getFaceNormal(f);
            for (size_t ft = 0; ft < faceTriangles; ++ft) {
                // Create the vertices for the face
                vertices.push_back({ vec3(solid.vertices[face[0]]), normal, color });
                vertices.push_back({ vec3(solid.vertices[face[2 + ft]]), normal, color });
                vertices.push_back({ vec3(solid.vertices[face[1 + ft]]), normal, color });
            }
        }
        shape.vertices = vertices.size() - shape.baseVertex;
        shapes.push_back(shape);
    }

    void loadShapes() {
        std::vector<Vertex> vertexData;
        size_t vertexCount = 0;
        appendShape<>(geometry::tetrahedron(), vertexData);
        appendShape<>(geometry::octahedron(), vertexData);
        appendShape<>(geometry::cube(), vertexData);
        appendShape<>(geometry::dodecahedron(), vertexData);
        appendShape<>(geometry::icosahedron(), vertexData);
        for (auto& vertex : vertexData) {
            vertex.position *= 0.2f;
        }
        meshes = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eVertexBuffer, vertexData);
    }

    void setupDescriptorPool() {
        // Example uses one ubo
        std::vector<vk::DescriptorPoolSize> poolSizes{
            { uniformType, 1 },
        };
        descriptorPool = context.device.createDescriptorPool({ {}, 1, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        // Binding 0 : Vertex shader uniform buffer
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{ { 0, uniformType, 1, vk::ShaderStageFlagBits::eVertex } };
        descriptorSetLayout = context.device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = context.device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        descriptorSet = context.device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];

        // Binding 0 : Vertex shader uniform buffer
        vk::WriteDescriptorSet writeDescriptorSet;
        writeDescriptorSet.dstSet = descriptorSet;
        writeDescriptorSet.descriptorType = uniformType;
        writeDescriptorSet.dstBinding = 0;
        writeDescriptorSet.pBufferInfo = &uniformData.vsScene.descriptor;
        writeDescriptorSet.descriptorCount = 1;

        context.device.updateDescriptorSets(writeDescriptorSet, nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder builder{ device, pipelineLayout, renderPass };
        builder.loadShader(getAssetPath() + "shaders/indirect/indirect.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/indirect/indirect.frag.spv", vk::ShaderStageFlagBits::eFragment);
        // Mesh vertex buffer (description) at binding point 0
        builder.vertexInputState.bindingDescriptions = { { 0, sizeof(Vertex), vk::VertexInputRate::eVertex },
                                                         { 1, sizeof(InstanceData), vk::VertexInputRate::eInstance } };

        // Attribute descriptions
        // Describes memory layout and shader positions
        auto& attributes = builder.vertexInputState.attributeDescriptions;
        // Per-Vertex attributes
        attributes.push_back({ 0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, position) });
        attributes.push_back({ 1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color) });
        attributes.push_back({ 2, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal) });

        // Instanced attributes
        attributes.push_back({ 3, 1, vk::Format::eR32G32B32Sfloat, offsetof(InstanceData, pos) });
        attributes.push_back({ 4, 1, vk::Format::eR32G32B32Sfloat, offsetof(InstanceData, rot) });
        attributes.push_back({ 5, 1, vk::Format::eR32Sfloat, offsetof(InstanceData, scale) });

        pipelines.solid = builder.create(context.pipelineCache);
    }

    void prepareIndirectData() {
        std::vector<vk::DrawIndirectCommand> indirectData;
        indirectData.resize(SHAPES_COUNT);
        for (auto i = 0; i < SHAPES_COUNT; ++i) {
            auto& drawIndirectCommand = indirectData[i];
            const auto& shapeData = shapes[i];
            drawIndirectCommand.firstInstance = i * INSTANCES_PER_SHAPE;
            drawIndirectCommand.instanceCount = INSTANCES_PER_SHAPE;
            drawIndirectCommand.firstVertex = (uint32_t)shapeData.baseVertex;
            drawIndirectCommand.vertexCount = (uint32_t)shapeData.vertices;
        }
        indirectBuffer = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eIndirectBuffer, indirectData);
    }

    void prepareInstanceData() {
        std::vector<InstanceData> instanceData;
        instanceData.resize(INSTANCE_COUNT);

        std::mt19937 rndGenerator((uint32_t)time(nullptr));
        std::uniform_real_distribution<float> uniformDist(0.0, 1.0);
        std::exponential_distribution<float> expDist(1);

        for (auto i = 0; i < INSTANCE_COUNT; i++) {
            auto& instance = instanceData[i];
            instance.rot = glm::vec3(M_PI * uniformDist(rndGenerator), M_PI * uniformDist(rndGenerator), M_PI * uniformDist(rndGenerator));
            float theta = 2 * (float)M_PI * uniformDist(rndGenerator);
            float phi = acos(1 - 2 * uniformDist(rndGenerator));
            instance.scale = 0.1f + expDist(rndGenerator) * 3.0f;
            instance.pos = glm::normalize(glm::vec3(sin(phi) * cos(theta), sin(theta), cos(phi)));
            instance.pos *= instance.scale * (1.0f + expDist(rndGenerator) / 2.0f) * 4.0f;
        }

        instanceBuffer = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eVertexBuffer, instanceData);
    }

    void prepareUniformBuffers() { uniformData.vsScene = context.createUniformBuffer(uboVS); }

    void prepare() {
        depthFormat = context.getSupportedDepthFormat();
        OffscreenRenderer::prepare();
        loadShapes();
        prepareInstanceData();
        prepareIndirectData();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffer();
    }

    void update(float deltaTime, const glm::mat4& projection, const glm::mat4& view) { update(deltaTime, { projection, projection }, { view, view }); }

    void update(float deltaTime, const std::array<glm::mat4, 2>& projections, const std::array<glm::mat4, 2>& views) {
        uboVS.time += deltaTime * 0.05f;
        uboVS.projection = projections[0];
        uboVS.view = views[0];
        uniformData.vsScene.copy(uboVS);

        uboVS.projection = projections[1];
        uboVS.view = views[1];
        uniformData.vsScene.copy(uboVS, uniformData.vsScene.alignment);
#if 0
            frameTimer = deltaTime;
            if (!paused) {
                accumulator += frameTimer;
                if (accumulator < duration) {
                    zoom = easings::inOutQuint(accumulator, duration, zoomStart, zoomDelta);
                    updateUniformBuffer(true);
                } else {
                    updateUniformBuffer(false);
                }

                if (accumulator >= interval) {
                    accumulator = 0;
                    zoomStart = zoom;
                    if (zoom < -2) {
                        zoomDelta = 135;
                    } else {
                        zoomDelta = -135;
                    }
                }
            }
#endif
    }

    void render(const vk::ArrayProxy<const vks::Context::SemaphoreStagePair>& wait,
                const vk::ArrayProxy<const vk::Semaphore>& signals,
                const vk::Fence& fence = vk::Fence()) {
        context.submit(cmdBuffer, wait, signals, fence);
    }

    void render() { render({ { semaphores.renderStart, vk::PipelineStageFlagBits::eBottomOfPipe } }, { semaphores.renderComplete }); }

    void renderWithoutSemaphores() { render({}, {}); }
};

}  // namespace vkx
