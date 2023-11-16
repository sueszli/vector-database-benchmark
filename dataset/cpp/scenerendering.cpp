/*
* Vulkan Example -  Rendering a scene with multiple meshes and materials
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanExampleBase.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>

// Vertex layout used in this example
struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
    glm::vec3 color;
};

vks::model::VertexLayout vertexLayout{ {
    vks::model::VERTEX_COMPONENT_POSITION,
    vks::model::VERTEX_COMPONENT_NORMAL,
    vks::model::VERTEX_COMPONENT_UV,
    vks::model::VERTEX_COMPONENT_COLOR,
} };

// Scene related structs

// Shader properites for a material
// Will be passed to the shaders using push constant
struct SceneMaterialProperites {
    glm::vec4 ambient;
    glm::vec4 diffuse;
    glm::vec4 specular;
    float opacity;
};

// Stores info on the materials used in the scene
struct SceneMaterial {
    std::string name;
    // Material properties
    SceneMaterialProperites properties;
    // The example only uses a diffuse channel
    vks::texture::Texture2D diffuse;
    // The material's descriptor contains the material descriptors
    vk::DescriptorSet descriptorSet;
    // Pointer to the pipeline used by this material
    vk::Pipeline* pipeline;
};

// Stores per-mesh Vulkan resources
struct SceneMesh {
    vks::Buffer vertices;
    vks::Buffer indices;
    uint32_t indexCount;

    // Pointer to the material used by this mesh
    SceneMaterial* material;
};

// Class for loading the scene and generating all Vulkan resources
class Scene {
private:
    const vks::Context& context;
    const vk::Device& device{ context.device };
    const vk::Queue& queue{ context.queue };

    vk::DescriptorPool descriptorPool;

    // We will be using separate descriptor sets (and bindings)
    // for material and scene related uniforms
    struct {
        vk::DescriptorSetLayout material;
        vk::DescriptorSetLayout scene;
    } descriptorSetLayouts;

    vk::DescriptorSet descriptorSetScene;

    const aiScene* aScene;

    // Get materials from the assimp scene and map to our scene structures
    void loadMaterials() {
        materials.resize(aScene->mNumMaterials);

        for (size_t i = 0; i < materials.size(); i++) {
            materials[i] = {};

            aiString name;
            aScene->mMaterials[i]->Get(AI_MATKEY_NAME, name);

            // Properties
            aiColor4D color;
            aScene->mMaterials[i]->Get(AI_MATKEY_COLOR_AMBIENT, color);
            materials[i].properties.ambient = glm::make_vec4(&color.r) + glm::vec4(0.1f);
            aScene->mMaterials[i]->Get(AI_MATKEY_COLOR_DIFFUSE, color);
            materials[i].properties.diffuse = glm::make_vec4(&color.r);
            aScene->mMaterials[i]->Get(AI_MATKEY_COLOR_SPECULAR, color);
            materials[i].properties.specular = glm::make_vec4(&color.r);
            aScene->mMaterials[i]->Get(AI_MATKEY_OPACITY, materials[i].properties.opacity);

            if ((materials[i].properties.opacity) > 0.0f)
                materials[i].properties.specular = glm::vec4(0.0f);

            materials[i].name = name.C_Str();
            std::cout << "Material \"" << materials[i].name << "\"" << std::endl;

            // Textures
            aiString texturefile;
            // Diffuse
            aScene->mMaterials[i]->GetTexture(aiTextureType_DIFFUSE, 0, &texturefile);
            if (aScene->mMaterials[i]->GetTextureCount(aiTextureType_DIFFUSE) > 0) {
                std::cout << "  Diffuse: \"" << texturefile.C_Str() << "\"" << std::endl;
                std::string fileName = std::string(texturefile.C_Str());
                std::replace(fileName.begin(), fileName.end(), '\\', '/');
                materials[i].diffuse.loadFromFile(context, assetPath + fileName, vk::Format::eBc3UnormBlock);
            } else {
                std::cout << "  Material has no diffuse, using dummy texture!" << std::endl;
                // todo : separate pipeline and layout
                materials[i].diffuse.loadFromFile(context, assetPath + "dummy.ktx", vk::Format::eBc2UnormBlock);
            }

            // For scenes with multiple textures per material we would need to check for additional texture types, e.g.:
            // aiTextureType_HEIGHT, aiTextureType_OPACITY, aiTextureType_SPECULAR, etc.

            // Assign pipeline
            materials[i].pipeline = (materials[i].properties.opacity == 0.0f) ? &pipelines.solid : &pipelines.blending;
        }

        // Generate descriptor sets for the materials

        // Descriptor pool
        std::vector<vk::DescriptorPoolSize> poolSizes{
            { vk::DescriptorType::eUniformBuffer, static_cast<uint32_t>(materials.size()) },
            { vk::DescriptorType::eCombinedImageSampler, static_cast<uint32_t>(materials.size()) },
        };

        vk::DescriptorPoolCreateInfo descriptorPoolInfo{ {},
                                                         static_cast<uint32_t>(materials.size()) + 1,
                                                         static_cast<uint32_t>(poolSizes.size()),
                                                         poolSizes.data() };

        descriptorPool = device.createDescriptorPool(descriptorPoolInfo);

        // Descriptor set and pipeline layouts
        // Set 0: Scene matrices
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, 0 },
        };
        descriptorSetLayouts.scene = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });

        // Set 1: Material data
        setLayoutBindings = {
            { 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };
        descriptorSetLayouts.material = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });

        // Setup pipeline layout
        std::array<vk::DescriptorSetLayout, 2> setLayouts = { descriptorSetLayouts.scene, descriptorSetLayouts.material };
        // We will be using a push constant block to pass material properties to the fragment shaders
        vk::PushConstantRange pushConstantRange{ vk::ShaderStageFlagBits::eFragment, 0, sizeof(SceneMaterialProperites) };
        pipelineLayout = device.createPipelineLayout({ {}, static_cast<uint32_t>(setLayouts.size()), setLayouts.data(), 1, &pushConstantRange });

        // Material descriptor sets
        for (size_t i = 0; i < materials.size(); i++) {
            // Descriptor set
            materials[i].descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.material })[0];

            vk::DescriptorImageInfo texDescriptor{ materials[i].diffuse.sampler, materials[i].diffuse.view, vk::ImageLayout::eGeneral };
            std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
                // Binding 0: Diffuse texture
                { materials[i].descriptorSet, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptor },
            };

            device.updateDescriptorSets(writeDescriptorSets, nullptr);
        }

        // Scene descriptor set
        descriptorSetScene = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.scene })[0];
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            // Binding 0 : Vertex shader uniform buffer
            { descriptorSetScene, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffer.descriptor },
        };
        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    // Load all meshes from the scene and generate the Vulkan resources
    // for rendering them
    void loadMeshes(vk::CommandBuffer copyCmd) {
        meshes.resize(aScene->mNumMeshes);
        for (uint32_t i = 0; i < meshes.size(); i++) {
            aiMesh* aMesh = aScene->mMeshes[i];

            std::cout << "Mesh \"" << aMesh->mName.C_Str() << "\"" << std::endl;
            std::cout << "    Material: \"" << materials[aMesh->mMaterialIndex].name << "\"" << std::endl;
            std::cout << "    Faces: " << aMesh->mNumFaces << std::endl;

            meshes[i].material = &materials[aMesh->mMaterialIndex];

            // Vertices
            std::vector<Vertex> vertices;
            vertices.resize(aMesh->mNumVertices);

            bool hasUV = aMesh->HasTextureCoords(0);
            bool hasColor = aMesh->HasVertexColors(0);
            bool hasNormals = aMesh->HasNormals();

            for (uint32_t v = 0; v < aMesh->mNumVertices; v++) {
                vertices[v].pos = glm::make_vec3(&aMesh->mVertices[v].x);
                vertices[v].pos.y = -vertices[v].pos.y;
                vertices[v].uv = hasUV ? glm::make_vec2(&aMesh->mTextureCoords[0][v].x) : glm::vec2(0.0f);
                vertices[v].normal = hasNormals ? glm::make_vec3(&aMesh->mNormals[v].x) : glm::vec3(0.0f);
                vertices[v].normal.y = -vertices[v].normal.y;
                vertices[v].color = hasColor ? glm::make_vec3(&aMesh->mColors[0][v].r) : glm::vec3(1.0f);
            }
            meshes[i].vertices = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eVertexBuffer, vertices);

            // Indices
            std::vector<uint32_t> indices;
            meshes[i].indexCount = aMesh->mNumFaces * 3;
            indices.resize(aMesh->mNumFaces * 3);
            for (uint32_t f = 0; f < aMesh->mNumFaces; f++) {
                memcpy(&indices[f * 3], &aMesh->mFaces[f].mIndices[0], sizeof(uint32_t) * 3);
            }
            meshes[i].indices = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eIndexBuffer, indices);
        }
    }

public:
    std::string assetPath = "";

    std::vector<SceneMaterial> materials;
    std::vector<SceneMesh> meshes;

    // Shared ubo containing matrices used by all
    // materials and meshes
    vks::Buffer uniformBuffer;
    struct {
        glm::mat4 projection;
        glm::mat4 view;
        glm::mat4 model;
        glm::vec4 lightPos = glm::vec4(1.25f, 8.35f, 0.0f, 0.0f);
    } uniformData;

    // Scene uses multiple pipelines
    struct {
        vk::Pipeline solid;
        vk::Pipeline blending;
        vk::Pipeline wireframe;
    } pipelines;

    // Shared pipeline layout
    vk::PipelineLayout pipelineLayout;

    // For displaying only a single part of the scene
    bool renderSingleScenePart = false;
    uint32_t scenePartIndex = 0;

    Scene(const vks::Context& context)
        : context(context) {
        uniformBuffer = context.createUniformBuffer(uniformData);
    }

    ~Scene() {
        for (auto mesh : meshes) {
            mesh.vertices.destroy();
            mesh.indices.destroy();
        }
        for (auto material : materials) {
            material.diffuse.destroy();
        }
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.material, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.scene, nullptr);
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyPipeline(device, pipelines.solid, nullptr);
        vkDestroyPipeline(device, pipelines.blending, nullptr);
        vkDestroyPipeline(device, pipelines.wireframe, nullptr);
        uniformBuffer.destroy();
    }

    void load(const std::string& filename, vk::CommandBuffer copyCmd) {
        Assimp::Importer Importer;
        vks::file::withBinaryFileContents(filename, [&](size_t size, const void* data) {
            int flags = aiProcess_PreTransformVertices | aiProcess_Triangulate | aiProcess_GenNormals;
            aScene = Importer.ReadFileFromMemory(data, size, flags);
        });

        if (aScene) {
            loadMaterials();
            loadMeshes(copyCmd);
        } else {
            printf("Error parsing '%s': '%s'\n", filename.c_str(), Importer.GetErrorString());
        }
    }

    // Renders the scene into an active command buffer
    // In a real world application we would do some visibility culling in here
    void render(vk::CommandBuffer cmdBuffer, bool wireframe) {
        vk::DeviceSize offsets[1] = { 0 };
        for (size_t i = 0; i < meshes.size(); i++) {
            if ((renderSingleScenePart) && (i != scenePartIndex))
                continue;

            //if (meshes[i].material->opacity == 0.0f)
            //    continue;

            // todo : per material pipelines
            //            vkCmdBindPipeline(cmdBuffer, vk::PipelineBindPoint::eGraphics, *mesh.material->pipeline);

            // We will be using multiple descriptor sets for rendering
            // In GLSL the selection is done via the set and binding keywords
            // VS: layout (set = 0, binding = 0) uniform UBO;
            // FS: layout (set = 1, binding = 0) uniform sampler2D samplerColorMap;

            std::array<vk::DescriptorSet, 2> descriptorSets;
            // Set 0: Scene descriptor set containing global matrices
            descriptorSets[0] = descriptorSetScene;
            // Set 1: Per-Material descriptor set containing bound images
            descriptorSets[1] = meshes[i].material->descriptorSet;

            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, wireframe ? pipelines.wireframe : *meshes[i].material->pipeline);
            cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets, {});

            // Pass material properies via push constants
            vkCmdPushConstants(cmdBuffer, pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SceneMaterialProperites), &meshes[i].material->properties);

            cmdBuffer.bindVertexBuffers(0, meshes[i].vertices.buffer, { 0 });
            cmdBuffer.bindIndexBuffer(meshes[i].indices.buffer, 0, vk::IndexType::eUint32);
            cmdBuffer.drawIndexed(meshes[i].indexCount, 1, 0, 0, 0);
        }

        // Render transparent objects last
    }
};

class VulkanExample : public vkx::ExampleBase {
    using Parent = ExampleBase;

public:
    bool wireframe = false;
    bool attachLight = false;

    Scene* scene = nullptr;

    VulkanExample() {
        rotationSpeed = 0.5f;
        camera.type = Camera::CameraType::firstperson;
        camera.movementSpeed = 7.5f;
        camera.position = { 15.0f, -13.5f, 0.0f };
        camera.setRotation(glm::vec3(5.0f, 90.0f, 0.0f));
        camera.setPerspective(60.0f, size, 0.1f, 256.0f);
        title = "Vulkan Example - Scene rendering";
    }

    ~VulkanExample() { delete (scene); }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        scene->render(cmdBuffer, wireframe);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, scene->pipelineLayout, renderPass };
        pipelineBuilder.vertexInputState.appendVertexLayout(vertexLayout);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/scenerendering/scene.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/scenerendering/scene.frag.spv", vk::ShaderStageFlagBits::eFragment);
        // Solid frame rendering pipeline
        scene->pipelines.solid = pipelineBuilder.create(context.pipelineCache);

        // Wire frame rendering pipeline
        pipelineBuilder.rasterizationState.polygonMode = vk::PolygonMode::eLine;
        scene->pipelines.wireframe = pipelineBuilder.create(context.pipelineCache);

        // Alpha blended pipeline
        pipelineBuilder.rasterizationState.polygonMode = vk::PolygonMode::eFill;
        pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        auto& blendAttachmentState = pipelineBuilder.colorBlendState.blendAttachmentStates[0];
        blendAttachmentState.blendEnable = VK_TRUE;
        blendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
        blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eSrcColor;
        blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcColor;
        scene->pipelines.blending = pipelineBuilder.create(context.pipelineCache);
    }

    void updateUniformBuffers() {
        if (attachLight) {
            scene->uniformData.lightPos = glm::vec4(-camera.position, 1.0f);
        }

        scene->uniformData.projection = camera.matrices.perspective;
        scene->uniformData.view = camera.matrices.view;
        scene->uniformData.model = glm::mat4();

        memcpy(scene->uniformBuffer.mapped, &scene->uniformData, sizeof(scene->uniformData));
    }

    void loadScene() {
        context.withPrimaryCommandBuffer([&](const vk::CommandBuffer& cmdBuffer) {
            scene = new Scene(context);
            scene->assetPath = getAssetPath() + "models/sibenik/";
            scene->load(getAssetPath() + "models/sibenik/sibenik.dae", cmdBuffer);
        });

        updateUniformBuffers();
    }

    void prepare() override {
        Parent::prepare();
        loadScene();
        preparePipelines();
        buildCommandBuffers();
        prepared = true;
    }

    void render() override {
        if (!prepared)
            return;
        draw();
    }

    void viewChanged() override { updateUniformBuffers(); }

    void keyPressed(uint32_t keyCode) override {
        Parent::keyPressed(keyCode);
        switch (keyCode) {
            case KEY_W:
            case GAMEPAD_BUTTON_A:
                wireframe = !wireframe;
                buildCommandBuffers();
                break;
            case KEY_P:
                scene->renderSingleScenePart = !scene->renderSingleScenePart;
                buildCommandBuffers();
                break;
            case KEY_KPADD:
                scene->scenePartIndex = (scene->scenePartIndex < static_cast<uint32_t>(scene->meshes.size())) ? scene->scenePartIndex + 1 : 0;
                buildCommandBuffers();
                break;
            case KEY_KPSUB:
                scene->scenePartIndex = (scene->scenePartIndex > 0) ? scene->scenePartIndex - 1 : static_cast<uint32_t>(scene->meshes.size()) - 1;
                buildCommandBuffers();
                break;
            case KEY_L:
                attachLight = !attachLight;
                updateUniformBuffers();
                break;
        }
    }
};

RUN_EXAMPLE(VulkanExample)
