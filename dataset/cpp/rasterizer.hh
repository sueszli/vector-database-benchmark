#ifndef VKHR_RASTERIZER_HH
#define VKHR_RASTERIZER_HH

#include <vkhr/vkhr.hh>

#include <vkhr/rasterizer/model.hh>
#include <vkhr/rasterizer/hair_style.hh>
#include <vkhr/rasterizer/billboard.hh>
#include <vkhr/rasterizer/linked_list.hh>
#include <vkhr/rasterizer/volume.hh>

#include <vkhr/rasterizer/depth_map.hh>
#include <vkhr/renderer.hh>
#include <vkhr/rasterizer/pipeline.hh>

#include <vkpp/vkpp.hh>

#include <queue>
#include <vector>
#include <unordered_map>
#include <string>

namespace vk = vkpp;

namespace vkhr {
    class Rasterizer final : public Renderer {
    public:
        Rasterizer(Window& window, const SceneGraph& scene_graph);

        void build_render_passes();
        void recreate_swapchain(Window& window, SceneGraph& scene_graph);
        void build_pipelines();

        void load(const SceneGraph& scene) override;
        void update(const SceneGraph& scene_graphs);

        std::uint32_t fetch_next_frame();

        void draw(const SceneGraph& scene) override;
        void draw(Image& fullscreen_image);

        void draw_depth(const SceneGraph& scene_graph, vk::CommandBuffer& command_buffer);
        void draw_model(const SceneGraph& scene_graph, Pipeline& pipeline, vk::CommandBuffer& command_buffer, glm::mat4 = glm::mat4 { 1.0f });
        void draw_color(const SceneGraph& scene_graph, vk::CommandBuffer& command_buffer);
        void draw_hairs(const SceneGraph& scene_graph, Pipeline& pipeline, vk::CommandBuffer& command_buffer, glm::mat4 = glm::mat4 { 1.0f });
        void voxelize(const SceneGraph& a_scene_graph, vk::CommandBuffer& command_buffer);

        // Direct Volume Render (DVR) the hair strands. This needs to be done after drawing models and styles.
        void strand_dvr(const SceneGraph& scene_graph, Pipeline& pipeline, vk::CommandBuffer& command_buffer);

        void destroy_pipelines();
        void destroy_render_passes();
        bool recompile_pipeline_shaders(Pipeline& pipeline);
        void recompile();

        bool swapchain_is_dirty() const;

        Interface& get_imgui();

        struct Benchmark {
            std::string description;
            std::string scene;

            int width, height;

            Renderer::Type renderer;

            float viewing_distance;
            float strand_reduction;
            int   raymarch_steps;
        };

        void append_benchmark(const Benchmark& benchmark_parameters);
        void run_benchmarks(SceneGraph& scene_graph);
        void append_benchmarks(const std::vector<Benchmark>& params);

        bool benchmark(SceneGraph& scene_graph);

        Image get_screenshot(const SceneGraph& scene_graph);
        Image get_screenshot(const SceneGraph& scene_graph,
                             Raytracer& raytracer_instance);

    private:
        Image get_screenshot();

        vk::Instance instance;
        vk::PhysicalDevice physical_device;
        vk::Device device;

        vk::CommandPool command_pool;

        vk::Surface window_surface;
        vk::SwapChain swap_chain;

        mutable bool swapchain_dirty { false };

        vk::RenderPass depth_pass;
        vk::RenderPass color_pass;
        vk::RenderPass imgui_pass;

        vk::DescriptorPool descriptor_pool;

        std::vector<vkpp::Framebuffer> framebuffers;
        std::vector<vk::Semaphore> image_available, render_complete;
        std::vector<vk::Fence> command_buffer_finished;

        vk::Sampler depth_sampler;

        std::uint32_t frame { 0 };
        std::uint32_t latest_drawn_frame { 0 };
        float level_of_detail = 0;

        std::vector<vk::UniformBuffer> camera;
        std::vector<vk::UniformBuffer> lights;
        std::vector<vk::UniformBuffer> params;

        Pipeline hair_depth_pipeline;
        Pipeline mesh_depth_pipeline;
        Pipeline hair_voxel_pipeline;

        Pipeline strand_dvr_pipeline;
        Pipeline ppll_blend_pipeline;

        Pipeline hair_style_pipeline;
        Pipeline model_mesh_pipeline;
        Pipeline billboards_pipeline;

        std::vector<vulkan::DepthMap> shadow_maps;
        std::unordered_map<const HairStyle*, vulkan::HairStyle> hair_styles;
        std::unordered_map<const Model*, vulkan::Model> models;
        vulkan::Billboard fullscreen_billboard;

        vulkan::LinkedList ppll;

        Interface imgui;

        void set_benchmark_configurations(const Benchmark& benchmark,       SceneGraph& scene_graph);
        std::string get_benchmark_results(const Benchmark& benchmark, const SceneGraph& scene_graph,
                                          const Image& screenshot); // For finding the shaded pixels.
        std::string get_benchmark_header();

        std::string final_benchmark_csv { "" };
        std::string benchmark_start_time;
        int benchmark_counter  = 0;
        int queued_benchmarks  = 0;
        std::queue<Benchmark> benchmark_queue;
        int frames_benchmarked = 0;
        Benchmark loaded_benchmark;
        std::string benchmark_directory { "" };

        std::vector<vk::QueryPool> query_pools;

        std::vector<vk::CommandBuffer> command_buffers;

        friend class vulkan::HairStyle;
        friend class vulkan::Model;
        friend class vulkan::Volume;
        friend class vulkan::Billboard;
        friend class vulkan::LinkedList;

        friend class vulkan::DepthMap;

        friend class ::vkhr::Interface;
    };
}

#endif
