#include "application.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include <imgui.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace application {

using graphics::internal::context;

namespace {
// Типы данных
// Вершина
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
};
static_assert(sizeof(Vertex) == 6 * sizeof(float), "Vertex must be tightly packed");

// Данные сцены
struct SceneData {
    glm::mat4 view_proj;
    glm::vec4 camera_position;
    glm::vec4 ambient;
    glm::vec4 dir_direction;
    glm::vec4 dir_color;
};
static_assert(sizeof(SceneData) == 128, "SceneData must match the shader (std140)");

// Один прожектор в storage buffer
struct SpotLightData {
    glm::vec4 position;
    glm::vec4 direction;
    glm::vec4 color;
    glm::vec4 params;
};
static_assert(sizeof(SpotLightData) == 64, "SpotLightData must match the shader (std430)");

// Данные модели
struct ModelData {
    glm::mat4 model;
    glm::mat4 normal_matrix;
    glm::vec4 albedo;
    glm::vec4 specular;
};
static_assert(sizeof(ModelData) == 160, "ModelData must match the shader (std140)");

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
};

struct PointLightSettings {
    glm::vec3 position{0.0f, 1.5f, 1.5f};
    glm::vec3 color{1.0f};
    float intensity = 5.0f;
};

struct Settings {
    // Камера
    glm::vec3 camera_position{0.0f, 0.0f, 4.0f};
    float camera_yaw = 0.0f;
    float camera_pitch = 0.0f;
    float camera_fov = 60.0f;
    int projection = 0;
    float ortho_half_height = 2.3f;

    glm::vec3 light_direction{-0.4f, -0.7f, -0.6f};
    glm::vec3 light_color{1.0f};
    float light_intensity = 1.0f;
    glm::vec3 ambient_color{1.0f};
    float ambient_intensity = 0.1f;

    glm::vec3 albedo{0.85f, 0.45f, 0.25f};
    glm::vec3 specular_color{1.0f};
    float shininess = 32.0f;

    // Преобразования объекта
    glm::vec3 position{0.0f};
    glm::vec3 rotation_degrees{20.0f, 30.0f, 0.0f};
    glm::vec3 scale{1.0f};

    bool playing = false;
    float speed = 1.0f;
    float path_radius = 1.2f;
    float path_height = 0.6f;
    float spin_speed = 60.0f;
};

// Усечённый правильный тетраэдр

constexpr int corners[12][3] = {
    { 3,  1,  1},
    {-3, -1,  1},
    {-3,  1, -1},
    { 3, -1, -1},
    { 1,  3,  1},
    {-1, -3,  1},
    { 1, -3, -1},
    {-1,  3, -1},
    { 1,  1,  3},
    {-1, -1,  3},
    { 1, -1, -3},
    {-1,  1, -3},
};

// Грани
constexpr int triangle_faces[4][3] = {
    {0, 4, 8},
    {1, 5, 9},
    {2, 7, 11},
    {3, 6, 10},
};

constexpr int hexagon_faces[4][6] = {
    {1, 5, 6, 10, 11, 2},
    {0, 3, 10, 11, 7, 4},
    {0, 3, 6, 5, 9, 8},
    {1, 2, 7, 4, 8, 9},
};

// Глобальное состояние

Settings settings;

SceneData gpu_scene{};
ModelData gpu_model{};

double animation_time = 0.0;
double last_time = -1.0;

VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout model_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet scene_set = VK_NULL_HANDLE;
VkDescriptorSet model_set = VK_NULL_HANDLE;

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

Buffer vertex_buffer;
Buffer index_buffer;
Buffer scene_buffer;
Buffer model_buffer;
uint32_t index_count = 0;

// Вспомогательные функции

std::vector<uint32_t> readSpirv(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open shader file " << path << '\n';
        return {};
    }

    const std::streamsize size = file.tellg();
    if (size <= 0 || size % 4 != 0) {
        std::cerr << "Shader file " << path << " has invalid size\n";
        return {};
    }

    std::vector<uint32_t> code(size_t(size) / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), size);
    return code;
}

VkShaderModule createShaderModule(const char* path) {
    const std::vector<uint32_t> code = readSpirv(path);
    if (code.empty()) {
        return VK_NULL_HANDLE;
    }

    const VkShaderModuleCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size() * sizeof(uint32_t),
        .pCode = code.data(),
    };

    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(context.device, &info, nullptr, &module) != VK_SUCCESS) {
        std::cerr << "Failed to create shader module from " << path << '\n';
        return VK_NULL_HANDLE;
    }
    return module;
}

// Создаёт буфер в памяти, доступной процессору, и копирует в него данные
bool createBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
    const VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    const VmaAllocationCreateInfo allocation_info = {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };

    if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info, &out.handle, &out.allocation, nullptr) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan buffer\n";
        return false;
    }

    if (vmaCopyMemoryToAllocation(context.allocator, data, out.allocation, 0, size) != VK_SUCCESS) {
        std::cerr << "Failed to copy data into Vulkan buffer\n";
        return false;
    }

    return true;
}

void destroyBuffer(Buffer& buffer) {
    if (buffer.handle != VK_NULL_HANDLE) {
        vmaDestroyBuffer(context.allocator, buffer.handle, buffer.allocation);
        buffer.handle = VK_NULL_HANDLE;
        buffer.allocation = nullptr;
    }
}

// Добавляет одну грань в массивы вершин и индексов
void addFace(const int* ids, int count, std::vector<Vertex>& vertices, std::vector<uint16_t>& indices) {
    glm::vec3 p[6];
    glm::vec3 centroid(0.0f);

    for (int i = 0; i < count; ++i) {
        p[i] = glm::vec3(float(corners[ids[i]][0]), float(corners[ids[i]][1]), float(corners[ids[i]][2])) / 3.0f;
        centroid += p[i];
    }
    centroid /= float(count);

    glm::vec3 normal = glm::normalize(glm::cross(p[1] - p[0], p[2] - p[0]));

    const bool flipped = glm::dot(normal, centroid) < 0.0f;
    if (flipped) {
        normal = -normal;
    }

    const int base = int(vertices.size());
    for (int i = 0; i < count; ++i) {
        vertices.push_back(Vertex{p[i], normal});
    }

    for (int i = 1; i + 1 < count; ++i) {
        const int a = base;
        const int b = base + (flipped ? i + 1 : i);
        const int c = base + (flipped ? i : i + 1);
        indices.push_back(uint16_t(a));
        indices.push_back(uint16_t(b));
        indices.push_back(uint16_t(c));
    }
}

// Макет набора дескрипторов
VkDescriptorSetLayout createUniformSetLayout() {
    const VkDescriptorSetLayoutBinding binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
    };

    const VkDescriptorSetLayoutCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &binding,
    };

    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(context.device, &info, nullptr, &layout) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor set layout\n";
        return VK_NULL_HANDLE;
    }
    return layout;
}

bool createDescriptors() {
    scene_set_layout = createUniformSetLayout();
    model_set_layout = createUniformSetLayout();
    if (scene_set_layout == VK_NULL_HANDLE || model_set_layout == VK_NULL_HANDLE) {
        return false;
    }

    // Пул, из которого выделяются наборы
    const VkDescriptorPoolSize pool_sizes[] = {
        { .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 2 },
        { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 },
    };

    const VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 2,
        .poolSizeCount = 2,
        .pPoolSizes = pool_sizes,
    };

    if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor pool\n";
        return false;
    }

    const VkDescriptorSetLayout layouts[] = { scene_set_layout, model_set_layout };
    const VkDescriptorSetAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 2,
        .pSetLayouts = layouts,
    };

    VkDescriptorSet sets[2] = {};
    if (vkAllocateDescriptorSets(context.device, &allocate_info, sets) != VK_SUCCESS) {
        std::cerr << "Failed to allocate descriptor sets\n";
        return false;
    }
    scene_set = sets[0];
    model_set = sets[1];

    const VkDescriptorBufferInfo scene_info = {
        .buffer = scene_buffer.handle,
        .offset = 0,
        .range = sizeof(SceneData),
    };
    const VkDescriptorBufferInfo model_info = {
        .buffer = model_buffer.handle,
        .offset = 0,
        .range = sizeof(ModelData),
    };

    const VkWriteDescriptorSet writes[] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = scene_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &scene_info,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = model_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &model_info,
        },
    };
    vkUpdateDescriptorSets(context.device, 2, writes, 0, nullptr);

    return true;
}

bool createPipeline() {
    VkShaderModule vert = createShaderModule("shaders/lit.vert.spv");
    VkShaderModule frag = createShaderModule("shaders/lit.frag.spv");

    const auto cleanup = [&](bool result) {
        vkDestroyShaderModule(context.device, vert, nullptr);
        vkDestroyShaderModule(context.device, frag, nullptr);
        return result;
    };

    if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) {
        return cleanup(false);
    }

    const VkDescriptorSetLayout set_layouts[] = { scene_set_layout, model_set_layout };
    const VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 2,
        .pSetLayouts = set_layouts,
    };

    if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
        std::cerr << "Failed to create pipeline layout\n";
        return cleanup(false);
    }

    const VkPipelineShaderStageCreateInfo stages[] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vert,
            .pName = "main",
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = frag,
            .pName = "main",
        },
    };

    const VkVertexInputBindingDescription binding = {
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };

    // location 0 - позиция, location 1 - нормаль
    const VkVertexInputAttributeDescription attributes[] = {
        {
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, position),
        },
        {
            .location = 1,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, normal),
        },
    };

    const VkPipelineVertexInputStateCreateInfo vertex_input = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = 2,
        .pVertexAttributeDescriptions = attributes,
    };

    const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };

    const VkPipelineViewportStateCreateInfo viewport_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    const VkPipelineRasterizationStateCreateInfo rasterization = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .lineWidth = 1.0f,
    };

    const VkPipelineMultisampleStateCreateInfo multisample = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
    };

    const VkPipelineColorBlendAttachmentState blend_attachment = {
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    const VkPipelineColorBlendStateCreateInfo color_blend = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &blend_attachment,
    };

    const VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    const VkPipelineDynamicStateCreateInfo dynamic_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = dynamic_states,
    };

    const VkGraphicsPipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = stages,
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depth_stencil,
        .pColorBlendState = &color_blend,
        .pDynamicState = &dynamic_state,
        .layout = pipeline_layout,
        .renderPass = context.render_pass,
        .subpass = 0,
    };

    if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
        std::cerr << "Failed to create graphics pipeline\n";
        return cleanup(false);
    }

    return cleanup(true);
}

} // namespace

// Функции приложения, которые вызывает main
bool initialize() {
    // Геометрия: 4 треугольника + 4 шестиугольника
    std::vector<Vertex> vertices;
    std::vector<uint16_t> indices;

    for (const auto& face : triangle_faces) {
        addFace(face, 3, vertices, indices);
    }
    for (const auto& face : hexagon_faces) {
        addFace(face, 6, vertices, indices);
    }
    index_count = uint32_t(indices.size());

    if (!createBuffer(vertices.data(), vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer)) {
        return false;
    }

    if (!createBuffer(indices.data(), indices.size() * sizeof(uint16_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer)) {
        return false;
    }

    // Uniform-буферы создаются один раз, а содержимое обновляется каждый кадр.
    if (!createBuffer(&gpu_scene, sizeof(SceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, scene_buffer)) {
        return false;
    }

    if (!createBuffer(&gpu_model, sizeof(ModelData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, model_buffer)) {
        return false;
    }

    return createDescriptors() && createPipeline();
}

void shutdown() {
    vkQueueWaitIdle(context.graphics_queue);

    vkDestroyPipeline(context.device, pipeline, nullptr);
    vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);

    vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
    vkDestroyDescriptorSetLayout(context.device, model_set_layout, nullptr);
    vkDestroyDescriptorSetLayout(context.device, scene_set_layout, nullptr);

    destroyBuffer(model_buffer);
    destroyBuffer(scene_buffer);
    destroyBuffer(index_buffer);
    destroyBuffer(vertex_buffer);
}

void update(double time) {
    Settings& s = settings;

    // Часы анимации
    if (last_time < 0.0) {
        last_time = time;
    }
    const double dt = std::fmin(time - last_time, 0.1);
    last_time = time;
    if (s.playing) {
        animation_time += dt * double(s.speed);
    }

    // Интерфейс
    ImGui::Begin("Lab 2: camera and lighting");

    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragFloat3("Camera position", &s.camera_position.x, 0.02f);
        ImGui::SliderFloat("Yaw (deg)", &s.camera_yaw, -180.0f, 180.0f);
        ImGui::SliderFloat("Pitch (deg)", &s.camera_pitch, -89.0f, 89.0f);
        ImGui::RadioButton("Perspective", &s.projection, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Orthographic", &s.projection, 1);
        if (s.projection == 0) {
            ImGui::SliderFloat("FOV (deg)", &s.camera_fov, 20.0f, 120.0f);
        } else {
            ImGui::SliderFloat("Half height", &s.ortho_half_height, 0.5f, 5.0f);
        }
    }

    if (ImGui::CollapsingHeader("Directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat3("Direction", &s.light_direction.x, -1.0f, 1.0f);
        ImGui::ColorEdit3("Light color", &s.light_color.x);
        ImGui::SliderFloat("Light intensity", &s.light_intensity, 0.0f, 5.0f);
        ImGui::ColorEdit3("Ambient color", &s.ambient_color.x);
        ImGui::SliderFloat("Ambient intensity", &s.ambient_intensity, 0.0f, 1.0f);
    }

    if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Albedo", &s.albedo.x);
        ImGui::ColorEdit3("Specular color", &s.specular_color.x);
        ImGui::SliderFloat("Shininess", &s.shininess, 1.0f, 256.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
    }

    if (ImGui::CollapsingHeader("Object")) {
        ImGui::DragFloat3("Position", &s.position.x, 0.01f);
        ImGui::SliderFloat3("Rotation (deg)", &s.rotation_degrees.x, -180.0f, 180.0f);
        ImGui::DragFloat3("Scale", &s.scale.x, 0.01f, 0.1f, 5.0f);
    }

    if (ImGui::CollapsingHeader("Animation")) {
        if (ImGui::Button(s.playing ? "Pause" : "Play")) {
            s.playing = !s.playing;
        }
        ImGui::SameLine();
        if (ImGui::Button("Restart")) {
            animation_time = 0.0;
        }
        ImGui::SliderFloat("Speed", &s.speed, 0.0f, 5.0f);
        ImGui::SliderFloat("Path radius", &s.path_radius, 0.0f, 3.0f);
        ImGui::SliderFloat("Path height", &s.path_height, 0.0f, 2.0f);
        ImGui::SliderFloat("Spin (deg/s)", &s.spin_speed, 0.0f, 360.0f);
    }

    ImGui::Separator();
    if (ImGui::Button("Reset all")) {
        s = Settings{};
        animation_time = 0.0;
    }

    ImGui::End();

    // Камера
    const VkExtent2D extent = context.swapchain_extent;
    const float aspect = float(extent.width) / float(extent.height);

    glm::mat4 projection(1.0f);
    if (s.projection == 0) {
        projection = glm::perspective(glm::radians(s.camera_fov), aspect, 0.1f, 100.0f);
    } else {
        const float h = s.ortho_half_height;
        const float w = h * aspect;
        projection = glm::ortho(-w, w, -h, h, 0.1f, 100.0f);
    }

    glm::mat4 camera_transform = glm::translate(glm::mat4(1.0f), s.camera_position);
    camera_transform = glm::rotate(camera_transform, glm::radians(s.camera_yaw), glm::vec3(0.0f, 1.0f, 0.0f));
    camera_transform = glm::rotate(camera_transform, glm::radians(s.camera_pitch), glm::vec3(1.0f, 0.0f, 0.0f));
    const glm::mat4 view = glm::inverse(camera_transform);

    // Модель
    const float t = float(animation_time);
    const glm::vec3 path_offset(
        s.path_radius * std::sin(t),
        s.path_height * std::sin(2.0f * t),
        0.5f * s.path_radius * std::sin(3.0f * t));
    const float spin = s.spin_speed * t;

    glm::mat4 model = glm::translate(glm::mat4(1.0f), s.position + path_offset);
    model = glm::rotate(model, glm::radians(s.rotation_degrees.x + 0.5f * spin), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, glm::radians(s.rotation_degrees.y + spin), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, glm::radians(s.rotation_degrees.z), glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::scale(model, s.scale);

    // Данные для видеокарты
    glm::vec3 light_direction = s.light_direction;
    if (glm::length(light_direction) < 1e-4f) {
        light_direction = glm::vec3(0.0f, -1.0f, 0.0f);
    } else {
        light_direction = glm::normalize(light_direction);
    }

    gpu_scene.view_proj = projection * view;
    gpu_scene.camera_position = glm::vec4(s.camera_position, 1.0f);
    gpu_scene.ambient = glm::vec4(s.ambient_color, s.ambient_intensity);
    gpu_scene.dir_direction = glm::vec4(light_direction, 0.0f);
    gpu_scene.dir_color = glm::vec4(s.light_color, s.light_intensity);

    gpu_model.model = model;
    gpu_model.normal_matrix = glm::transpose(glm::inverse(model));
    gpu_model.albedo = glm::vec4(s.albedo, 1.0f);
    gpu_model.specular = glm::vec4(s.specular_color, s.shininess);

}

void render(const graphics::internal::FrameData& fd) {
    const VkCommandBuffer cmd = fd.command_buffer;
    const VkExtent2D extent = context.swapchain_extent;

    // Загружаем данные в uniform-буферы
    vmaCopyMemoryToAllocation(context.allocator, &gpu_scene, scene_buffer.allocation, 0, sizeof(SceneData));
    vmaCopyMemoryToAllocation(context.allocator, &gpu_model, model_buffer.allocation, 0, sizeof(ModelData));

    vkResetCommandBuffer(cmd, 0);

    const VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &begin_info);

    VkClearValue clear_values[2] = {};
    clear_values[0].color = {{0.08f, 0.08f, 0.10f, 1.0f}};
    clear_values[1].depthStencil = {1.0f, 0};

    const VkRenderPassBeginInfo pass_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = context.render_pass,
        .framebuffer = fd.framebuffer,
        .renderArea = { .offset = {0, 0}, .extent = extent },
        .clearValueCount = 2,
        .pClearValues = clear_values,
    };
    vkCmdBeginRenderPass(cmd, &pass_info, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    const VkViewport viewport = {
        .x = 0.0f,
        .y = float(extent.height),
        .width = float(extent.width),
        .height = -float(extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    const VkRect2D scissor = { .offset = {0, 0}, .extent = extent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Подключаем оба набора дескрипторов
    const VkDescriptorSet sets[] = { scene_set, model_set };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 2, sets, 0, nullptr);

    const VkDeviceSize vertex_offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.handle, &vertex_offset);
    vkCmdBindIndexBuffer(cmd, index_buffer.handle, 0, VK_INDEX_TYPE_UINT16);
    vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

} // namespace application