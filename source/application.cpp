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

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace application {

using graphics::internal::context;

namespace {
// Типы данных
// Вершина
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};
static_assert(sizeof(Vertex) == 8 * sizeof(float), "Vertex must be tightly packed");

// Данные сцены
struct SceneData {
    glm::mat4 view_proj;
    glm::vec4 camera_position;
    glm::vec4 ambient;
    glm::vec4 dir_direction;
    glm::vec4 dir_color;
};
static_assert(sizeof(SceneData) == 128, "SceneData must match the shader (std140)");

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

// Диапазон индексов одной модели внутри общего индексного буфера
struct Mesh {
    uint32_t first_index = 0;
    uint32_t index_count = 0;
};

struct PushConstants {
    uint32_t flags;
    float time;
    float emissive_strength;
    float uv_scale;
};
static_assert(sizeof(PushConstants) == 16, "PushConstants must match the shader");

struct DrawItem {
    Mesh mesh;
    ModelData data;
    uint32_t texture = 0;
    PushConstants push{};
};

// Текстура
struct Texture {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VkImageView view = VK_NULL_HANDLE;
};

struct Settings {
    // Камера
    glm::vec3 camera_position{0.0f, 1.5f, 5.0f};
    float camera_yaw = 0.0f;
    float camera_pitch = -15.0f;
    float camera_fov = 60.0f;
    int projection = 0;
    float ortho_half_height = 2.3f;

    glm::vec3 light_direction{-0.4f, -0.7f, -0.6f};
    glm::vec3 light_color{1.0f};
    float light_intensity = 1.0f;
    glm::vec3 ambient_color{1.0f};
    float ambient_intensity = 0.1f;

    glm::vec3 albedo{1.0f, 1.0f, 1.0f};
    glm::vec3 specular_color{1.0f};
    float shininess = 32.0f;

    int tetra_texture = 3;
    int ground_texture = 0;
    bool effect_enabled = true;
    float emissive_strength = 1.0f;
    float effect_uv_scale = 1.7f;

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
std::vector<DrawItem> draw_list;

double animation_time = 0.0;
double last_time = -1.0;

VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout model_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet scene_set = VK_NULL_HANDLE;

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

Buffer vertex_buffer;
Buffer index_buffer;
Buffer scene_buffer;
Buffer model_buffer;

constexpr uint32_t max_objects = 16;
constexpr uint32_t floor_tiles = 3;
VkDeviceSize model_stride = 0;

Mesh tetra_mesh;
Mesh floor_mesh;

VkCommandPool transfer_pool = VK_NULL_HANDLE;

const char* const texture_paths[] = {
    "textures/checker.png",
    "textures/bricks.png",
    "textures/wood.png",
    "textures/blotches.png",
};
const char* const texture_names[] = { "Checker", "Bricks", "Wood", "Blotches" };
constexpr uint32_t texture_count = sizeof(texture_paths) / sizeof(texture_paths[0]);
std::vector<Texture> textures;
std::vector<VkDescriptorSet> material_sets;

VkSampler albedo_sampler = VK_NULL_HANDLE;

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

bool createTransferPool() {
    const VkCommandPoolCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = context.graphics_queue_index,
    };

    if (vkCreateCommandPool(context.device, &info, nullptr, &transfer_pool) != VK_SUCCESS) {
        std::cerr << "Failed to create transfer command pool\n";
        return false;
    }
    return true;
}

// Начинает запись разового командного буфера
VkCommandBuffer beginOneTimeCommands() {
    const VkCommandBufferAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = transfer_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(context.device, &allocate_info, &cmd);

    const VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &begin_info);
    return cmd;
}

// Заканчивает запись, отправляет соманды на GPU и ждет их выполнения
void endOneTimeCommands(VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);

    const VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence = VK_NULL_HANDLE;
    vkCreateFence(context.device, &fence_info, nullptr, &fence);

    const VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };
    vkQueueSubmit(context.graphics_queue, 1, &submit, fence);
    vkWaitForFences(context.device, 1, &fence, VK_TRUE, UINT64_MAX);

    vkDestroyFence(context.device, fence, nullptr);
    vkFreeCommandBuffers(context.device, transfer_pool, 1, &cmd);
}

// Буфер в VRAM
bool createDeviceBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
    const VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    const VmaAllocationCreateInfo allocation_info = {
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };

    if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info, &out.handle, &out.allocation, nullptr) != VK_SUCCESS) {
        std::cerr << "Failed to create device-local Vulkan buffer\n";
        return false;
    }

    Buffer staging;
    if (!createBuffer(data, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging)) {
        destroyBuffer(staging);
        return false;
    }

    VkCommandBuffer cmd = beginOneTimeCommands();
    const VkBufferCopy region = { .srcOffset = 0, .dstOffset = 0, .size = size };
    vkCmdCopyBuffer(cmd, staging.handle, out.handle, 1, &region);
    endOneTimeCommands(cmd);

    destroyBuffer(staging);
    return true;
}

// Загружает изображение из файла и создает из него текстуру в VRAM
bool createTexture(const char* path, Texture& out) {
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* pixels = stbi_load(path, &width, &height, &channels, STBI_rgb_alpha);
    if (pixels == nullptr) {
        std::cerr << "Failed to load image " << path << ": " << stbi_failure_reason() << '\n';
        return false;
    }
    const VkDeviceSize size = VkDeviceSize(width) * VkDeviceSize(height) * 4;

    // Изображение в VRAM
    const VkImageCreateInfo image_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_SRGB,
        .extent = { uint32_t(width), uint32_t(height), 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    const VmaAllocationCreateInfo allocation_info = {
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };

    if (vmaCreateImage(context.allocator, &image_info, &allocation_info, &out.image, &out.allocation, nullptr) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan image for " << path << '\n';
        stbi_image_free(pixels);
        return false;
    }

    // Пиксели копируем в staging буфер
    Buffer staging;
    const bool staged = createBuffer(pixels, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging);
    stbi_image_free(pixels);
    if (!staged) {
        vmaDestroyImage(context.allocator, out.image, out.allocation);
        out.image = VK_NULL_HANDLE;
        out.allocation = nullptr;
        return false;
    }

    const VkImageSubresourceRange range = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };

    VkCommandBuffer cmd = beginOneTimeCommands();

    const VkImageMemoryBarrier to_transfer = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = out.image,
        .subresourceRange = range,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_transfer);

    const VkBufferImageCopy region = {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .imageOffset = { 0, 0, 0 },
        .imageExtent = { uint32_t(width), uint32_t(height), 1 },
    };
    vkCmdCopyBufferToImage(cmd, staging.handle, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    const VkImageMemoryBarrier to_shader = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = out.image,
        .subresourceRange = range,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_shader);

    endOneTimeCommands(cmd);
    destroyBuffer(staging);

    const VkImageViewCreateInfo view_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = out.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_SRGB,
        .subresourceRange = range,
    };

    if (vkCreateImageView(context.device, &view_info, nullptr, &out.view) != VK_SUCCESS) {
        std::cerr << "Failed to create image view for " << path << '\n';
        return false;
    }
    return true;
}

// Правила чтения текстуры
bool createAlbedoSampler() {
    const VkSamplerCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .minLod = 0.0f,
        .maxLod = 0.0f,
        .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
    };

    if (vkCreateSampler(context.device, &info, nullptr, &albedo_sampler) != VK_SUCCESS) {
        std::cerr << "Failed to create sampler\n";
        return false;
    }
    return true;
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

    const glm::vec3 axis_u = glm::normalize(p[1] - p[0]);
    const glm::vec3 axis_v = glm::cross(normal, axis_u);
    const float uv_scale = 2.0f * glm::length(p[1] - p[0]);

    const int base = int(vertices.size());
    for (int i = 0; i < count; ++i) {
        const glm::vec3 offset = p[i] - centroid;
        const glm::vec2 uv(glm::dot(offset, axis_u) / uv_scale + 0.5f, glm::dot(offset, axis_v) / uv_scale + 0.5f);
        vertices.push_back(Vertex{p[i], normal, uv});
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

// Набор модели
VkDescriptorSetLayout createModelSetLayout() {
    const VkDescriptorSetLayoutBinding bindings[] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
    };

    const VkDescriptorSetLayoutCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2,
        .pBindings = bindings,
    };

    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(context.device, &info, nullptr, &layout) != VK_SUCCESS) {
        std::cerr << "Failed to create model descriptor set layout\n";
        return VK_NULL_HANDLE;
    }
    return layout;
}

bool createDescriptors() {
    scene_set_layout = createUniformSetLayout();
    model_set_layout = createModelSetLayout();
    if (scene_set_layout == VK_NULL_HANDLE || model_set_layout == VK_NULL_HANDLE) {
        return false;
    }

    const VkDescriptorPoolSize pool_sizes[] = {
        { .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1 },
        { .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = texture_count },
        { .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = texture_count },
    };

    const VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1 + texture_count,
        .poolSizeCount = 3,
        .pPoolSizes = pool_sizes,
    };

    if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor pool\n";
        return false;
    }

    // Набор сцены
    const VkDescriptorSetAllocateInfo scene_allocate = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &scene_set_layout,
    };
    if (vkAllocateDescriptorSets(context.device, &scene_allocate, &scene_set) != VK_SUCCESS) {
        std::cerr << "Failed to allocate scene descriptor set\n";
        return false;
    }

    // Наборы материалов
    const std::vector<VkDescriptorSetLayout> material_layouts(texture_count, model_set_layout);
    material_sets.resize(texture_count);

    const VkDescriptorSetAllocateInfo material_allocate = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = texture_count,
        .pSetLayouts = material_layouts.data(),
    };
    if (vkAllocateDescriptorSets(context.device, &material_allocate, material_sets.data()) != VK_SUCCESS) {
        std::cerr << "Failed to allocate material descriptor sets\n";
        return false;
    }

    // Записываем в наборы реальные ресурсы
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

    std::vector<VkDescriptorImageInfo> image_infos(texture_count);
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(1 + 2 * size_t(texture_count));

    writes.push_back(VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = scene_set,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &scene_info,
    });

    for (uint32_t i = 0; i < texture_count; ++i) {
        image_infos[i] = VkDescriptorImageInfo{
            .sampler = albedo_sampler,
            .imageView = textures[i].view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };

        writes.push_back(VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = material_sets[i],
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .pBufferInfo = &model_info,
        });

        writes.push_back(VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = material_sets[i],
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &image_infos[i],
        });
    }

    vkUpdateDescriptorSets(context.device, uint32_t(writes.size()), writes.data(), 0, nullptr);

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

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0,
        .size = sizeof(PushConstants),
    };

    const VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 2,
        .pSetLayouts = set_layouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
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
        {
            .location = 2,
            .binding = 0,
            .format = VK_FORMAT_R32G32_SFLOAT,
            .offset = offsetof(Vertex, uv),
        },
    };

    const VkPipelineVertexInputStateCreateInfo vertex_input = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = 3,
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
static bool initializeResources() {
    if (!createTransferPool()) {
        return false;
    }

    // Геометрия: фигура и пол лежат в общих вершинном и индексном буферах.
    std::vector<Vertex> vertices;
    std::vector<uint16_t> indices;

    for (const auto& face : triangle_faces) {
        addFace(face, 3, vertices, indices);
    }
    for (const auto& face : hexagon_faces) {
        addFace(face, 6, vertices, indices);
    }
    tetra_mesh = Mesh{0, uint32_t(indices.size())};

    // Пол: квадрат 2x2 в плоскости XZ
    const uint16_t floor_base = uint16_t(vertices.size());
    const float tiles = float(floor_tiles);
    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    vertices.push_back(Vertex{{-1.0f, 0.0f, -1.0f}, up, {0.0f, 0.0f}});
    vertices.push_back(Vertex{{ 1.0f, 0.0f, -1.0f}, up, {tiles, 0.0f}});
    vertices.push_back(Vertex{{ 1.0f, 0.0f,  1.0f}, up, {tiles, tiles}});
    vertices.push_back(Vertex{{-1.0f, 0.0f,  1.0f}, up, {0.0f, tiles}});

    floor_mesh = Mesh{uint32_t(indices.size()), 6};
    const uint16_t floor_indices[] = {0, 1, 2, 0, 2, 3};
    for (const uint16_t i : floor_indices) {
        indices.push_back(uint16_t(floor_base + i));
    }

    // Вершинный и индексный буферы в VRAM
    if (!createDeviceBuffer(vertices.data(), vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer)) {
        return false;
    }

    if (!createDeviceBuffer(indices.data(), indices.size() * sizeof(uint16_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer)) {
        return false;
    }

    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(context.physical_device, &properties);
    const VkDeviceSize alignment = properties.limits.minUniformBufferOffsetAlignment;
    model_stride = (VkDeviceSize(sizeof(ModelData)) + alignment - 1) / alignment * alignment;

    const std::vector<uint8_t> zeros(size_t(model_stride) * max_objects);
    if (!createBuffer(zeros.data(), zeros.size(), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, model_buffer)) {
        return false;
    }

    if (!createBuffer(&gpu_scene, sizeof(SceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, scene_buffer)) {
        return false;
    }

    textures.resize(texture_count);
    for (uint32_t i = 0; i < texture_count; i++) {
        if (!createTexture(texture_paths[i], textures[i])) {
            return false;
        }
    }

    if (!createAlbedoSampler()) {
        return false;
    }

    return createDescriptors() && createPipeline();
}

bool initialize() {
    if (!initializeResources()) {
        shutdown();
        return false;
    }
    return true;
}

void shutdown() {
    vkQueueWaitIdle(context.graphics_queue);

    vkDestroyPipeline(context.device, pipeline, nullptr);
    vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);

    vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
    vkDestroyDescriptorSetLayout(context.device, model_set_layout, nullptr);
    vkDestroyDescriptorSetLayout(context.device, scene_set_layout, nullptr);

    vkDestroySampler(context.device, albedo_sampler, nullptr);

    for (Texture& texture : textures) {
        vkDestroyImageView(context.device, texture.view, nullptr);
        vmaDestroyImage(context.allocator, texture.image, texture.allocation);
    }

    vkDestroyCommandPool(context.device, transfer_pool, nullptr);

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

    if (ImGui::CollapsingHeader("Textures", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Combo("Tetrahedron texture", &s.tetra_texture, texture_names, int(texture_count));
        ImGui::Combo("Ground texture", &s.ground_texture, texture_names, int(texture_count));

        ImGui::Separator();
        ImGui::Checkbox("Non-trivial sampling (tetrahedron)", &s.effect_enabled);
        ImGui::SliderFloat("Emissive strength", &s.emissive_strength, 0.0f, 3.0f);
        ImGui::SliderFloat("Second sample UV scale", &s.effect_uv_scale, 0.5f, 4.0f);
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

    draw_list.clear();

    ModelData tetra{};
    tetra.model = model;
    tetra.normal_matrix = glm::transpose(glm::inverse(model));
    tetra.albedo = glm::vec4(s.albedo, 1.0f);
    tetra.specular = glm::vec4(s.specular_color, s.shininess);
    
    DrawItem tetra_item{tetra_mesh, tetra};
    tetra_item.texture = uint32_t(s.tetra_texture);
    tetra_item.push = PushConstants{
        s.effect_enabled ? 1u : 0u,
        float(time),
        s.emissive_strength,
        s.effect_uv_scale,
    };
    draw_list.push_back(tetra_item);

    // Пол: квадрат 2x2, растянутый в 6 раз и опущенный под фигуру
    glm::mat4 ground_model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1.6f, 0.0f));
    ground_model = glm::scale(ground_model, glm::vec3(6.0f));

    ModelData ground{};
    ground.model = ground_model;
    ground.normal_matrix = glm::transpose(glm::inverse(ground_model));
    ground.albedo = glm::vec4(1.0f);
    ground.specular = glm::vec4(0.2f, 0.2f, 0.2f, 16.0f);
    
    DrawItem ground_item{floor_mesh, ground};
    ground_item.texture = uint32_t(s.ground_texture);
    draw_list.push_back(ground_item);

}

void render(const graphics::internal::FrameData& fd) {
    const VkCommandBuffer cmd = fd.command_buffer;
    const VkExtent2D extent = context.swapchain_extent;

    // Загружаем данные в uniform-буферы
    vmaCopyMemoryToAllocation(context.allocator, &gpu_scene, scene_buffer.allocation, 0, sizeof(SceneData));
    
    // Каждая модель записывается в свою строку динамического uniform buffer.
    for (size_t i = 0; i < draw_list.size(); ++i) {
        vmaCopyMemoryToAllocation(context.allocator, &draw_list[i].data, model_buffer.allocation, VkDeviceSize(i) * model_stride, sizeof(ModelData));
      }

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

    const VkDeviceSize vertex_offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.handle, &vertex_offset);
    vkCmdBindIndexBuffer(cmd, index_buffer.handle, 0, VK_INDEX_TYPE_UINT16);

    for (size_t i = 0; i < draw_list.size(); ++i) {
        const DrawItem& item = draw_list[i];

        const VkDescriptorSet sets[] = { scene_set, material_sets[item.texture] };
        const uint32_t dynamic_offset = uint32_t(VkDeviceSize(i) * model_stride);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 2, sets, 1, &dynamic_offset);

        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &item.push);

        vkCmdDrawIndexed(cmd, item.mesh.index_count, 1, item.mesh.first_index, 0, 0);
    }

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

} // namespace application