#version 450

// Атрибуты вершины
layout(location = 0) in vec3 in_position; // позиция в локальных координатах модели
layout(location = 1) in vec3 in_normal; // нормаль грани в локальных координатах

// Набор дескрипторов 0: данные сцены
layout(set = 0, binding = 0, std140) uniform SceneData {
    mat4 view_proj;
    vec4 camera_position;
    vec4 ambient;
    vec4 dir_direction;
    vec4 dir_color;
} scene;

// Набор дескрипторов 1: данные модели
layout(set = 1, binding = 0, std140) uniform ModelData {
    mat4 model;
    mat4 normal_matrix;
    vec4 albedo;
    vec4 specular;
} object;

layout(location = 0) out vec3 frag_world_position;
layout(location = 1) out vec3 frag_normal;

void main() {
    // Позиция вершины в мировых координатах
    vec4 world_position = object.model * vec4(in_position, 1.0);
    frag_world_position = world_position.xyz;

    frag_normal = mat3(object.normal_matrix) * in_normal;

    gl_Position = scene.view_proj * world_position;
}
