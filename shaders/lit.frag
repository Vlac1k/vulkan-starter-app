#version 450

layout(location = 0) in vec3 frag_world_position;
layout(location = 1) in vec3 frag_normal;

layout(set = 0, binding = 0, std140) uniform SceneData {
    mat4 view_proj;
    vec4 camera_position;
    vec4 ambient;
    vec4 dir_direction;
    vec4 dir_color;
} scene;

layout(set = 1, binding = 0, std140) uniform ModelData {
    mat4 model;
    mat4 normal_matrix;
    vec4 albedo;
    vec4 specular;
} object;

layout(location = 0) out vec4 out_color;

vec3 shade(vec3 N, vec3 V, vec3 L, vec3 radiance) {
    float n_dot_l = max(dot(N, L), 0.0);
    vec3 diffuse = radiance * object.albedo.rgb * n_dot_l;

    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), object.specular.w);
    vec3 specular = radiance * object.specular.rgb * (n_dot_l > 0.0 ? spec : 0.0);

    return diffuse + specular;
}

void main() {
    vec3 N = normalize(frag_normal);
    vec3 V = normalize(scene.camera_position.xyz - frag_world_position);

    // Рассеянный свет
    vec3 color = scene.ambient.rgb * scene.ambient.w * object.albedo.rgb;

    // Направленный источник
    vec3 L = normalize(-scene.dir_direction.xyz);
    color += shade(N, V, L, scene.dir_color.rgb * scene.dir_color.w);

    out_color = vec4(color, 1.0);
}