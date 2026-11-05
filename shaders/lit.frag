#version 450

layout(location = 0) in vec3 frag_world_position;
layout(location = 1) in vec3 frag_normal;
layout(location = 2) in vec2 frag_uv;

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

layout(set = 1, binding = 1) uniform sampler2D albedo_texture;

layout(push_constant) uniform PushConstants {
    uint flags; // бит 0: включить нетривиальное сэмплирование
    float time;
    float emissive_strength;
    float uv_scale;
} pc;

const uint FLAG_NONTRIVIAL_SAMPLING = 1u;

layout(location = 0) out vec4 out_color;

void main() {
    vec3 N = normalize(frag_normal);
    vec3 V = normalize(scene.camera_position.xyz - frag_world_position);
    vec3 L = normalize(-scene.dir_direction.xyz);
    vec3 light = scene.dir_color.rgb * scene.dir_color.w;

    vec3 albedo;
    vec3 emissive = vec3(0.0);

    if ((pc.flags & FLAG_NONTRIVIAL_SAMPLING) != 0u) {
        float t = pc.time;

        vec2 uv_a = frag_uv + vec2(0.05, 0.03) * t;
        vec3 sample_a = texture(albedo_texture, uv_a).rgb;

        float angle = 0.4 * t;
        mat2 rotation = mat2(cos(angle), sin(angle), -sin(angle), cos(angle));
        vec2 uv_b = rotation * (frag_uv - 0.5) * pc.uv_scale + 0.5;
        vec3 sample_b = texture(albedo_texture, uv_b).rgb;

        vec3 modulation = 0.6 + 0.4 * cos(t + vec3(0.0, 2.1, 4.2));

        albedo = sample_a * modulation * object.albedo.rgb;

        float pulse = 0.5 + 0.5 * sin(2.0 * t);
        emissive = sample_b * modulation * pc.emissive_strength * pulse;
    } else {
        albedo = texture(albedo_texture, frag_uv).rgb * object.albedo.rgb;
    }

    // Рассеянный свет
    vec3 ambient = scene.ambient.rgb * scene.ambient.w * albedo;

    // Диффузная составляющая
    float n_dot_l = max(dot(N, L), 0.0);
    vec3 diffuse = light * albedo * n_dot_l;

    // Зеркальная составляющая
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), object.specular.w);
    vec3 specular = light * object.specular.rgb * (n_dot_l > 0.0 ? spec : 0.0);

    out_color = vec4(ambient + diffuse + specular + emissive, 1.0);
}
