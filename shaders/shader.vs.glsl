#version 450

layout (push_constant, std140) uniform PushConstants {
        vec4 forward;
        vec4 eye;
        vec4 dir;
        vec4 aspect;
        vec4 exp;
} constants;

vec2 positions[6] = vec2[](
        vec2(-1.0, -1.0),
        vec2(1.0, -1.0),
        vec2(1.0, 1.0),
        vec2(-1.0, -1.0),
        vec2(1.0, 1.0),
        vec2(-1.0, 1.0)
);

layout (location = 0) out vec3 out_dir;

void main() {
        vec2 pos = positions[gl_VertexIndex];
	gl_Position = vec4(pos, 0.0, 1.0);

	vec3 right = normalize(cross(constants.forward.xyz, vec3(0, -1, 0))) * constants.aspect.x;
	vec3 up = normalize(cross(constants.forward.xyz, right));
	out_dir = normalize(pos.x * right + -pos.y * up + constants.forward.xyz * 1.5);
}
