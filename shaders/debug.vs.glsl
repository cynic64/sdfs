#version 450

#include constants.glsl

layout (push_constant, std140) uniform PushConstants {
       vec4 iResolution;
       vec4 iMouse;
       float iFrame;
       float iTime;
       vec4 forward;
       vec4 eye;
       vec4 dir;
       mat4 view;
       mat4 proj;
} constants;


layout(std140, binding = 0) buffer Debug {
	vec4 line_poss[DEBUG_MAX_LINES];
	vec4 line_dirs[DEBUG_MAX_LINES];
} data;

layout (location = 0) out vec3 color;

void main() {
	vec3 pos = data.line_poss[gl_InstanceIndex].xyz;
	vec3 dir = data.line_dirs[gl_InstanceIndex].xyz;
	vec3 vertices[] = vec3[](pos, pos + dir * 0.2);
	vec3 colors[] = vec3[](vec3(0), dir * 0.5 + 0.5);
	vec3 pos_worldspace = vertices[gl_VertexIndex];

	color = colors[gl_VertexIndex];

	gl_Position = constants.proj * constants.view * vec4(pos_worldspace, 1);
}
