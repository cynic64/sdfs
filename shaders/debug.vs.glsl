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

vec2 vertices[3] = vec2[](
			  vec2(-1, 1),
			  vec2(1, 1),
			  vec2(1, -1));

void main() {
	gl_Position = vec4(vertices[gl_VertexIndex], 0, 1);
}
