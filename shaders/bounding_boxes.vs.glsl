#version 450

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

layout (std140, set = 0, binding = 0) uniform Uniform {
	int box_count;
	vec4 box_poss[256];
} uni;

vec2 positions[6] = vec2[](
        vec2(-0.02, -0.02),
        vec2(0.02, -0.02),
        vec2(0.02, 0.02),
        vec2(-0.02, -0.02),
        vec2(0.02, 0.02),
        vec2(-0.02, 0.02)
);


void main() {
        vec2 offset = positions[gl_VertexIndex % 6];

	vec4 pos_worldspace = vec4(uni.box_poss[gl_VertexIndex / 6].xyz, 1);
	vec4 pos_screenspace = constants.proj * constants.view * pos_worldspace;
	vec3 thing = pos_screenspace.xyz / pos_screenspace.w;
	thing.xy += offset;

	gl_Position = vec4(thing, 1);
}
