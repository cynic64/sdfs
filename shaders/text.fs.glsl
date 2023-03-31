#version 450

#include constants.glsl

#define TOTAL_CHARS 95
#define START_CHAR 32 // ' '

layout (set = 0, binding = 0) uniform sampler2D tex_sampler;
layout (set = 0, binding = 1) uniform Text {
	mat4 transform;
	int chars[TEXT_MAX_CHARS];
	int char_count;
} in_text;

layout (location = 0) in vec2 in_pos;

layout (location = 0) out vec4 out_color;

void main() {
	vec2 p = in_pos;
	p.y += in_text.chars[int(p.x)] - START_CHAR;
	p.y /= TOTAL_CHARS;
	p.x = mod(p.x, 1);
	out_color = texture(tex_sampler, p);
	if (out_color.r == 0) discard;
}
