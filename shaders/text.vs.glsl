#version 450

#include constants.glsl

layout (set = 0, binding = 1) uniform Text {
	mat4 transform;
	// We use a whopping 16x more space than necessary like this. But whatever, the alternative
	// (write to texture and texelFetch) is a pain and not worth it for now.
	int chars[TEXT_MAX_CHARS];
	int char_count;
} in_text;

layout (location = 0) out vec2 out_pos;

vec2 vertices[6] = vec2[](
			  vec2(-1.0f, 1.0f),
			  vec2(1.0f, 1.0f),
			  vec2(1.0f, -1.0f),
			  vec2(-1.0f, 1.0f),
			  vec2(1.0f, -1.0f),
			  vec2(-1.0f, -1.0f)
			  );

void main() {
	vec2 p = vertices[gl_VertexIndex];
	gl_Position = in_text.transform * vec4(p, 0, 1);

	out_pos = vec2((p.x * 0.5 + 0.5) * in_text.char_count, p.y * 0.5 + 0.5);
}
