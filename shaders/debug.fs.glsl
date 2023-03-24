#version 450

layout (location = 0) in vec3 in_dir;

layout (location = 0) out vec4 out_color;

void main() {
	out_color = vec4(normalize(in_dir) * 0.5 + 0.5, 1);
}
