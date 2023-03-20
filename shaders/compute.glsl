#version 450

layout(std140, binding = 0) readonly buffer stuff {
	int x;
};

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
	x + 1;
}
