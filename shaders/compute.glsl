#version 450

layout(std140, binding = 0) buffer Stuff {
	int x;
};

layout(std140, binding = 1) buffer Stuff2 {
	int y;
};

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main() {
	x++;
}
