#version 450

layout(std140, binding = 0) buffer InData {
	int count;
	int types[512];
	mat4 transforms[512];
} in_buf;

layout(std140, binding = 1) buffer OutData {
	int count;
	int types[512];
	mat4 transforms[512];
} out_buf;

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main() {
	for (int i = 0; i < in_buf.count; i++) {
		out_buf.count = in_buf.count;
		out_buf.types[i] = in_buf.types[i];
		out_buf.transforms[i] = in_buf.transforms[i];
	}
}
