#version 450
#extension GL_EXT_shader_atomic_float : enable

#include constants.glsl

struct Object {
	int type;
	mat4 transform;

	vec3 pos;
	mat4 orientation;
	vec3 linear_vel;
	vec3 angular_vel;
};

layout (std140, binding = 0) buffer readonly Scene {
	int count;
	Object objects[];
} in_buf;

layout (std140, binding = 1) buffer DebugOut {
	int idk;
} out_buf;

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

#include common.glsl

void main() {
	out_buf.idk = 99;
}
