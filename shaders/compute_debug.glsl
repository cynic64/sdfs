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
	vec3 a_com;
	vec3 b_com;
	vec3 a_linear_vel;
	vec3 b_linear_vel;
	vec3 a_angular_vel;
	vec3 b_angular_vel;
} out_buf;

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

#include common.glsl

void main() {
	Object a = in_buf.objects[0], b = in_buf.objects[1];

	// Objects' center of mass
	out_buf.a_com = (a.transform * vec4(0, 0, 0, 1)).xyz;
	out_buf.b_com = (b.transform * vec4(0, 0, 0, 1)).xyz;

	// Objects' velocities
	out_buf.a_linear_vel = a.linear_vel;
	out_buf.b_linear_vel = b.linear_vel;
	out_buf.a_angular_vel = a.angular_vel;
	out_buf.b_angular_vel = b.angular_vel;
}
