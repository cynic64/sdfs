#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_EXT_shader_atomic_float : enable

#include constants.glsl

struct Object {
	int type;
	mat4 transform;
};

layout(std140, binding = 0) buffer SceneIn {
	int count;
	Object objects[];
} in_buf;

layout(std140, binding = 1) buffer ComputeOut {
	vec3 total_force;
	vec3 total_torque;
} out_buf;

layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

#include common.glsl

void main() {
	// Compute intersection between first 2 objects
	float x = gl_GlobalInvocationID.x * 0.025 - 1 + 0.0125;
	float y = gl_GlobalInvocationID.y * 0.025 - 1 + 0.0125;
	float z = gl_GlobalInvocationID.z * 0.025 - 1 + 0.0125;

	vec3 point = vec3(x, y, z);
	float dist0 = scene_sdf(in_buf.objects[0].type, in_buf.objects[0].transform, point);
	float dist1 = scene_sdf(in_buf.objects[1].type, in_buf.objects[1].transform, point);

	if (dist0 <= 0 && dist1 <= 0) {
		vec3 my_normal = calc_normal(in_buf.objects[0].type,
							     in_buf.objects[0].transform, point);
		vec3 other_normal = calc_normal(in_buf.objects[1].type,
							     in_buf.objects[1].transform, point);

		// If there's a collision, write the average of (the opposite of our normal) and
		// (the other normal). Those should both point roughly in the right direction to
		// un-intersect ourselves.

		// For deep penetrations, the two might cancel each other out. Oh well, nothing we
		// can really do.
		vec3 force = (-my_normal + other_normal) * 0.5;
		force.z = 0;
		if (length(force.xy) > 0) force.xy = normalize(force.xy);

		// Object's center of mass
		vec3 com = (in_buf.objects[0].transform * vec4(0, 0, 0, 1)).xyz;

		// Vector from COM to point
		vec3 r = point - com;

		// Cross of force and vector to COM, which is the force's contribution to torque or
		// something
		// man this hurts my brain
		vec3 torque = cross(r, force);

		// This seems like it should be really slow, but somehow it isn't.
		atomicAdd(out_buf.total_force.x, force.x);
		atomicAdd(out_buf.total_force.y, force.y);
		atomicAdd(out_buf.total_force.z, force.z);
		atomicAdd(out_buf.total_torque.x, torque.x);
		atomicAdd(out_buf.total_torque.y, torque.y);
		atomicAdd(out_buf.total_torque.z, torque.z);
	}
}
