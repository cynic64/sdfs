#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable
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

layout (std140, binding = 1) buffer ComputeOut {
	vec3 force;
	vec3 torque;
	vec3 linear_impulse;
	uint collision_count;

	// Idk where else to put this. Surely there is a better way to set some value to 0 before
	// all invocations run?
	uint debug_out_idx;
} out_buf;

layout (std140, binding = 2) buffer DebugIn {
	vec3 line_poss[DEBUG_MAX_LINES];
	vec3 line_dirs[DEBUG_MAX_LINES];
} debug_buf;

layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

#include common.glsl

void main() {
	// Compute intersection between first 2 objects
	float x = gl_GlobalInvocationID.x * 0.025 - 1 + 0.0125;
	float y = gl_GlobalInvocationID.y * 0.025 - 1 + 0.0125;
	float z = gl_GlobalInvocationID.z * 0.025 - 1 + 0.0125;

	int a_type = in_buf.objects[0].type, b_type = in_buf.objects[1].type;
	mat4 a_transform = in_buf.objects[0].transform,
		b_transform = in_buf.objects[1].transform;

	vec3 point = vec3(x, y, z);
	float dist0 = scene_sdf(a_type, a_transform, point);
	float dist1 = scene_sdf(b_type, b_transform, point);

	if (dist0 <= 0 && dist1 <= 0) {
		vec3 my_normal = calc_normal(a_type, a_transform, point);
		vec3 other_normal = calc_normal(b_type, b_transform, point);

		// If there's a collision, write the average of (the opposite of our normal) and
		// (the other normal). Those should both point roughly in the right direction to
		// un-intersect ourselves.

		// For deep penetrations, the two might cancel each other out. Oh well, nothing we
		// can really do.
		//vec3 collision_normal = (-my_normal + other_normal) * 0.5;
		vec3 collision_normal = other_normal;

		if (length(collision_normal) == 0) return;
		collision_normal = normalize(collision_normal);

		// Object's center of mass
		vec3 com = (a_transform * vec4(0, 0, 0, 1)).xyz;

		// Vector from COM to point
		vec3 r = point - com;

		// Cross of force and vector to COM, which is the force's contribution to torque or
		// something
		// man this hurts my brain
		//vec3 torque = cross(r, force);

		// 1 = perfectly elastic, 0 = all momentum absorbed on collision
		float restitution = 1;
		// Eventually these should be passed as part of Object
		float a_mass = 1;
		// Object B can't be moved, so it has infinite mass
		float b_mass = 1 / 0;
		vec3 rel_vel = in_buf.objects[0].linear_vel - in_buf.objects[1].linear_vel;
		float impulse_mag = (-(1 + restitution) * dot(rel_vel, collision_normal))
			/ (dot(collision_normal, collision_normal) * (1.0 / a_mass + 1.0 / b_mass));
		vec3 linear_impulse = impulse_mag * collision_normal;

		// This seems like it should be really slow, but somehow it isn't.
		atomicAdd(out_buf.linear_impulse.x, linear_impulse.x);
		atomicAdd(out_buf.linear_impulse.y, linear_impulse.y);
		atomicAdd(out_buf.linear_impulse.z, linear_impulse.z);
		atomicAdd(out_buf.collision_count, 1);

		// Add a line to debug view, as long as there is space
		uint idx = atomicAdd(out_buf.debug_out_idx, 1);
		if (idx < DEBUG_MAX_LINES) {
			debug_buf.line_poss[idx] = point;
			debug_buf.line_dirs[idx] = vec3(collision_normal);
		}
	}
}
