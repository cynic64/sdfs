#version 450

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
	vec3 forces[40*40*40];
	vec3 torques[40*40*40];
} out_buf;

layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

#include common.glsl

mat4 translation(vec3 offset) {
	return mat4(1, 0, 0, 0,
		    0, 1, 0, 0,
		    0, 0, 1, 0,
		    offset, 1);
}

mat4 scale(float s) {
	return mat4(s, 0, 0, 0,
		    0, s, 0, 0,
		    0, 0, s, 0,
		    0, 0, 0, 1);
}

// Makes 0 1 0 point in `normal`. `normal` should be normalized.
mat4 normal_rotation(vec3 normal) {
	// Get two vectors perpendicular to normal
	vec3 perp1;
	if (normal.x < normal.y) {
		perp1 = normal.x < normal.z ? vec3(0, -normal.z, normal.y)
			: vec3(-normal.y, normal.x, 0);
	} else {
		perp1 = normal.y < normal.z ? vec3(normal.z, 0, -normal.x)
			: vec3(normal.z, 0, -normal.x);
	}
	perp1 = normalize(perp1);
	vec3 perp2 = normalize(cross(normal, perp1));

	// The first column is where 1 0 0 ends up. We don't really care as long as its
	// perpendicular to `normal`. The second column is where 0 1 0 ends up, which should be
	// normal. The third column we also don't really care about.
	return mat4(perp1.x, normal.x, perp2.x, 0,
	            perp1.y, normal.y, perp2.y, 0,
	            perp1.z, normal.z, perp2.z, 0,
		    0,       0,        0,       1);
}

void main() {
	// Compute intersection between first 2 objects
	float x = gl_GlobalInvocationID.x * 0.1 - 2 + 0.05;
	float y = gl_GlobalInvocationID.y * 0.1 - 2 + 0.05;
	float z = gl_GlobalInvocationID.z * 0.1 - 2 + 0.05;

	vec3 point = vec3(x, y, z);
	float dist0 = scene_sdf(in_buf.objects[0].type, in_buf.objects[0].transform, point);
	float dist1 = scene_sdf(in_buf.objects[1].type, in_buf.objects[1].transform, point);

	uint index = gl_GlobalInvocationID.x*40*40
		+ gl_GlobalInvocationID.y*40
		+ gl_GlobalInvocationID.z;
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
		
		out_buf.forces[index] = force;
		out_buf.torques[index] = torque;
	} else {
		out_buf.forces[index] = vec3(0);
		out_buf.torques[index] = vec3(0);
	}
}
