#version 450

#include constants.glsl

layout(std140, binding = 0) buffer InData {
	int count;
	int types[MAX_OBJ_COUNT];
	mat4 transforms[MAX_OBJ_COUNT];
} in_buf;

layout(std140, binding = 1) buffer OutData {
	int count;
	int types[MAX_OBJ_COUNT];
	mat4 transforms[MAX_OBJ_COUNT];
} out_buf;

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

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
	/*
	for (int i = 0; i < in_buf.count; i++) {
		out_buf.types[i] = in_buf.types[i];
		out_buf.transforms[i] = in_buf.transforms[i];
	}
	out_buf.count = in_buf.count;
	*/
	out_buf.count = 0;

	// Compute intersection between first 2 objects
	for (float x = -2; x < 2; x += 0.1) {
		for (float y = -2; y < 2; y += 0.1) {
			for (float z = -2; z < 2; z += 0.1) {
				vec3 point = vec3(x, y, z);
				float dist0 = scene_sdf(in_buf.types[0], in_buf.transforms[0], point);
				float dist1 = scene_sdf(in_buf.types[1], in_buf.transforms[1], point);

				if (dist0 <= 0 && dist1 <= 0 && out_buf.count + 1 < MAX_OBJ_COUNT) {
					out_buf.types[out_buf.count] = 1;
					out_buf.transforms[out_buf.count] =
						translation(point - vec3(5, 0, 0)) * scale(0.03);
					out_buf.count++;
				}
			}
		}
	}
}
