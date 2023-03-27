#version 450

#include constants.glsl

layout (push_constant, std140) uniform PushConstants {
       vec4 iResolution;
       vec4 iMouse;
       float iFrame;
       float iTime;
       vec3 forward;
       vec3 eye;
       vec3 dir;
       mat4 view;
       mat4 proj;
} constants;

struct Object {
	int type;
	mat4 transform;

	vec3 pos;
	mat4 orientation;
	vec3 linear_vel;
	vec3 angular_vel;
};

layout(std140, binding = 0) buffer Scene {
	int count;
	Object objects[];
} scene;

#include common.glsl

struct RayShot {
	vec3 intersect;
	float closest;
	bool hit;
	float depth;
	int steps;
};

layout (location = 0) flat in int obj_idx;
// Should be normalized
layout (location = 1) in vec3 pos_worldspace;

layout (location = 0) out vec4 out_color;

RayShot raymarch(vec3 ray_origin, vec3 ray_dir, float initial_depth) {
	float threshold = 0.001;
	float depth = initial_depth;
	float closest = 99999999;
	for (int i = 0; i < 64; i++) {
		vec3 point = ray_origin + ray_dir * depth;
		float dist = scene_sdf(scene.objects[obj_idx].type,
				       scene.objects[obj_idx].transform, point);
		closest = min(closest, dist);
		if (dist < threshold) {
			RayShot shot;
			shot.intersect = point;
			shot.depth = depth;
			shot.hit = true;
			shot.steps = i;
			shot.closest = closest;
			return shot;
		}
		depth += dist;
	}

	RayShot shot;
	shot.hit = false;
	shot.depth = 999999999;
	shot.steps = 64;
	shot.closest = closest;
	return shot;
}

void main()
{
	vec3 ray_origin = constants.eye.xyz;
	vec3 ray_dir = normalize(pos_worldspace - ray_origin);

	// Raycast
	RayShot shot = raymarch(ray_origin, ray_dir, length(pos_worldspace - ray_origin));
	if (shot.hit) {
		out_color = vec4((calc_normal(scene.objects[obj_idx].type,
					      scene.objects[obj_idx].transform,
					      shot.intersect,
					      shot.depth / constants.iResolution.x)
				  * 0.5 + 0.5), 1);
		gl_FragDepth = clamp(shot.depth / 100, 0, 1);
	}  else {
		out_color = vec4(vec3(scene.objects[obj_idx].type * 0.2) + 0.1, 1);
		// Use 0.9 if you want to see bounding boxes
		gl_FragDepth = 1;
	}
}
