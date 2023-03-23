#version 450

#include constants.glsl

layout (push_constant, std140) uniform PushConstants {
       vec4 iResolution;
       vec4 iMouse;
       float iFrame;
       float iTime;
       vec4 forward;
       vec4 eye;
       vec4 dir;
       mat4 view;
       mat4 proj;
} constants;

struct Object {
	int type;
	mat4 transform;
};

layout(std140, binding = 0) buffer Scene {
	int count;
	Object objects[];
} scene;

#include common.glsl

struct RayShot {
	vec3 closest;
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
	for (int i = 0; i < 128; i++) {
		vec3 point = ray_origin + ray_dir * depth;
		float dist = scene_sdf(scene.objects[obj_idx].type,
				       scene.objects[obj_idx].transform, point);
		if (dist < threshold) {
			RayShot shot;
			shot.closest = point;
			shot.depth = depth;
			shot.hit = true;
			shot.steps = i;
			return shot;
		}
		depth += dist;
	}

	RayShot shot;
	shot.hit = false;
	shot.depth = 999999999;
	shot.steps = 128;
	return shot;
}

// Taken from iquilezles.org/articles/normalsSDF/
// Messing with this is bad juju
vec3 calc_normal(int type, mat4 transform, vec3 point) {
    const float h = 0.0002;
    const vec2 k = vec2(1,-1);
    return normalize(k.xyy*scene_sdf(type, transform, point + k.xyy*h)
		     + k.yyx*scene_sdf(type, transform, point + k.yyx*h)
		     + k.yxy*scene_sdf(type, transform, point + k.yxy*h)
		     + k.xxx*scene_sdf(type, transform, point + k.xxx*h));
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
					      shot.closest) * 0.5 + 0.5), 1);
		gl_FragDepth = clamp(shot.depth / 100, 0, 1);
	} else {
		out_color = vec4(vec3(scene.objects[obj_idx].type * 0.2) + 0.1, 1);
		gl_FragDepth = 0.9;
	}
}
