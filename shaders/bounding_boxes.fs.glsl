#version 450

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

layout (std140, set = 0, binding = 0) uniform Uniform {
	int count;
	vec4 poss[256];
	int types[256];
} objects;

struct RayShot {
	vec3 closest;
	bool hit;
	float depth;
};

layout (location = 0) flat in int obj_idx;
// Should be normalized
layout (location = 1) in vec3 ray_dir;

layout (location = 0) out vec4 out_color;

float sd_box(vec3 point, vec3 box_size) {
	vec3 d = abs(point) - box_size;
	return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

float sd_sphere(vec3 point, float radius) {
	return length(point) - radius;
}

float scene_sdf(vec3 point) {
	float min_dist = 99999999;
	int type = objects.types[obj_idx];
	float dist;
	if (type == 0) {
		dist = sd_sphere(point - objects.poss[obj_idx].xyz, 2);
	} else {
		dist = sd_box(point - objects.poss[obj_idx].xyz, vec3(2));
	}

	if (dist < min_dist) min_dist = dist;

	return min_dist;
}

RayShot raymarch(vec3 ray_origin, vec3 ray_dir) {
	float threshold = 0.005;
	float depth = 0;
	for (int i = 0; i < 128; i++) {
		vec3 point = ray_origin + ray_dir * depth;
		float dist = scene_sdf(point);
		if (dist < threshold) {
			RayShot shot;
			shot.closest = point;
			shot.depth = depth;
			shot.hit = true;
			return shot;
		}
		depth += dist;
	}

	RayShot shot;
	shot.hit = false;
	shot.depth = 999999999;
	return shot;
}

// Taken from iquilezles.org/articles/normalsSDF/
// Messing with this is bad juju
vec3 calc_normal(in vec3 point) {
    const float h = 0.0002;
    const vec2 k = vec2(1,-1);
    return normalize(k.xyy*scene_sdf(point + k.xyy*h)
		     + k.yyx*scene_sdf(point + k.yyx*h)
		     + k.yxy*scene_sdf(point + k.yxy*h)
		     + k.xxx*scene_sdf(point + k.xxx*h));
}

void main()
{
	vec3 ray_origin = constants.eye.xyz;

	// Raycast
	RayShot shot = raymarch(ray_origin, ray_dir);
	if (shot.hit) {
		out_color = vec4(calc_normal(shot.closest) * 0.5 + 0.5, 1);
	} else {
		out_color = vec4(0, 0, 0, 1);
	}
	gl_FragDepth = shot.depth / 1000;
}
