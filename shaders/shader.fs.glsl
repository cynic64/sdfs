#version 450

layout (push_constant, std140) uniform PushConstants {
       vec4 iResolution;
       vec4 iMouse;
       float iFrame;
       float iTime;
       vec4 forward;
       vec4 eye;
       vec4 dir;
} constants;

layout (std140, set = 0, binding = 0) uniform Uniform {
	int box_count;
	vec4 box_poss[256];
} uni;

struct RayShot {
	vec3 closest;
	bool hit;
};

// Should be normalized
layout (location = 0) in vec3 ray_dir;

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
	vec3 box_size = vec3(1, 2, 3);
	for (int i = 0; i < uni.box_count; i++) {
		float dist = sd_box(point - uni.box_poss[i].xyz, box_size);
		if (dist < min_dist) min_dist = dist;
	}

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
			shot.hit = true;
			return shot;
		}
		depth += dist;
	}

	RayShot shot;
	shot.hit = false;
	return shot;
}

// Taken from iquilezles.org/articles/normalsSDF/
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
	vec3 box_size = vec3(0.5);

	// Raycast
	RayShot shot = raymarch(ray_origin, ray_dir);
	if (shot.hit) {
		out_color = vec4(calc_normal(shot.closest) * 0.5 + 0.5, 1);
	} else {
		out_color = vec4(0, 0, 0, 1);
	}
}
