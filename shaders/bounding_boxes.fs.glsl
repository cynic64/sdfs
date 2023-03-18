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
	float sizes[256];
} objects;

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

float sd_box(vec3 point, vec3 box_size) {
	vec3 d = abs(point) - box_size;
	return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

float sd_sphere(vec3 point, float radius) {
	return length(point) - radius;
}

float sd_cross(vec3 point) {
	float inf = 1.0 / 0.0;
	float a = sd_box(point, vec3(inf, 1, 1));
	float b = sd_box(point, vec3(1, inf, 1));
	float c = sd_box(point, vec3(1, 1, inf));
	return min(a, min(b, c));
}

// From https://www.shadertoy.com/view/4sX3Rn
const mat3 ma = mat3( 0.60, 0.00,  0.80,
                      0.00, 1.00,  0.00,
                     -0.80, 0.00,  0.60 );
float sd_menger(vec3 p) {
	p /= objects.sizes[obj_idx];

	float d = sd_box(p,vec3(1.0));
	vec4 res = vec4( d, 1.0, 0.0, 0.0 );

	float s = 1.0;
	for( int m=0; m<5; m++ )
	{
		vec3 a = mod( p*s, 2.0 )-1.0;
		s *= 3.0;
		vec3 r = abs(1.0 - 3.0*abs(a));
		float da = max(r.x,r.y);
		float db = max(r.y,r.z);
		float dc = max(r.z,r.x);
		float c = (min(da,min(db,dc))-1.0)/s;

		if( c>d ) {
			d = c;
			res = vec4( d, min(res.y,0.2*da*db*dc), (1.0+float(m))/4.0, 0.0 );
		}
	}

	return res.x;
}

// Taken from https://iquilezles.org/articles/distfunctions/
float sd_box_frame(vec3 point, vec3 box_size, float e) {
	vec3 p = abs(point)-box_size;
	vec3 q = abs(p+e)-e;
	return min(min(length(max(vec3(p.x,q.y,q.z),0.0))+min(max(p.x,max(q.y,q.z)),0.0),
		       length(max(vec3(q.x,p.y,q.z),0.0))+min(max(q.x,max(p.y,q.z)),0.0)),
		   length(max(vec3(q.x,q.y,p.z),0.0))+min(max(q.x,max(q.y,p.z)),0.0));
}

float scene_sdf(vec3 point) {
	float min_dist = 99999999;
	int type = objects.types[obj_idx];
	float dist;
	if (type == 0) {
		dist = sd_sphere(point - objects.poss[obj_idx].xyz, objects.sizes[obj_idx]);
	} else if (type == 1) {
		dist = sd_box(point - objects.poss[obj_idx].xyz, vec3(objects.sizes[obj_idx]));
	} else if (type == 2) {
		dist = sd_menger(point - objects.poss[obj_idx].xyz);
	} else if (type == 3) {
		float size = objects.sizes[obj_idx];
		dist = sd_box_frame(point - objects.poss[obj_idx].xyz, vec3(size),
				    0.1 * size);
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
	vec3 ray_dir = normalize(pos_worldspace - ray_origin);

	// Raycast
	RayShot shot = raymarch(ray_origin, ray_dir);
	if (shot.hit) {
		out_color = vec4((calc_normal(shot.closest) * 0.5 + 0.5), 1);
		gl_FragDepth = clamp(shot.depth / 1000, 0, 1);
	} else {
		out_color = vec4(vec3(objects.types[obj_idx] * 0.2) + 0.1, 1);
		gl_FragDepth = 1;
	}
}
