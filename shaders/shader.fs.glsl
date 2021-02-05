#version 450

layout (push_constant, std140) uniform PushConstants {
        vec4 forward;
        vec4 eye;
        vec4 dir;
        float aspect;
} constants;

layout (location = 0) in vec3 dir;

layout (location = 0) out vec4 out_color;

float MAX_DIST = 20.0;

float sphere_sdf(vec3 pos) {
        vec3 sphere_center = vec3(0, 0, 3);
        float sphere_radius = 1;

        float dist_to_sphere_center = length(pos - sphere_center);
        return dist_to_sphere_center - sphere_radius;
}

// http://blog.hvidtfeldts.net/index.php/2011/09/distance-estimated-3d-fractals-v-the-mandelbulb-different-de-approximations/
float mandel_sdf(vec3 pos) {
        int Power = 8;
	vec3 z = pos;
	float dr = 1.0;
	float r = 0.0;
	for (int i = 0; i < 10; i++) {
		r = length(z);
		if (r>2.0) break;
		
		// convert to polar coordinates
		float theta = acos(z.z/r);
		float phi = atan(z.y,z.x);
		dr =  pow( r, Power-1.0)*Power*dr + 1.0;
		
		// scale and rotate the point
		float zr = pow( r,Power);
		theta = theta*Power;
		phi = phi*Power;
		
		// convert back to cartesian coordinates
		z = zr*vec3(sin(theta)*cos(phi), sin(phi)*sin(theta), cos(theta));
		z+=pos;
	}
	return 0.5*log(r)*r/dr;
}

float distance_fun(vec3 pos) {
        return mandel_sdf(pos);
}

// Returns the point of collision.
vec3 march(vec3 pos, vec3 dir) {
        vec3 start = pos;
        normalize(dir);

        float threshold = 0;

	float last_step_len = 999.9;
	while (last_step_len > threshold && length(pos - start) < MAX_DIST) {
        	// Move as far as we safely can in the direction of [dir]
                float dist = distance_fun(pos); 
                vec3 step = dist * dir;
                pos += step;

                last_step_len = dist;

        	float dist_to_cam = length(constants.eye.xyz - pos);
                threshold = max(dist_to_cam * 0.005, 0.00001);
	}

	return pos;
}

void main() {
	vec3 point = constants.eye.xyz;

	vec3 pos = march(point, dir);
	float distance = length(pos - point);

	float epsilon = 0.001;
	float point_dist = distance_fun(pos);
	vec3 normal = normalize(vec3(
		distance_fun(pos + vec3(epsilon, 0.0, 0.0)) - point_dist,
		distance_fun(pos + vec3(0.0, epsilon, 0.0)) - point_dist,
		distance_fun(pos + vec3(0.0, 0.0, epsilon)) - point_dist
	));

	if (distance < MAX_DIST) {
        	out_color = vec4(normal * 0.5 + 0.5, 1.0);
	} else {
        	out_color = vec4(0.0, 0.0, 0.0, 1.0);
	}
}
