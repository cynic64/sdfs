#version 450

layout (push_constant, std140) uniform PushConstants {
        vec4 forward;
        vec4 eye;
        vec4 dir;
        vec4 aspect;
        vec4 exp;
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

// https://www.shadertoy.com/view/ltfSWn
float mandel_sdf(vec3 pos) {
        pos.yz = pos.zy;
        float exp = constants.exp.x;
        vec3 w = pos;
        float m = dot(w,w);

        vec4 trap = vec4(abs(w),m);
        float dz = 1.0;

        for( int i=0; i<8; i++ ) {
                dz = exp*pow(sqrt(m),exp-1)*dz + 1.0;

                float r = length(w);
                float b = exp*acos( w.y/r);
                float a = exp*atan( w.x, w.z );
                w = pos + pow(r,exp) * vec3( sin(b)*sin(a), cos(b), sin(b)*cos(a) );

                m = dot(w,w);
                if( m > 256.0 ) break;
        }

        return 0.25*log(m)*sqrt(m)/dz;
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
                threshold = max(dist_to_cam * 0.001, 0.0001);
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
