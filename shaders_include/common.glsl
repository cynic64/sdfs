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

// Taken from same page ^
float sd_cone(vec3 p, vec2 c, float h) {
	//p.y -= h / 2;
	// c is sin(angle), cos(angle)

	// Alternatively pass q instead of (c,h),
	// which is the point at the base in 2D
	vec2 q = h*vec2(c.x/c.y,-1.0);
    
	vec2 w = vec2( length(p.xz), p.y );
	vec2 a = w - q*clamp( dot(w,q)/dot(q,q), 0.0, 1.0 );
	vec2 b = w - q*vec2( clamp( w.x/q.x, 0.0, 1.0 ), 1.0 );
	float k = sign( q.y );
	float d = min(dot( a, a ),dot(b, b));
	float s = max( k*(w.x*q.y-w.y*q.x),k*(w.y-q.y)  );
	return sqrt(d)*sign(s);
}

float sd_pyramid( vec3 p, float h) {
	float m2 = h*h + 0.25;
    
	p.xz = abs(p.xz);
	p.xz = (p.z>p.x) ? p.zx : p.xz;
	p.xz -= 0.5;

	vec3 q = vec3( p.z, h*p.y - 0.5*p.x, h*p.x + 0.5*p.y);
   
	float s = max(-q.x,0.0);
	float t = clamp( (q.y-0.5*p.z)/(m2+0.25), 0.0, 1.0 );
    
	float a = m2*(q.x+s)*(q.x+s) + q.y*q.y;
	float b = m2*(q.x+0.5*t)*(q.x+0.5*t) + (q.y-m2*t)*(q.y-m2*t);
    
	float d2 = min(q.y,-q.x*m2-q.y*0.5) > 0.0 ? 0.0 : min(a,b);
    
	return sqrt( (d2+q.z*q.z)/m2 ) * sign(max(q.z,-p.y));
}

float scene_sdf(int type, mat4 transform, vec3 point) {
	vec3 point_rel = (inverse(transform) * vec4(point, 1)).xyz;
	float dist;
	if (type == 0) {
		dist = sd_sphere(point_rel, 1);
	} else if (type == 1) {
		dist = sd_box(point_rel, vec3(1));
	} else if (type == 2) {
		dist = sd_menger(point_rel);
	} else if (type == 3) {
		dist = sd_box_frame(point_rel, vec3(1), 0.05);
	} else if (type == 4) {
		dist = sd_pyramid(point_rel, 1);
	} else {
		dist = 0;
	}

	return dist;
}

// Taken from iquilezles.org/articles/normalsSDF/
// Messing with this is bad juju
vec3 calc_normal(int type, mat4 transform, vec3 point) {
    const float h = 0.0002;
    const vec2 k = vec2(1,-1);
    vec3 n = k.xyy*scene_sdf(type, transform, point + k.xyy*h)
		     + k.yyx*scene_sdf(type, transform, point + k.yyx*h)
		     + k.yxy*scene_sdf(type, transform, point + k.yxy*h)
		     + k.xxx*scene_sdf(type, transform, point + k.xxx*h);
    if (n.x == 0 && n.y == 0 && n.z == 0) {
	    return vec3(0);
    } else {
	    return normalize(n);
    }
}
