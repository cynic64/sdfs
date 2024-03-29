#version 450
#extension GL_EXT_shader_atomic_float : enable

#include constants.glsl

struct Object {
	int type;
	mat4 transform;

	vec3 pos;
	mat4 orientation;
	vec3 linear_vel;
	vec3 angular_vel;
};

layout (std140, binding = 0) buffer readonly Scene {
	int count;
	Object objects[];
} in_buf;

// Gives us more info on CPU about how the collision was calculated
struct CollisionDebug {
	// Where the collision happened
	vec3 pos;
	// Object velocities
	vec3 a_linear_vel;
	vec3 b_linear_vel;
	vec3 a_angular_vel;
	vec3 b_angular_vel;
	// Object normals
	vec3 a_normal;
	vec3 b_normal;
	// Ideally, the two normals would be exactly opposite. Since we can only approximate normals,
	// however, we have to take the average of -a_normal and b_normal. That's what this is.
	vec3 collision_normal;
	// If a_normal and b_normal cancel each other out exactly, we return early.
	uint return_early;
	// Object center of masses
	vec3 a_com;
	vec3 b_com;
	// Vectors from each object's COM to the point of collision
	vec3 a_from_com;
	vec3 b_from_com;
	// Velocity of each object (linear + angular) at the point of collision
	vec3 a_vel;
	vec3 b_vel;
	// Relative velocity between A and B at collision point
	vec3 rel_vel;

	float impulse;
	// The effect the collision will have on A's linear/angular velocity
	vec3 linear_impulse;
	vec3 angular_impulse;
};

layout (std140, binding = 1) buffer ComputeOut {
	vec3 force;
	vec3 torque;
	vec3 linear_impulse;
	vec3 angular_impulse;

	uint collision_count;
	CollisionDebug debug;
} out_buf;

layout (local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

#include common.glsl

void compute_impulse(vec3 point, Object a, Object b) {
	float normal_detail = 0.00002;
	vec3 a_normal = calc_normal(a.type, a.transform, point, normal_detail);
	vec3 b_normal = calc_normal(b.type, b.transform, point, normal_detail);

	// If there's a collision, write the average of (the opposite of our normal) and
	// (the other normal). Those should both point roughly in the right direction to
	// un-intersect ourselves.

	// For deep penetrations, the two might cancel each other out. Oh well, nothing we
	// can really do.
	vec3 collision_normal = (-a_normal + b_normal) * 0.5;
	//vec3 collision_normal = -a_normal;

	if (length(collision_normal) == 0) {
		out_buf.debug.return_early = 1;
		return;
	}
	collision_normal = normalize(collision_normal);

	// Objects' center of mass
	vec3 a_com = (a.transform * vec4(0, 0, 0, 1)).xyz;
	vec3 b_com = (b.transform * vec4(0, 0, 0, 1)).xyz;

	// Vector from COM to point
	vec3 a_from_com = point - a_com;
	vec3 b_from_com = point - b_com;

	// 1 = perfectly elastic, 0 = all momentum absorbed on collision
	float restitution = 1;
	// Eventually these should be passed as part of Object
	float a_mass = 1;
	// Object B can't be moved, so it has infinite mass
	float b_mass = 1 / 0;
	// Eventually I will calculate these properly
	mat3 a_inertia_inverse = mat3(1, 0, 0,
				      0, 1, 0,
				      0, 0, 1);
	// Again, B has infinite mass so the inertia inverse is all 0's
	mat3 b_inertia_inverse = mat3(0, 0, 0,
				      0, 0, 0,
				      0, 0, 0);

	// The cross product to find angular velocity's effect in world-space should make
	// sense: The more perpendicular `x_from_com` is to the axis of rotation, the more
	// angular velocity will contribute.
	vec3 a_vel = a.linear_vel
		+ cross(a.angular_vel, a_from_com);
	vec3 b_vel = b.linear_vel
		+ cross(b.angular_vel, b_from_com);
	vec3 rel_vel = a_vel - b_vel;

	// Taken from https://www.chrishecker.com/images/b/bb/Gdmphys4.pdf, what an ungodly
	// formula
	vec3 n = collision_normal;
	float impulse =
		// How much relative velocity there is along the collision normal: the
		// relative velocity might be high, but if it's at a grazing angle, the
		// impulse should be scaled down. This is what the dot product accomplishes.
		dot(-(1 + restitution) * rel_vel, n)
		/
		(
		    // Since we normalize `n`, I don't think all the dot products are
		    // necessary, but whatever. Dividing by the inverse of mass should make
		    // sense: If either thing is super heavy, the impulse will be
		    // greater. Though dividing by mass would usually make the impulse smaller,
		    // we're in the denominator so this has the effect we want.
		    dot(n, n) * (1.0 / a_mass + 1.0 / b_mass)
		    // If we weren't bothering with angular effects, we would be done. But we
		    // do care about angular stuff, so here we go:
		    /*
		    +
		    dot
		    (
		    // I still have no idea what the fuck an inertia tensor is, but the rest
		    // kind of makes sense. The inner cross product (a_from_com x n) tells us
		    // around what axis this impulse will try to spin us around (If I whack
		    // you on an ice rink, you will start spinning around an axis
		    // perpendicular both to the vector I whacked you along (n) and the vector
		    // going from where I whacked you to your center of mass (a_from_com)).
		    //
		    // Then I guess the inertia tensor inverse accounts for it being harder to
		    // spin around some axes than others. It's a lot easier to spin a pencil
		    // between your fingers than when it's lying on a table, for example.
		    //
		    // Not sure what the outer cross product accomplishes. I think the
		    // resulting vector will face in direction -n.
		    cross(a_inertia_inverse * cross(a_from_com, n), a_from_com)
		    +
		    // Now we do the same for B
		    cross(b_inertia_inverse * cross(b_from_com, n), b_from_com)
		    // And now we dot the whole thing with n, I dunno why...
		    , n
		    )
		    */
		    );
	// Phew...

	// Maybe impulse is the wrong name for this, since impulse is scalar. But I need to
	// be able to sum the individual effects impulse has on velocity. The formula for
	// updating velocity from impulse looks like this:
	//
	// new_v = old_v + (impulse / mass) * collision_normal
	//
	// To avoid having having to store all collision normals, we do the multiplication
	// here so we can sum it into a single vec3.
	vec3 linear_impulse = (impulse / a_mass) * collision_normal;
	atomicAdd(out_buf.linear_impulse.x, linear_impulse.x);
	atomicAdd(out_buf.linear_impulse.y, linear_impulse.y);
	atomicAdd(out_buf.linear_impulse.z, linear_impulse.z);

	// Same thing here:
	//
	// new_omega = old_omega + a_inertia_inverse * cross(a_from_com, collision_normal * j)
	//
	// Or something like that, it's not in Chris' article so I guessed. Mass doesn't
	// matter because it's encoded in the inertia tensor.
	vec3 angular_impulse = a_inertia_inverse
		* cross(a_from_com, collision_normal * impulse);
	/*
	atomicAdd(out_buf.angular_impulse.x, angular_impulse.x);
	atomicAdd(out_buf.angular_impulse.y, angular_impulse.y);
	atomicAdd(out_buf.angular_impulse.z, angular_impulse.z);
	*/

	// Record debug info
	out_buf.debug.pos = point;
	out_buf.debug.a_linear_vel = a.linear_vel;
	out_buf.debug.b_linear_vel = b.linear_vel;
	out_buf.debug.a_angular_vel = a.angular_vel;
	out_buf.debug.b_angular_vel = b.angular_vel;
	out_buf.debug.a_normal = a_normal;
	out_buf.debug.b_normal = b_normal;
	out_buf.debug.collision_normal = collision_normal;
	out_buf.debug.a_com = a_com;
	out_buf.debug.b_com = b_com;
	out_buf.debug.a_from_com = a_from_com;
	out_buf.debug.b_from_com = b_from_com;
	out_buf.debug.a_vel = a_vel;
	out_buf.debug.b_vel = b_vel;
	out_buf.debug.rel_vel = rel_vel;
	out_buf.debug.impulse = impulse;
	out_buf.debug.linear_impulse = linear_impulse;
	out_buf.debug.angular_impulse = angular_impulse;

	// All the atomic adds seem like they should be really slow, but somehow aren't.
	//atomicAdd(out_buf.collision_count, 1);
}

void main() {
	// Compute intersection between first 2 objects
	float x = gl_GlobalInvocationID.x * 0.0125 - 1 + 0.00625;
	float y = gl_GlobalInvocationID.y * 0.0125 - 1 + 0.00625;
	float z = gl_GlobalInvocationID.z * 0.0125 - 1 + 0.00625;

	int a_type = in_buf.objects[0].type, b_type = in_buf.objects[1].type;
	mat4 a_transform = in_buf.objects[0].transform,
		b_transform = in_buf.objects[1].transform;

	vec3 point = vec3(x, y, z);
	float dist0 = scene_sdf(a_type, a_transform, point);
	float dist1 = scene_sdf(b_type, b_transform, point);

	float thresh = length(vec3(0.0125));
	bool collision = dist0 <= thresh && dist1 <= thresh;
	if (collision) {
		// Only compute impulse for one point
		if (atomicAdd(out_buf.collision_count, 1) == 0) {
			compute_impulse(point, in_buf.objects[0], in_buf.objects[1]);
		}
	}
}
