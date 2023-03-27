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

layout (std140, binding = 1) buffer ComputeOut {
	vec3 force;
	vec3 torque;
	vec3 linear_impulse;
	vec3 angular_impulse;
	uint collision_count;

	// Idk where else to put this. Surely there is a better way to set some value to 0 before
	// all invocations run?
	uint debug_out_idx;
} out_buf;

layout (std140, binding = 2) buffer DebugIn {
	vec3 line_poss[DEBUG_MAX_LINES];
	vec3 line_dirs[DEBUG_MAX_LINES];
} debug_buf;

layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

#include common.glsl

void main() {
	// Compute intersection between first 2 objects
	float x = gl_GlobalInvocationID.x * 0.025 - 1 + 0.0125;
	float y = gl_GlobalInvocationID.y * 0.025 - 1 + 0.0125;
	float z = gl_GlobalInvocationID.z * 0.025 - 1 + 0.0125;

	int a_type = in_buf.objects[0].type, b_type = in_buf.objects[1].type;
	mat4 a_transform = in_buf.objects[0].transform,
		b_transform = in_buf.objects[1].transform;

	vec3 point = vec3(x, y, z);
	float dist0 = scene_sdf(a_type, a_transform, point);
	float dist1 = scene_sdf(b_type, b_transform, point);

	if (dist0 <= 0 && dist1 <= 0) {
		vec3 my_normal = calc_normal(a_type, a_transform, point);
		vec3 other_normal = calc_normal(b_type, b_transform, point);

		// If there's a collision, write the average of (the opposite of our normal) and
		// (the other normal). Those should both point roughly in the right direction to
		// un-intersect ourselves.

		// For deep penetrations, the two might cancel each other out. Oh well, nothing we
		// can really do.
		//vec3 collision_normal = (-my_normal + other_normal) * 0.5;
		vec3 collision_normal = other_normal;

		if (length(collision_normal) == 0) return;
		collision_normal = normalize(collision_normal);

		// Objects' center of mass
		vec3 a_com = (a_transform * vec4(0, 0, 0, 1)).xyz;
		vec3 b_com = (b_transform * vec4(0, 0, 0, 1)).xyz;

		// Vector from COM to point
		vec3 a_from_com = point - a_com;
		vec3 b_from_com = point - b_com;

		// Cross of force and vector to COM, which is the force's contribution to torque or
		// something
		// man this hurts my brain
		//vec3 torque = cross(r, force);

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
		mat3 b_inertia_inverse = mat3(0, 0, 0,
					      0, 0, 0,
					      0, 0, 0);

		// The cross product to find angular velocity's effect in world-space should make
		// sense: The more perpendicular `x_from_com` is to the axis of rotation, the more
		// angular velocity will contribute.
		vec3 a_vel = in_buf.objects[0].linear_vel
			+ cross(in_buf.objects[0].angular_vel, a_from_com);
		vec3 b_vel = in_buf.objects[1].linear_vel
			+ cross(in_buf.objects[1].angular_vel, b_from_com);
		vec3 rel_vel = a_vel - b_vel;

		// Taken from https://www.chrishecker.com/images/b/bb/Gdmphys4.pdf, what an ungodly
		// formula
		vec3 n = collision_normal;
		float impulse =
			// How much relative velocity there is along the collision normal: the
			// relative velocity might be high, but it it's at a grazing angle, the
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
		// Or something like that, it's not Chris' article so I guessed. Mass doesn't matter
		// because it's encoded in the inertia tensor.
		vec3 angular_impulse = a_inertia_inverse
			* cross(a_from_com, collision_normal * impulse);
		atomicAdd(out_buf.angular_impulse.x, angular_impulse.x);
		atomicAdd(out_buf.angular_impulse.y, angular_impulse.y);
		atomicAdd(out_buf.angular_impulse.z, angular_impulse.z);

		// All the atomic adds seem like they should be really slow, but somehow aren't.
		atomicAdd(out_buf.collision_count, 1);

		// Add a line to debug view, as long as there is space
		uint idx = atomicAdd(out_buf.debug_out_idx, 1);
		if (idx < DEBUG_MAX_LINES) {
			debug_buf.line_poss[idx] = point;
			debug_buf.line_dirs[idx] = vec3(collision_normal);
		}
	}
}
