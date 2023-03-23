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

vec3 vertices[36] = vec3[](
    vec3(-1.0f,-1.0f,-1.0f),
    vec3(-1.0f,-1.0f, 1.0f),
    vec3(-1.0f, 1.0f, 1.0f),
    vec3(1.0f, 1.0f,-1.0f),
    vec3(-1.0f,-1.0f,-1.0f),
    vec3(-1.0f, 1.0f,-1.0f),
    vec3(1.0f,-1.0f, 1.0f),
    vec3(-1.0f,-1.0f,-1.0f),
    vec3(1.0f,-1.0f,-1.0f),
    vec3(1.0f, 1.0f,-1.0f),
    vec3(1.0f,-1.0f,-1.0f),
    vec3(-1.0f,-1.0f,-1.0f),
    vec3(-1.0f,-1.0f,-1.0f),
    vec3(-1.0f, 1.0f, 1.0f),
    vec3(-1.0f, 1.0f,-1.0f),
    vec3(1.0f,-1.0f, 1.0f),
    vec3(-1.0f,-1.0f, 1.0f),
    vec3(-1.0f,-1.0f,-1.0f),
    vec3(-1.0f, 1.0f, 1.0f),
    vec3(-1.0f,-1.0f, 1.0f),
    vec3(1.0f,-1.0f, 1.0f),
    vec3(1.0f, 1.0f, 1.0f),
    vec3(1.0f,-1.0f,-1.0f),
    vec3(1.0f, 1.0f,-1.0f),
    vec3(1.0f,-1.0f,-1.0f),
    vec3(1.0f, 1.0f, 1.0f),
    vec3(1.0f,-1.0f, 1.0f),
    vec3(1.0f, 1.0f, 1.0f),
    vec3(1.0f, 1.0f,-1.0f),
    vec3(-1.0f, 1.0f,-1.0f),
    vec3(1.0f, 1.0f, 1.0f),
    vec3(-1.0f, 1.0f,-1.0f),
    vec3(-1.0f, 1.0f, 1.0f),
    vec3(1.0f, 1.0f, 1.0f),
    vec3(-1.0f, 1.0f, 1.0f),
    vec3(1.0f,-1.0f, 1.0)
);

layout (location = 0) out int obj_idx;

// Originally I calculated the ray direction in the vertex shader, but that doesn't get interpolated
// correctly. Instead we output worldspace coordinates (which get interpolated correctly) and let
// the fragment shader calculate the ray direction.
layout (location = 1) out vec3 pos_worldspace;

void main() {
	// Figure out which object we are
	int vertex_count = 36;
	obj_idx = gl_VertexIndex / vertex_count;

	// If this object doesn't actually exist, put it off screen so it gets clipped
	if (obj_idx >= scene.count) {
		gl_Position = vec4(-100, -100, -100, 1);
		return;
	}

	// Otherwise figure out where it goes on screen
        pos_worldspace = (scene.objects[obj_idx].transform
		       * vec4(vertices[gl_VertexIndex % vertex_count], 1)).xyz;

	vec4 pos_screenspace = constants.proj * constants.view * vec4(pos_worldspace, 1);

	gl_Position = pos_screenspace;

}
