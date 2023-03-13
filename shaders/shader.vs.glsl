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

vec2 positions[6] = vec2[](
        vec2(-1.0, -1.0),
        vec2(1.0, -1.0),
        vec2(1.0, 1.0),
        vec2(-1.0, -1.0),
        vec2(1.0, 1.0),
        vec2(-1.0, 1.0)
);

layout (location = 0) out vec3 ray_dir;
//layout (location = 0) out vec2 out_pos;

void main() {
        vec2 pos = positions[gl_VertexIndex];
	gl_Position = vec4(pos, 0.0, 1.0);

	float aspect = constants.iResolution.x / constants.iResolution.y;
	vec3 right = normalize(cross(constants.forward.xyz, vec3(0, -1, 0))) * aspect;
	vec3 up = normalize(cross(constants.forward.xyz, right));
	ray_dir = normalize(pos.x * right + -pos.y * up + constants.forward.xyz * 1.5);

	// For fragment shaders that do the camera calculations themselves
	/*
	out_pos = (pos * 0.5 + 0.5) * constants.iResolution.xy;
	*/
}
