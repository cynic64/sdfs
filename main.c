#include "external/cglm/include/cglm/affine.h"
#include "external/cglm/include/cglm/mat4.h"
#include "external/cglm/include/cglm/vec3.h"
#include "external/render-c/src/cbuf.h"
#include <vulkan/vulkan_core.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#include "external/cglm/include/cglm/cglm.h"

#include "external/render-c/src/base.h"
#include "external/render-c/src/buffer.h"
#include "external/render-c/src/glfw_error.h"
#include "external/render-c/src/image.h"
#include "external/render-c/src/pipeline.h"
#include "external/render-c/src/rpass.h"
#include "external/render-c/src/set.h"
#include "external/render-c/src/shader.h"
#include "external/render-c/src/swapchain.h"
#include "external/render-c/src/sync.h"

#include "external/render-utils/src/camera.h"
#include "external/render-utils/src/timer.h"

#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <sys/param.h>
#include <stdint.h>

const char* DEVICE_EXTS[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
const int DEVICE_EXT_CT = 1;

const double MOUSE_SENSITIVITY_FACTOR = 0.001;
const float MOVEMENT_SPEED = 0.6F;

#define CONCURRENT_FRAMES 4

const VkFormat SC_FORMAT_PREF = VK_FORMAT_B8G8R8A8_UNORM;
const VkPresentModeKHR SC_PRESENT_MODE_PREF = VK_PRESENT_MODE_IMMEDIATE_KHR;
const VkFormat DEPTH_FORMAT = VK_FORMAT_D32_SFLOAT;

struct PushConstants {
	// Pad to vec4 because std140
	vec4 iResolution;
	vec4 iMouse;
	float iFrame[1];
	float iTime[1];
	// Camera stuff
	vec4 forward;
	vec4 eye;
	vec4 dir;
	mat4 view;
	mat4 proj;
};

#define MAX_OBJ_COUNT 512

struct Uniform {
	int32_t count;

	// Shader pads everything to 32 bytes, even arrays :(
	__attribute__((aligned(16))) int32_t types[MAX_OBJ_COUNT * 4];

	// The base box has a side length of 2 (goes from -1 to 1)
	__attribute__((aligned(16))) mat4 transforms[MAX_OBJ_COUNT];
};

struct StorageData {
	int32_t x;
};

// This stuff exists for every concurrent frame
struct SyncSet {
        VkFence render_fence;
        VkSemaphore acquire_sem;
	// Signalled when compute shader finishes. Graphics submission waits on this and
	// acquire_sem.
	VkSemaphore compute_sem;
        VkSemaphore render_sem;
};

void sync_set_create(VkDevice device, struct SyncSet* sync_set) {
        fence_create(device, VK_FENCE_CREATE_SIGNALED_BIT, &sync_set->render_fence);
        semaphore_create(device, &sync_set->acquire_sem);
        semaphore_create(device, &sync_set->compute_sem);
        semaphore_create(device, &sync_set->render_sem);
}

void sync_set_destroy(VkDevice device, struct SyncSet* sync_set) {
	vkDestroyFence(device, sync_set->render_fence, NULL);
	vkDestroySemaphore(device, sync_set->acquire_sem, NULL);
	vkDestroySemaphore(device, sync_set->compute_sem, NULL);
	vkDestroySemaphore(device, sync_set->render_sem, NULL);
}

void abs_vec3(vec3 in, vec3 out) {
	for (int i = 0; i < 3; i++) {
		out[i] = in[i] > 0 ? in[i] : -in[i];
	}
}

float length(vec3 in) {
	return sqrtf(in[0]*in[0] + in[1]*in[1] + in[2]*in[2]);
}

float sd_box(vec3 point, vec3 box_size) {
	vec3 d;
	abs_vec3(point, d); // d = abs(point)
	for (int i = 0; i < 3; i++) {
		// d -= box_size;
		d[i] -= box_size[i];
	}
	vec3 d_max;
	for (int i = 0; i < 3; i++) {
		d_max[i] = d[i] > 0 ? d[i] : 0;
	}
	return fmin(fmax(d[0],fmax(d[1],d[2])),0.0) + length(d_max);
}

float sd_sphere(vec3 point, float radius) {
	return length(point) - radius;
}

// Creates a rotation matrix that makes (0, 1, 0) point in the direction of `normal`
// Expects `normal` to be normalized
void normal_matrix(vec3 normal, mat4 out) {
	vec3 perp = {-normal[1], normal[0], 0};
	vec3 perp2;
	glm_vec3_cross(normal, perp, perp2);
	glm_vec3_normalize(perp2);
	glm_vec3_normalize(perp);

	glm_mat4_identity(out);
	// This row is where 1 0 0 ends up
	memcpy(out[0], perp, sizeof(vec3));
	// This is where 0 1 0 ends up
	memcpy(out[1], normal, sizeof(vec3));
	// This is where 0 0 1 ends up
	memcpy(out[2], perp2, sizeof(vec3));
}

void calc_normal(vec3 in, vec3 out) {
    const float h = 0.0002;

    vec3 a_point = {in[0] + h, in[1] - h, in[2] - h};
    float a_dist = sd_sphere(a_point, 2);
    vec3 a_vec = {+a_dist, -a_dist, -a_dist};

    vec3 b_point = {in[0] - h, in[1] - h, in[2] + h};
    float b_dist = sd_sphere(b_point, 2);
    vec3 b_vec = {-b_dist, -b_dist, +b_dist};

    vec3 c_point = {in[0] - h, in[1] + h, in[2] - h};
    float c_dist = sd_sphere(c_point, 2);
    vec3 c_vec = {-c_dist, +c_dist, -c_dist};

    vec3 d_point = {in[0] + h, in[1] + h, in[2] + h};
    float d_dist = sd_sphere(d_point, 2);
    vec3 d_vec = {+d_dist, +d_dist, +d_dist};

    out[0] = a_vec[0] + b_vec[0] + c_vec[0] + d_vec[0];
    out[1] = a_vec[1] + b_vec[1] + c_vec[1] + d_vec[1];
    out[2] = a_vec[2] + b_vec[2] + c_vec[2] + d_vec[2];

    glm_vec3_normalize(out);
}

int calc_intersect(struct Uniform* uni,
		   vec3 sphere_pos,
		   float start_x, float start_y, float start_z,
		   float end_x, float end_y, float end_z, int depth) {
	vec3 box_pos = {0, 3, 0};
	float radius = 2;

	int steps = 2;
	float cell_width = (end_x - start_x) / steps;
	float cell_height = (end_y - start_y) / steps;
	float cell_depth = (end_z - start_z) / steps;
	assert(cell_width == cell_height && cell_height == cell_depth);
	// Length of cell's diagonal
	float margin = sqrtf(cell_width*cell_width*0.25
			     + cell_height*cell_height*0.25
			     + cell_depth*cell_depth*0.25);

	int iter_count = 0;

	for (int x = 0; x < steps; x++) {
		for (int y = 0; y < steps; y++) {
			for (int z = 0; z < steps; z++) {
				vec3 point = {start_x + (x + 0.5)*cell_width,
				              start_y + (y + 0.5)*cell_height,
				              start_z + (z + 0.5)*cell_depth};

				// Point relative to box's position
				vec3 box_point;
				glm_vec3_sub(point, box_pos, box_point);
				float box_dist = sd_box(box_point, (vec3) {2, 2, 2});

				// Point relative to spheres's position
				vec3 sphere_point;
				glm_vec3_sub(point, sphere_pos, sphere_point);
				float sphere_dist = sd_sphere(sphere_point, radius);

				float max_dist = fmax(sphere_dist, box_dist);
				float min_dist = fmin(sphere_dist, box_dist);
				
				if (max_dist < margin && -min_dist < margin) {
					if (depth < 2) {
						iter_count +=
							calc_intersect(uni,
								       sphere_pos,
								       start_x + x*cell_width,
								       start_y + y*cell_height,
								       start_z + z*cell_depth,
								       start_x + (x+1)*cell_width,
								       start_y + (y+1)*cell_height,
								       start_z + (z+1)*cell_depth,
								       depth + 1);
					} else if (uni->count < MAX_OBJ_COUNT) {
						vec3 normal;
						calc_normal(point, normal);

						point[0] -= 4;

						// Box frame
						uni->types[4 * uni->count] = 3;
						glm_translate_make(uni->transforms[uni->count], point);
						glm_scale(uni->transforms[uni->count],
							  (vec3) {cell_width / 2,
								  cell_height / 2,
								  cell_depth / 2});
						uni->count++;

						// Box showing normal (cones don't scale well for
						// whatever reason)
						uni->types[4 * uni->count] = 1;
						mat4 trans1;
						glm_translate_make(trans1,
								   (vec3) {point[0],
									   point[1],
									   point[2]});
						mat4 trans2;
						glm_translate_make(trans2, (vec3) {0, 2.5, 0});
						mat4 scale;
						glm_scale_make(scale,
							       (vec3) {cell_width / 4,
								       cell_height / 4,
								       cell_depth / 4});
						mat4 norm;
						normal_matrix(normal, norm);
						glm_mat4_mulN((mat4*[])
							      {&trans1, &scale, &norm, &trans2}, 4,
							      uni->transforms[uni->count]);
						uni->count++;
					}
				}

				iter_count++;
			}
		}
	}

	return iter_count;
}

int main() {
	printf("Uniform buffer size: %lu\n", sizeof(struct Uniform));
	
        glfwInit();
        glfwSetErrorCallback(glfw_error_callback);
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        GLFWwindow* window = glfwCreateWindow(800, 600, "Vulkan", NULL, NULL);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// Base
        struct Base base;
        base_create(window, 1, 1, 0, NULL, DEVICE_EXT_CT, DEVICE_EXTS, &base);

	// Swapchain
        struct Swapchain swapchain;
        swapchain_create(base.surface, base.phys_dev, base.device,
			 SC_FORMAT_PREF, SC_PRESENT_MODE_PREF, &swapchain);

        // Load shaders
        VkShaderModule graphics_vs, graphics_fs;
        VkPipelineShaderStageCreateInfo graphics_shaders[2] = {0};

	// Graphics shaders
	load_shader(base.device, "shaders/graphics.vs.spv",
		    &graphics_vs, VK_SHADER_STAGE_VERTEX_BIT, &graphics_shaders[0]);
	load_shader(base.device, "shaders/graphics.fs.spv",
		    &graphics_fs, VK_SHADER_STAGE_FRAGMENT_BIT, &graphics_shaders[1]);

	// Compute shader
	VkShaderModule compute_shader;
	VkPipelineShaderStageCreateInfo compute_shader_info = {0};
	load_shader(base.device, "shaders/compute.spv",
		    &compute_shader, VK_SHADER_STAGE_COMPUTE_BIT, &compute_shader_info);

        // Render pass
	VkRenderPass rpass;
	rpass_color_depth(base.device, swapchain.format, DEPTH_FORMAT, &rpass);

	// Allocate storage buffers for compute shader
	struct Buffer compute_in_bufs[CONCURRENT_FRAMES], compute_out_bufs[CONCURRENT_FRAMES];
	struct Buffer compute_buf_staging, compute_buf_reader;
	struct StorageData compute_buf_initial_data = {0};
	// Staging buffer
	buffer_create(base.phys_dev, base.device,
		      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		      sizeof(struct StorageData), &compute_buf_staging);

	// Actual storage buffers
	for (int i = 0; i < CONCURRENT_FRAMES; i++) {
		buffer_create(base.phys_dev, base.device,
			    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			    sizeof(struct StorageData), &compute_in_bufs[i]);
		buffer_create(base.phys_dev, base.device,
			    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			    sizeof(struct StorageData), &compute_out_bufs[i]);
	}

	// This is so we can read data back out
	buffer_create(base.phys_dev, base.device,
		      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		      sizeof(struct StorageData), &compute_buf_reader);

	// Descriptor info for compute input buffer (corresponds to compute_bufs[0]);
	struct DescriptorInfo compute_in_desc = {0};
	compute_in_desc.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	compute_in_desc.stage = VK_SHADER_STAGE_COMPUTE_BIT;

	struct DescriptorInfo compute_out_desc = {0};
	compute_out_desc.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	compute_out_desc.stage = VK_SHADER_STAGE_COMPUTE_BIT
		| VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

	// Create descriptor pool
	// Gotta be honest I have no idea what I'm doing here
	VkDescriptorPoolSize dpool_sizes[1] = {0};
	dpool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	// I think it has to be 3 for each CONCURRENT_FRAME because there's the two descriptors for
	// the compute stage and the descriptor for the graphics stage
	dpool_sizes[0].descriptorCount = CONCURRENT_FRAMES*3;

	VkDescriptorPoolCreateInfo dpool_info = {0};
	dpool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	dpool_info.poolSizeCount = sizeof(dpool_sizes) / sizeof(dpool_sizes[0]);
	dpool_info.pPoolSizes = dpool_sizes;
	// Need two sets per CONCURRENT_FRAME because there's 1 for compute and 1 for graphics.
	dpool_info.maxSets = 2*CONCURRENT_FRAMES;

	VkDescriptorPool dpool;
	VkResult res = vkCreateDescriptorPool(base.device, &dpool_info, NULL, &dpool);
	assert(res == VK_SUCCESS);

	// Now make the sets
	struct DescriptorInfo compute_descs[] = {compute_in_desc, compute_out_desc};
	struct SetInfo compute_set_info = {0};
	compute_set_info.desc_ct = sizeof(compute_descs) / sizeof(compute_descs[0]);
	compute_set_info.descs = compute_descs;

	struct SetInfo graphics_set_info = {0};
	graphics_set_info.desc_ct = 1;
	graphics_set_info.descs = &compute_out_desc;
	
	VkDescriptorSetLayout compute_set_layout;
	set_layout_create(base.device, &compute_set_info, &compute_set_layout);

	VkDescriptorSetLayout graphics_set_layout;
	set_layout_create(base.device, &graphics_set_info, &graphics_set_layout);

	VkDescriptorSet compute_sets[CONCURRENT_FRAMES];
	VkDescriptorSet graphics_sets[CONCURRENT_FRAMES];

	for (int i = 0; i < CONCURRENT_FRAMES; i++) {
		// Set for compute
		union SetHandle compute_buffers[2] = {0};
		compute_buffers[0].buffer.buffer = compute_in_bufs[i].handle;
		compute_buffers[0].buffer.range = VK_WHOLE_SIZE;
		compute_buffers[1].buffer.buffer = compute_out_bufs[i].handle;
		compute_buffers[1].buffer.range = VK_WHOLE_SIZE;
		assert(sizeof(compute_buffers) / sizeof(compute_buffers[0])
		       == compute_set_info.desc_ct);

		set_create(base.device, dpool, compute_set_layout, &compute_set_info, compute_buffers,
			   &compute_sets[i]);

		// Set for graphics
		union SetHandle graphics_buffers[1] = {0};
		graphics_buffers[0].buffer.buffer = compute_out_bufs[i].handle;
		graphics_buffers[0].buffer.range = VK_WHOLE_SIZE;
		assert(sizeof(graphics_buffers) / sizeof(graphics_buffers[0])
		       == graphics_set_info.desc_ct);
		set_create(base.device, dpool, graphics_set_layout, &graphics_set_info,
			   graphics_buffers, &graphics_sets[i]);
	}

	// Compute pipeline
	// Layout
	VkPipelineLayoutCreateInfo compute_pipe_layout_info = {0};
	compute_pipe_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	compute_pipe_layout_info.setLayoutCount = 1;
	compute_pipe_layout_info.pSetLayouts = &compute_set_layout;

	VkPipelineLayout compute_pipe_layout;
	res = vkCreatePipelineLayout(base.device, &compute_pipe_layout_info, NULL,
				     &compute_pipe_layout);
	assert(res == VK_SUCCESS);

	// Actual pipeline
	VkComputePipelineCreateInfo compute_pipe_info = {0};
	compute_pipe_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	compute_pipe_info.layout = compute_pipe_layout;
	compute_pipe_info.stage = compute_shader_info;

	VkPipeline compute_pipe;
	res = vkCreateComputePipelines(base.device, NULL, 1, &compute_pipe_info, NULL, &compute_pipe);
	assert(res == VK_SUCCESS);

	vkDestroyShaderModule(base.device, compute_shader, NULL);

        // Graphics pipeline
	// Layout
	VkPushConstantRange pushc_range = {0};
        pushc_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pushc_range.size = sizeof(struct PushConstants);

        VkPipelineLayoutCreateInfo pipe_layout_info = {0};
        pipe_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipe_layout_info.pushConstantRangeCount = 1;
        pipe_layout_info.pPushConstantRanges = &pushc_range;
	pipe_layout_info.setLayoutCount = 1;
	pipe_layout_info.pSetLayouts = &graphics_set_layout;

        VkPipelineLayout graphics_pipe_layout;
        res = vkCreatePipelineLayout(base.device, &pipe_layout_info, NULL,
				     &graphics_pipe_layout);
        assert(res == VK_SUCCESS);

        // Actual pipeline
        struct PipelineSettings graphics_pipe_settings = PIPELINE_SETTINGS_DEFAULT;
	graphics_pipe_settings.depth.depthTestEnable = VK_TRUE;
	graphics_pipe_settings.depth.depthWriteEnable = VK_TRUE;
	graphics_pipe_settings.depth.depthCompareOp = VK_COMPARE_OP_LESS;
	graphics_pipe_settings.rasterizer.cullMode = VK_CULL_MODE_NONE;

	VkPipeline graphics_pipe;
	pipeline_create(base.device, &graphics_pipe_settings,
	                sizeof(graphics_shaders) / sizeof(graphics_shaders[0]), graphics_shaders,
	                graphics_pipe_layout, rpass, 0, &graphics_pipe);

        vkDestroyShaderModule(base.device, graphics_vs, NULL);
        vkDestroyShaderModule(base.device, graphics_fs, NULL);

        // Framebuffers, we'll create them later
        VkFramebuffer* framebuffers = malloc(swapchain.image_ct * sizeof(framebuffers[0]));
	bzero(framebuffers, swapchain.image_ct * sizeof(framebuffers[0]));
	int must_recreate = 1;

	struct Image depth_image = {0};

        // Command buffers
        VkCommandBuffer graphics_cbufs[CONCURRENT_FRAMES];
        VkCommandBuffer compute_cbufs[CONCURRENT_FRAMES];
        for (int i = 0; i < CONCURRENT_FRAMES; i++) {
		cbuf_alloc(base.device, base.cpool, &graphics_cbufs[i]);
		cbuf_alloc(base.device, base.cpool, &compute_cbufs[i]);
	}

        // Sync sets
        struct SyncSet sync_sets [CONCURRENT_FRAMES];
        for (int i = 0; i < CONCURRENT_FRAMES; i++) sync_set_create(base.device, &sync_sets[i]);

        // Image fences
        VkFence* image_fences = malloc(swapchain.image_ct * sizeof(image_fences[0]));
        for (int i = 0; i < swapchain.image_ct; i++) image_fences[i] = VK_NULL_HANDLE;

	// Camera
	struct CameraFly camera;
	camera.pitch = 0.0F;
	camera.yaw = 0.0F;
	camera.eye[0] = 0.0F; camera.eye[1] = 0.0F; camera.eye[2] = -10.0F; 
	double last_mouse_x, last_mouse_y;
	glfwGetCursorPos(window, &last_mouse_x, &last_mouse_y);

	// Main loop
        int frame_ct = 0;
        struct timespec start_time = timer_start();
        struct timespec last_frame_time = timer_start();
	double total_collision_time = 0;

        while (!glfwWindowShouldClose(window)) {
                while (must_recreate) {
                        must_recreate = 0;
                        vkDeviceWaitIdle(base.device);

                        VkFormat old_format = swapchain.format;
                        uint32_t old_image_ct = swapchain.image_ct;

                        swapchain_destroy(base.device, &swapchain);
                        swapchain_create(base.surface, base.phys_dev, base.device,
                                         old_format, SC_PRESENT_MODE_PREF, &swapchain);

                        assert(swapchain.format == old_format && swapchain.image_ct == old_image_ct);

                        for (int i = 0; i < CONCURRENT_FRAMES; i++) {
                                sync_set_destroy(base.device, &sync_sets[i]);
                                sync_set_create(base.device, &sync_sets[i]);
                        }

			// Recreate depth
			// Only destroy if it actually exists (it's initialized to all NULL)
			if (depth_image.handle != NULL) {
				image_destroy(base.device, &depth_image);
			}
			// Now actually recreate it
			image_create_depth(base.phys_dev, base.device, DEPTH_FORMAT,
					   swapchain.width, swapchain.height,
					   VK_SAMPLE_COUNT_1_BIT, &depth_image);

			// Recreate framebuffers
                        for (int i = 0; i < swapchain.image_ct; i++) {
				if (framebuffers[i] != NULL) {
					vkDestroyFramebuffer(base.device, framebuffers[i], NULL);
				}

                                VkImageView views[] = {swapchain.views[i], depth_image.view};
                                framebuffer_create(base.device, rpass,
						   swapchain.width, swapchain.height,
                                                   sizeof(views) / sizeof(views[0]), views,
						   &framebuffers[i]);

                                image_fences[i] = VK_NULL_HANDLE;
                        }

                        int real_width, real_height;
                        glfwGetFramebufferSize(window, &real_width, &real_height);
                        if (real_width != swapchain.width || real_height != swapchain.height) {
				must_recreate = 1;
			}
                }

                // Handle input
                // Mouse movement
                double new_mouse_x, new_mouse_y;
                glfwGetCursorPos(window, &new_mouse_x, &new_mouse_y);
                double d_mouse_x = new_mouse_x - last_mouse_x, d_mouse_y = new_mouse_y - last_mouse_y;
                double delta = timer_get_elapsed(&last_frame_time);
                last_frame_time = timer_start(last_frame_time);
        	last_mouse_x = new_mouse_x; last_mouse_y = new_mouse_y;

        	// Keys
        	vec3 cam_movement = {0.0F, 0.0F, 0.0F};
		float speed_multiplier = glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS ? 20 : 1;
        	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
			cam_movement[2] += MOVEMENT_SPEED * speed_multiplier;
        	}
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
			cam_movement[2] -= MOVEMENT_SPEED * speed_multiplier;
        	}
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
			cam_movement[0] -= MOVEMENT_SPEED * speed_multiplier;
        	}
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
			cam_movement[0] += MOVEMENT_SPEED * speed_multiplier;
		} 

        	// Update camera
        	camera_fly_update(&camera,
                                  d_mouse_x * MOUSE_SENSITIVITY_FACTOR,
				  d_mouse_y * MOUSE_SENSITIVITY_FACTOR,
        	                  cam_movement, delta);

		// Set up frame
                int frame_idx = frame_ct % CONCURRENT_FRAMES;
                struct SyncSet* sync_set = &sync_sets[frame_idx];

                VkCommandBuffer graphics_cbuf = graphics_cbufs[frame_idx],
			compute_cbuf = compute_cbufs[frame_idx];

                // Wait for the render process using these sync objects to finish rendering. There's
                // no need to explicitly wait for compute to finish because render waits for compute
                // anyway.
                res = vkWaitForFences(base.device, 1, &sync_set->render_fence, VK_TRUE, UINT64_MAX);
                assert(res == VK_SUCCESS);

                // Record compute dispatch
                vkResetCommandBuffer(compute_cbuf, 0);
		cbuf_begin_onetime(compute_cbuf);
		vkCmdBindPipeline(compute_cbuf, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipe);
		vkCmdBindDescriptorSets(compute_cbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
					compute_pipe_layout, 0, 1,
					&compute_sets[frame_idx], 0, NULL);
		vkCmdDispatch(compute_cbuf, 1, 1, 1);
                res = vkEndCommandBuffer(compute_cbuf);
                assert(res == VK_SUCCESS);

		VkSubmitInfo compute_submit_info = {0};
		compute_submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		compute_submit_info.commandBufferCount = 1;
		compute_submit_info.pCommandBuffers = &compute_cbuf;
		compute_submit_info.signalSemaphoreCount = 1;
		compute_submit_info.pSignalSemaphores = &sync_set->compute_sem;

		res = vkQueueSubmit(base.queue, 1, &compute_submit_info, NULL);
		assert(res == VK_SUCCESS);

                // Acquire an image
                uint32_t image_idx;
                res = vkAcquireNextImageKHR(base.device, swapchain.handle, UINT64_MAX,
                                            sync_set->acquire_sem, VK_NULL_HANDLE, &image_idx);
                if (res == VK_ERROR_OUT_OF_DATE_KHR) {
                        must_recreate = 1;
                        continue;
                } else assert(res == VK_SUCCESS);

                // Record graphics commands
                vkResetCommandBuffer(graphics_cbuf, 0);
                cbuf_begin_onetime(graphics_cbuf);

                VkClearValue clear_vals[] = {{{{0.0F, 0.0F, 0.0F, 1.0F}}},
                                             {{{1.0F}}}};

                VkRenderPassBeginInfo rpass_begin = {0};
                rpass_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
                rpass_begin.renderPass = rpass;
                rpass_begin.framebuffer = framebuffers[image_idx];
                rpass_begin.renderArea.extent.width = swapchain.width;
                rpass_begin.renderArea.extent.height = swapchain.height;
                rpass_begin.clearValueCount = sizeof(clear_vals) / sizeof(clear_vals[0]);
                rpass_begin.pClearValues = clear_vals;
                vkCmdBeginRenderPass(graphics_cbuf, &rpass_begin, VK_SUBPASS_CONTENTS_INLINE);

                VkViewport viewport = {0};
                viewport.width = swapchain.width;
                viewport.height = swapchain.height;
                viewport.minDepth = 0.0F;
                viewport.maxDepth = 1.0F;
                vkCmdSetViewport(graphics_cbuf, 0, 1, &viewport);

                VkRect2D scissor = {0};
                scissor.extent.width = swapchain.width;
                scissor.extent.height = swapchain.height;
                vkCmdSetScissor(graphics_cbuf, 0, 1, &scissor);

                struct PushConstants pushc_data;
		pushc_data.iResolution[0] = swapchain.width;
		pushc_data.iResolution[1] = swapchain.height;
		pushc_data.iMouse[0] = new_mouse_x;
		pushc_data.iMouse[1] = new_mouse_y;
		pushc_data.iMouse[2] = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
		pushc_data.iFrame[0] = frame_ct;
		pushc_data.iTime[0] = timer_get_elapsed(&start_time);
		memcpy(pushc_data.forward, camera.forward, sizeof(camera.forward));
		memcpy(pushc_data.eye, camera.eye, sizeof(camera.eye));
		pushc_data.dir[0] = camera.yaw;
		pushc_data.dir[1] = camera.pitch;
		memcpy(pushc_data.view, camera.view, sizeof(camera.view));
                glm_perspective(1.0F, (float) swapchain.width / (float) swapchain.height, 0.1F,
				10000.0F, pushc_data.proj);

                vkCmdBindPipeline(graphics_cbuf, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipe);
		vkCmdPushConstants(graphics_cbuf, graphics_pipe_layout,
				   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		                   0, sizeof(struct PushConstants), &pushc_data);
		vkCmdBindDescriptorSets(graphics_cbuf, VK_PIPELINE_BIND_POINT_GRAPHICS,
					graphics_pipe_layout,
					0, 1, &graphics_sets[frame_idx], 0, NULL);

                vkCmdDraw(graphics_cbuf, 36, 1, 0, 0);

                vkCmdEndRenderPass(graphics_cbuf);
		
                res = vkEndCommandBuffer(graphics_cbuf);
                assert(res == VK_SUCCESS);

                // Wait until whoever is rendering to the image is done
                if (image_fences[image_idx] != VK_NULL_HANDLE)
                        vkWaitForFences(base.device, 1, &image_fences[image_idx], VK_TRUE, UINT64_MAX);

                // Reset fence
                res = vkResetFences(base.device, 1, &sync_set->render_fence);
                assert(res == VK_SUCCESS);

                // Mark it as in use by us
                image_fences[image_idx] = sync_set->render_fence;

                // Submit
                VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
						      VK_PIPELINE_STAGE_VERTEX_SHADER_BIT};
		VkSemaphore wait_sems[] = {sync_set->acquire_sem, sync_set->compute_sem};
		static_assert(sizeof(wait_stages) / sizeof(wait_stages[0])
			      == sizeof(wait_sems) / sizeof(wait_sems[0]),
			      "Must have same # of wait stages and wait semaphores");

                VkSubmitInfo submit_info = {0};
                submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submit_info.waitSemaphoreCount = sizeof(wait_sems) / sizeof(wait_sems[0]);
                submit_info.pWaitSemaphores = wait_sems;
                submit_info.pWaitDstStageMask = wait_stages;
                submit_info.commandBufferCount = 1;
                submit_info.pCommandBuffers = &graphics_cbuf;
                submit_info.signalSemaphoreCount = 1;
                submit_info.pSignalSemaphores = &sync_set->render_sem;

                res = vkQueueSubmit(base.queue, 1, &submit_info, sync_set->render_fence);
                assert(res == VK_SUCCESS);

                // Present
                VkPresentInfoKHR present_info = {0};
                present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
                present_info.waitSemaphoreCount = 1;
                present_info.pWaitSemaphores = &sync_set->render_sem;
                present_info.swapchainCount = 1;
                present_info.pSwapchains = &swapchain.handle;
                present_info.pImageIndices = &image_idx;

                res = vkQueuePresentKHR(base.queue, &present_info);
                if (res == VK_ERROR_OUT_OF_DATE_KHR) must_recreate = 1;
                else assert(res == VK_SUCCESS);

                frame_ct++;
                glfwPollEvents();
        }

	double elapsed = timer_get_elapsed(&start_time);
        double fps = (double) frame_ct / elapsed;
        printf("FPS: %.2f, total frames: %d\n", fps, frame_ct);
        printf("Average collision time: %.2f ms\n", total_collision_time / frame_ct * 1000);

        vkDeviceWaitIdle(base.device);

        vkDestroyPipelineLayout(base.device, graphics_pipe_layout, NULL);
        vkDestroyPipelineLayout(base.device, compute_pipe_layout, NULL);

        vkDestroyPipeline(base.device, graphics_pipe, NULL);
        vkDestroyPipeline(base.device, compute_pipe, NULL);

        vkDestroyRenderPass(base.device, rpass, NULL);

        swapchain_destroy(base.device, &swapchain);
	image_destroy(base.device, &depth_image);

        for (int i = 0; i < CONCURRENT_FRAMES; i++) {
                sync_set_destroy(base.device, &sync_sets[i]);
        }

        for (int i = 0; i < swapchain.image_ct; i++) {
                vkDestroyFramebuffer(base.device, framebuffers[i], NULL);
        }

	vkDestroyDescriptorPool(base.device, dpool, NULL);

	vkDestroyDescriptorSetLayout(base.device, compute_set_layout, NULL);
	vkDestroyDescriptorSetLayout(base.device, graphics_set_layout, NULL);

	buffer_destroy(base.device, &compute_buf_staging);
	buffer_destroy(base.device, &compute_buf_reader);
	for (int i = 0; i < CONCURRENT_FRAMES; i++) {
		buffer_destroy(base.device, &compute_in_bufs[i]);
		buffer_destroy(base.device, &compute_out_bufs[i]);
	}

        base_destroy(&base);

        glfwTerminate();

        free(framebuffers);
        free(image_fences);
}
