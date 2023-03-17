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

const char* DEVICE_EXTS[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
const int DEVICE_EXT_CT = 1;

const double MOUSE_SENSITIVITY_FACTOR = 0.001;
const float MOVEMENT_SPEED = 0.6F;

#define CONCURRENT_FRAMES 4

const VkFormat SC_FORMAT_PREF = VK_FORMAT_B8G8R8A8_UNORM;
const VkPresentModeKHR SC_PRESENT_MODE_PREF = VK_PRESENT_MODE_IMMEDIATE_KHR;

struct SyncSet {
        VkFence render_fence;
        VkSemaphore acquire_sem;
        VkSemaphore render_sem;
};

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

#define MAX_BOX_COUNT 256

struct Uniform {
	int box_count;
	vec4 box_poss[MAX_BOX_COUNT];
};

void sync_set_create(VkDevice device, struct SyncSet* sync_set) {
        fence_create(device, VK_FENCE_CREATE_SIGNALED_BIT, &sync_set->render_fence);
        semaphore_create(device, &sync_set->acquire_sem);
        semaphore_create(device, &sync_set->render_sem);
}

void sync_set_destroy(VkDevice device, struct SyncSet* sync_set) {
	vkDestroyFence(device, sync_set->render_fence, NULL);
	vkDestroySemaphore(device, sync_set->acquire_sem, NULL);
	vkDestroySemaphore(device, sync_set->render_sem, NULL);
}

int main() {
        glfwInit();
        glfwSetErrorCallback(glfw_error_callback);
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        GLFWwindow* window = glfwCreateWindow(800, 600, "Vulkan", NULL, NULL);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// Base
        struct Base base;
        base_create(window, 1, 0, NULL, DEVICE_EXT_CT, DEVICE_EXTS, &base);

	// Swapchain
        struct Swapchain swapchain;
        swapchain_create(base.surface, base.phys_dev, base.device,
			 SC_FORMAT_PREF, SC_PRESENT_MODE_PREF, &swapchain);

        // Load shaders
        VkShaderModule fullscreen_vs, fullscreen_fs, boxes_vs, boxes_fs;
        VkPipelineShaderStageCreateInfo fullscreen_shaders[2] = {0};
        VkPipelineShaderStageCreateInfo boxes_shaders[2] = {0};

	// Fullscreen
	load_shader(base.device, "shaders/fullscreen.vs.spv",
		    &fullscreen_vs, VK_SHADER_STAGE_VERTEX_BIT, &fullscreen_shaders[0]);
	load_shader(base.device, "shaders/fullscreen.fs.spv",
		    &fullscreen_fs, VK_SHADER_STAGE_FRAGMENT_BIT, &fullscreen_shaders[1]);

	// Bounding boxes
	load_shader(base.device, "shaders/bounding_boxes.vs.spv",
		    &boxes_vs, VK_SHADER_STAGE_VERTEX_BIT, &boxes_shaders[0]);
	load_shader(base.device, "shaders/bounding_boxes.fs.spv",
		    &boxes_fs, VK_SHADER_STAGE_FRAGMENT_BIT, &boxes_shaders[1]);

        // Render pass
	VkRenderPass rpass;
	rpass_color(base.device, swapchain.format, &rpass);

	// Allocate uniform buffer
	struct Buffer uniform_buf, uniform_buf_staging;
	buffer_create_staged(base.phys_dev, base.device, base.queue, base.cpool,
			     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			     sizeof(struct Uniform), NULL, &uniform_buf, &uniform_buf_staging);

	struct Uniform* uniform_data;
	VkResult res = vkMapMemory(base.device, uniform_buf_staging.mem, 0, sizeof(struct Uniform),
				   0, (void **) &uniform_data);
	assert(res == VK_SUCCESS);

	// Create descriptor set
	struct DescriptorInfo uniform_desc = {0};
	uniform_desc.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	uniform_desc.shader_stage_flags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
	uniform_desc.buffer.buffer = uniform_buf.handle;
	uniform_desc.buffer.range = VK_WHOLE_SIZE;

	struct SetInfo set_info = {0};
	set_info.desc_ct = 1;
	set_info.descs = &uniform_desc;

	// Create descriptor pool
	VkDescriptorPool dpool;
	dpool_create(base.device, 1, &uniform_desc, &dpool);
	
	// Create the set
	VkDescriptorSetLayout set_layout;
	set_layout_create(base.device, &set_info, &set_layout);

	VkDescriptorSet set;
	set_create(base.device, dpool, set_layout, &set_info, &set);

        // Fullscreen pipeline
	// Layout
	VkPushConstantRange pushc_range = {0};
        pushc_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pushc_range.size = sizeof(struct PushConstants);

        VkPipelineLayoutCreateInfo pipe_layout_info = {0};
        pipe_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipe_layout_info.pushConstantRangeCount = 1;
        pipe_layout_info.pPushConstantRanges = &pushc_range;
	pipe_layout_info.setLayoutCount = 1;
	pipe_layout_info.pSetLayouts = &set_layout;

        VkPipelineLayout pipe_layout;
        res = vkCreatePipelineLayout(base.device, &pipe_layout_info, NULL,
				     &pipe_layout);
        assert(res == VK_SUCCESS);

        // Actual pipeline
        struct PipelineSettings pipe_settings = PIPELINE_SETTINGS_DEFAULT;

	VkPipeline fullscreen_pipe;
	pipeline_create(base.device, &pipe_settings,
	                sizeof(fullscreen_shaders) / sizeof(fullscreen_shaders[0]), fullscreen_shaders,
	                pipe_layout, rpass, 0, &fullscreen_pipe);

        vkDestroyShaderModule(base.device, fullscreen_vs, NULL);
        vkDestroyShaderModule(base.device, fullscreen_fs, NULL);

	// Aaaaaand bounding box pipeline
	VkPipeline boxes_pipe;
	pipeline_create(base.device, &pipe_settings,
	                sizeof(boxes_shaders) / sizeof(boxes_shaders[0]), boxes_shaders,
	                pipe_layout, rpass, 0, &boxes_pipe);

        vkDestroyShaderModule(base.device, boxes_vs, NULL);
        vkDestroyShaderModule(base.device, boxes_fs, NULL);

        // Framebuffers
        VkFramebuffer* framebuffers = malloc(swapchain.image_ct * sizeof(framebuffers[0]));
        for (int i = 0; i < swapchain.image_ct; i++) {
                VkImageView views[] = {swapchain.views[i]};
                framebuffer_create(base.device, rpass, swapchain.width, swapchain.height,
                                   sizeof(views) / sizeof(views[0]), views, &framebuffers[i]);
        }

        // Command buffers
        VkCommandBuffer cbufs[CONCURRENT_FRAMES];
        for (int i = 0; i < CONCURRENT_FRAMES; i++) cbuf_alloc(base.device, base.cpool, &cbufs[i]);

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
	camera.eye[0] = 0.0F; camera.eye[1] = 0.0F; camera.eye[2] = -100.0F; 
	double last_mouse_x, last_mouse_y;
	glfwGetCursorPos(window, &last_mouse_x, &last_mouse_y);

	// Main loop
        int frame_ct = 0;
        struct timespec start_time = timer_start();
        struct timespec last_frame_time = timer_start();

        int must_recreate = 0;
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

                        for (int i = 0; i < swapchain.image_ct; i++) {
                                vkDestroyFramebuffer(base.device, framebuffers[i], NULL);

                                VkImageView views[] = {swapchain.views[i]};
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

                VkCommandBuffer cbuf = cbufs[frame_idx];

                // Wait for the render process using these sync objects to finish rendering
                res = vkWaitForFences(base.device, 1, &sync_set->render_fence, VK_TRUE, UINT64_MAX);
                assert(res == VK_SUCCESS);

		// Update uniform buffer
		uniform_data->box_count = 40;

		srand(0);
		double elapsed = timer_get_elapsed(&start_time);
		for (int i = 0; i < MAX_BOX_COUNT; i++) {
			uniform_data->box_poss[i][0] = rand() % 64 * sin(elapsed);
			uniform_data->box_poss[i][1] = rand() % 64 * cos(elapsed);
			uniform_data->box_poss[i][2] = rand() % 64 * sin(elapsed + 3.14159265);
		};

		buffer_copy(base.queue, cbuf, uniform_buf_staging.handle, uniform_buf.handle,
			    sizeof(struct Uniform));

                // Reset command buffer
                vkResetCommandBuffer(cbuf, 0);

                // Acquire an image
                uint32_t image_idx;
                res = vkAcquireNextImageKHR(base.device, swapchain.handle, UINT64_MAX,
                                            sync_set->acquire_sem, VK_NULL_HANDLE, &image_idx);
                if (res == VK_ERROR_OUT_OF_DATE_KHR) {
                        must_recreate = 1;
                        continue;
                } else assert(res == VK_SUCCESS);

                // Record command buffer
                cbuf_begin_onetime(cbuf);

                VkClearValue clear_vals[] = {{{{0.0F, 0.0F, 0.0F, 1.0F}}},
                                             {{{0.0F, 0.0F, 0.0F, 1.0F}}}};

                VkRenderPassBeginInfo cbuf_rpass_info = {0};
                cbuf_rpass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
                cbuf_rpass_info.renderPass = rpass;
                cbuf_rpass_info.framebuffer = framebuffers[image_idx];
                cbuf_rpass_info.renderArea.extent.width = swapchain.width;
                cbuf_rpass_info.renderArea.extent.height = swapchain.height;
                cbuf_rpass_info.clearValueCount = sizeof(clear_vals) / sizeof(clear_vals[0]);
                cbuf_rpass_info.pClearValues = clear_vals;
                vkCmdBeginRenderPass(cbuf, &cbuf_rpass_info, VK_SUBPASS_CONTENTS_INLINE);

                //vkCmdBindPipeline(cbuf, VK_PIPELINE_BIND_POINT_GRAPHICS, fullscreen_pipe);

                VkViewport viewport = {0};
                viewport.width = swapchain.width;
                viewport.height = swapchain.height;
                viewport.minDepth = 0.0F;
                viewport.maxDepth = 1.0F;
                vkCmdSetViewport(cbuf, 0, 1, &viewport);

                VkRect2D scissor = {0};
                scissor.extent.width = swapchain.width;
                scissor.extent.height = swapchain.height;
                vkCmdSetScissor(cbuf, 0, 1, &scissor);

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

                vkCmdBindPipeline(cbuf, VK_PIPELINE_BIND_POINT_GRAPHICS, boxes_pipe);
		vkCmdPushConstants(cbuf, pipe_layout,
				   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		                   0, sizeof(struct PushConstants), &pushc_data);
		vkCmdBindDescriptorSets(cbuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_layout,
					0, 1, &set, 0, NULL);
                //vkCmdDraw(cbuf, 6, 1, 0, 0);

                vkCmdDraw(cbuf, 36 * uniform_data->box_count, 1, 0, 0);

                vkCmdEndRenderPass(cbuf);

                res = vkEndCommandBuffer(cbuf);
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
                VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                VkSubmitInfo submit_info = {0};
                submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submit_info.waitSemaphoreCount = 1;
                submit_info.pWaitSemaphores = &sync_set->acquire_sem;
                submit_info.pWaitDstStageMask = &wait_stage;
                submit_info.commandBufferCount = 1;
                submit_info.pCommandBuffers = &cbuf;
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
        printf("FPS: %.2f\n", fps);

        vkDeviceWaitIdle(base.device);

        vkDestroyPipeline(base.device, fullscreen_pipe, NULL);
        vkDestroyPipelineLayout(base.device, pipe_layout, NULL);

        vkDestroyPipeline(base.device, boxes_pipe, NULL);

        vkDestroyRenderPass(base.device, rpass, NULL);

        swapchain_destroy(base.device, &swapchain);

        for (int i = 0; i < CONCURRENT_FRAMES; i++) {
                sync_set_destroy(base.device, &sync_sets[i]);
        }

        for (int i = 0; i < swapchain.image_ct; i++) {
                vkDestroyFramebuffer(base.device, framebuffers[i], NULL);
        }

	vkDestroyDescriptorPool(base.device, dpool, NULL);
	vkDestroyDescriptorSetLayout(base.device, set_layout, NULL);
	vkUnmapMemory(base.device, uniform_buf_staging.mem);
	buffer_destroy(base.device, &uniform_buf);
	buffer_destroy(base.device, &uniform_buf_staging);
        base_destroy(&base);

        glfwTerminate();

        free(framebuffers);
        free(image_fences);
}
