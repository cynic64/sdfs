#include "external/cglm/include/cglm/mat4.h"
#include "external/cglm/include/cglm/vec3.h"
#include "external/render-c/src/cbuf.h"
#include "external/render-c/src/mem.h"
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

#include "font.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/param.h>

const char *DEVICE_EXTS[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                             VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME};
const int DEVICE_EXT_CT = 2;

const double MOUSE_SENSITIVITY_FACTOR = 0.001;
const float MOVEMENT_SPEED = 0.6F;

#define CONCURRENT_FRAMES 4

const VkFormat SC_FORMAT_PREF = VK_FORMAT_B8G8R8A8_UNORM;
const VkPresentModeKHR SC_PRESENT_MODE_PREF = VK_PRESENT_MODE_IMMEDIATE_KHR;
const VkFormat DEPTH_FORMAT = VK_FORMAT_D32_SFLOAT;

// aligned(16) only aligns the start of the structure, not the members. I don't think it's necessary
// but gcc complains otherwise.
struct __attribute__((packed, aligned(16))) PushConstants {
        // Pad to vec4 because std140
        vec4 iResolution; // vec2
        vec4 iMouse;      // float
                          // floats get packed together in std140...
        float iFrame;
        float iTime;
        // But the next vec4 will be 16-byte-aligned again
        vec2 _pad;
        // Camera stuff
        vec4 forward; // vec3
        vec4 eye;     // vec3
        vec4 dir;     // vec3
        mat4 view;
        mat4 proj;
};

// Both must be the same in ./shaders_include/constants.glsl
#define MAX_OBJECTS 512
#define DEBUG_MAX_LINES 65536

struct __attribute__((packed, aligned(16))) Object {
        int32_t type[4]; // int
        mat4 transform;

        // Stuff for physics
        vec4 pos; // vec3
        mat4 orientation;
        vec4 linear_vel;  // vec3
        vec4 angular_vel; // vec3
};

struct __attribute__((packed, aligned(16))) Scene {
        int32_t count[4]; // int
        struct Object objects[MAX_OBJECTS];
};

struct __attribute__((packed, aligned(16))) ComputeOut {
        vec4 force; // vec3

        // Derivative of the angular velocity
        vec4 torque; // vec3

        // Instantaneous change in velocity
        vec4 linear_impulse; // vec3

        // Instantaneous change in angular velocity
        vec3 angular_impulse;

        uint32_t collision_count;
};

// I need to be able to see what's going on when my compute shader inevitably breaks. This is what
// `compute_debug.glsl` outputs.
struct __attribute__((packed, aligned(16))) ComputeDebugOut {
        vec4 a_com; // vec3
        vec4 b_com; // vec3
};

// Gets passed to debug pass. Yes, I know I could have just used a vertex buffer, but this is more
// flexible if I want to include more stuff later. It's a debug layer, performance is less
// important.
struct __attribute__((packed, aligned(16))) Debug {
        vec4 line_poss[DEBUG_MAX_LINES]; // vec3[]
        vec4 line_dirs[DEBUG_MAX_LINES]; // vec3[]
};

#define TEXT_MAX_CHARS 256
struct __attribute__((packed, aligned(16))) Text {
        mat4 transform;
        // Yeah, this wastes a ton of space, but I can't avoid it without using textures and I'm too
        // lazy. It's not like we're passing megabytes of text, so whatever.
        int chars[4 * TEXT_MAX_CHARS]; // int[TEXT_MAX_CHARS]
        int char_count;
};

// This stuff exists for every concurrent frame
struct SyncSet {
        VkFence render_fence;
        VkSemaphore acquire_sem;
        // Signalled when compute shader finishes. Graphics submission waits on this and
        // acquire_sem.
        VkSemaphore render_sem;
};

// Maybe the debug stuff should go in its own struct? All they share is the input buffer and set
// layout. Ah, whatever.
struct PhysicsEngine {
        VkCommandBuffer cbufs[CONCURRENT_FRAMES];
        VkPipelineLayout pipe_layout;
        VkPipeline pipe;

        // I think it's totally uncessary to have 1 of these for each CONCURRENT_FRAME since I
        // always wait for the shader to finish before continuing. But whatever, it'll be useful
        // soon™.
        VkFence fences[CONCURRENT_FRAMES];
        struct Buffer in_bufs[CONCURRENT_FRAMES];
        struct Buffer out_bufs[CONCURRENT_FRAMES];
        struct Buffer reader;
        // Used to reset output before running shader
        struct Buffer reset;
        VkDescriptorSet sets[CONCURRENT_FRAMES];
        VkDescriptorSetLayout set_layout;
};

// Runs a compute shader that has access to all the same functions as the physics one. Lets me debug
// stuff GPU-side. It can only be run in the paused debug mode so we don't need one item for each
// CONCURRENT_FRAME.
struct PhysicsDebug {
        VkCommandBuffer cbuf;
        VkPipelineLayout pipe_layout;
        VkPipeline pipe;
        VkFence fence;
        struct Buffer in_buf;
        struct Buffer out_buf;
        struct Buffer reader;
        VkDescriptorSet set;
        VkDescriptorSetLayout set_layout;
};

void sync_set_create(VkDevice device, struct SyncSet *sync_set) {
        fence_create(device, VK_FENCE_CREATE_SIGNALED_BIT, &sync_set->render_fence);
        semaphore_create(device, &sync_set->acquire_sem);
        semaphore_create(device, &sync_set->render_sem);
}

void sync_set_destroy(VkDevice device, struct SyncSet *sync_set) {
        vkDestroyFence(device, sync_set->render_fence, NULL);
        vkDestroySemaphore(device, sync_set->acquire_sem, NULL);
        vkDestroySemaphore(device, sync_set->render_sem, NULL);
}

// `src` must be null-terminated
void text_write(struct Text *dst, const char *src, float x, float y, float scale) {
        int i;
        for (i = 0; src[i] != '\0'; i++) {
                assert(i < TEXT_MAX_CHARS);
                dst->chars[4 * i] = src[i];
        }
        dst->char_count = i;

        glm_scale_make(dst->transform, (vec3){scale * i / 2, scale, 0});
        glm_translated(dst->transform, (vec3){x, y, 0});
}

void object_make_transform(struct Object *object) {
        mat4 translate;
        glm_translate_make(translate, object->pos);
        glm_mat4_mul(translate, object->orientation, object->transform);
}

void get_init_data(struct Scene *data) {
        bzero(data, sizeof(struct Scene));
        data->count[0] = 2;
        // Cube 1
        data->objects[0].type[0] = 1;
        data->objects[0].pos[1] = 4;
        data->objects[0].linear_vel[1] = -0.001;
        glm_mat4_identity(data->objects[0].orientation);
        // data->objects[0].angular_vel[2] = 0.005;
        // data->objects[0].angular_vel[1] = 0.005;
        object_make_transform(&data->objects[0]);

        // Cube 2
        data->objects[1].type[0] = 1;
        glm_mat4_identity(data->objects[1].orientation);
        object_make_transform(&data->objects[1]);
}

// Adapted from https://varunagrawal.github.io/2020/02/11/fast-orthogonalization/
// Still works if `m` and `out` are the same matrix
void reorthogonalize(mat4 m, mat4 out) {
        vec3 x = {m[0][0], m[1][0], m[2][0]};
        vec3 y = {m[0][1], m[1][1], m[2][1]};

        float e = glm_dot(x, y);
        vec3 x_orth = {x[0] - 0.5 * e * y[0], x[1] - 0.5 * e * y[1], x[2] - 0.5 * e * y[2]};
        vec3 y_orth = {y[0] - 0.5 * e * x[0], y[1] - 0.5 * e * x[1], y[2] - 0.5 * e * x[2]};
        vec3 z_orth;
        glm_cross(x_orth, y_orth, z_orth);

        float x_dot = glm_dot(x_orth, x_orth);
        float y_dot = glm_dot(y_orth, y_orth);
        float z_dot = glm_dot(z_orth, z_orth);

        vec3 x_norm = {0.5 * (3 - x_dot) * x_orth[0], 0.5 * (3 - x_dot) * x_orth[1],
                       0.5 * (3 - x_dot) * x_orth[2]};
        vec3 y_norm = {0.5 * (3 - y_dot) * y_orth[0], 0.5 * (3 - y_dot) * y_orth[1],
                       0.5 * (3 - y_dot) * y_orth[2]};
        vec3 z_norm = {0.5 * (3 - z_dot) * z_orth[0], 0.5 * (3 - z_dot) * z_orth[1],
                       0.5 * (3 - z_dot) * z_orth[2]};

        bzero(out, sizeof(mat4));
        out[0][0] = x_norm[0];
        out[1][0] = x_norm[1];
        out[2][0] = x_norm[2];
        out[0][1] = y_norm[0];
        out[1][1] = y_norm[1];
        out[2][1] = y_norm[2];
        out[0][2] = z_norm[0];
        out[1][2] = z_norm[1];
        out[2][2] = z_norm[2];
        out[3][3] = 1;
}

// Call every time window is resized. Recreates everything, including depth pass and framebuffers
// and such.
void recreate_images(struct Base *base, VkRenderPass rpass, struct Swapchain *swapchain,
                     struct SyncSet sync_sets[CONCURRENT_FRAMES], struct Image *depth_image,
                     VkFramebuffer framebuffers[CONCURRENT_FRAMES],
                     VkFence image_fences[CONCURRENT_FRAMES]) {
        vkDeviceWaitIdle(base->device);

        VkFormat old_format = swapchain->format;
        uint32_t old_image_ct = swapchain->image_ct;

        swapchain_destroy(base->device, swapchain);
        swapchain_create(base->surface, base->phys_dev, base->device, old_format,
                         SC_PRESENT_MODE_PREF, swapchain);

        assert(swapchain->format == old_format && swapchain->image_ct == old_image_ct);

        for (int i = 0; i < CONCURRENT_FRAMES; i++) {
                sync_set_destroy(base->device, &sync_sets[i]);
                sync_set_create(base->device, &sync_sets[i]);
        }

        // Recreate depth
        // Only destroy if it actually exists (it's initialized to all NULL)
        if (depth_image->handle != NULL) {
                image_destroy(base->device, depth_image);
        }
        // Now actually recreate it
        image_create_depth(base->phys_dev, base->device, DEPTH_FORMAT, swapchain->width,
                           swapchain->height, VK_SAMPLE_COUNT_1_BIT, depth_image);

        // Recreate framebuffers
        for (int i = 0; i < swapchain->image_ct; i++) {
                if (framebuffers[i] != NULL) {
                        vkDestroyFramebuffer(base->device, framebuffers[i], NULL);
                }

                VkImageView views[] = {swapchain->views[i], depth_image->view};
                framebuffer_create(base->device, rpass, swapchain->width, swapchain->height,
                                   sizeof(views) / sizeof(views[0]), views, &framebuffers[i]);

                image_fences[i] = VK_NULL_HANDLE;
        }
}

// Will update last_mouse_{x,y}, camera position
void handle_input(GLFWwindow *window, double *last_mouse_x, double *last_mouse_y,
                  struct CameraFly *camera, struct Scene *scene_data, float delta,
                  int *open_debug) {
        // Mouse movement
        vec3 cam_movement = {0.0F, 0.0F, 0.0F};
        double new_mouse_x, new_mouse_y;
        glfwGetCursorPos(window, &new_mouse_x, &new_mouse_y);
        double d_mouse_x = new_mouse_x - *last_mouse_x, d_mouse_y = new_mouse_y - *last_mouse_y;
        *last_mouse_x = new_mouse_x;
        *last_mouse_y = new_mouse_y;

        // Camera keys
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

        if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
                // Keys to rotate cube
                if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS) {
                        glm_rotate(scene_data->objects[0].orientation, speed_multiplier * 0.001,
                                   (vec3){1, 0, 0});
                }
                if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS) {
                        glm_rotate(scene_data->objects[0].orientation, -speed_multiplier * 0.001,
                                   (vec3){1, 0, 0});
                }
                if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS) {
                        glm_rotate(scene_data->objects[0].orientation, speed_multiplier * 0.001,
                                   (vec3){0, 0, 1});
                }
                if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) {
                        glm_rotate(scene_data->objects[0].orientation, -speed_multiplier * 0.001,
                                   (vec3){0, 0, 1});
                }
                if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) {
                        glm_rotate(scene_data->objects[0].orientation, speed_multiplier * 0.001,
                                   (vec3){0, 1, 0});
                }
                if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS) {
                        glm_rotate(scene_data->objects[0].orientation, -speed_multiplier * 0.001,
                                   (vec3){0, 1, 0});
                }
        } else {
                // Keys to move cube around
                if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS) {
                        scene_data->objects[0].pos[2] += MOVEMENT_SPEED * speed_multiplier * 0.1;
                }
                if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS) {
                        scene_data->objects[0].pos[2] -= MOVEMENT_SPEED * speed_multiplier * 0.1;
                }
                if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS) {
                        scene_data->objects[0].pos[0] -= MOVEMENT_SPEED * speed_multiplier * 0.1;
                }
                if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) {
                        scene_data->objects[0].pos[0] += MOVEMENT_SPEED * speed_multiplier * 0.1;
                }
                if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) {
                        scene_data->objects[0].pos[1] += MOVEMENT_SPEED * speed_multiplier * 0.1;
                }
                if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS) {
                        scene_data->objects[0].pos[1] -= MOVEMENT_SPEED * speed_multiplier * 0.1;
                }
        }

        // Debug console
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
                *open_debug = 1;
        }

        // Update camera
        camera_fly_update(camera, d_mouse_x * MOUSE_SENSITIVITY_FACTOR,
                          d_mouse_y * MOUSE_SENSITIVITY_FACTOR, cam_movement, delta);
}

void physics_create(struct Base *base, VkDescriptorPool dpool, struct PhysicsEngine *physics) {
        // Makes debugging easier when I inevitably forget to init some field
        bzero(physics, sizeof(*physics));

        // Fences
        for (int i = 0; i < CONCURRENT_FRAMES; i++) {
                fence_create(base->device, 0, &physics->fences[i]);
        }

        // Command buffers
        for (int i = 0; i < CONCURRENT_FRAMES; i++) {
                cbuf_alloc(base->device, base->cpool, &physics->cbufs[i]);
        }

        // Shader
        VkShaderModule shader;
        VkPipelineShaderStageCreateInfo shader_info = {0};
        load_shader(base->device, "shaders_processed/compute.spv", &shader,
                    VK_SHADER_STAGE_COMPUTE_BIT, &shader_info);

        // Buffers
        for (int i = 0; i < CONCURRENT_FRAMES; i++) {
                // Compute shader takes Scene as input...
                buffer_create(base->phys_dev, base->device,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, sizeof(struct Scene),
                              &physics->in_bufs[i]);
                // And outputs ComputeOut.
                buffer_create(base->phys_dev, base->device,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, sizeof(struct ComputeOut),
                              &physics->out_bufs[i]);
        }

        // The accumulated force and torque needs to be reset each frame, and since all the compute
        // shaders run in parallel I think overwriting the old data with a zero buffer is the only
        // way to do it.
        buffer_create(base->phys_dev, base->device, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      sizeof(struct ComputeOut), &physics->reset);
        struct ComputeOut *reset_mapped;
        vkMapMemory(base->device, physics->reset.mem, 0, sizeof(struct ComputeOut), 0,
                    (void **)&reset_mapped);
        bzero(reset_mapped, sizeof(struct ComputeOut));
        vkUnmapMemory(base->device, physics->reset.mem);

        // This is so we can read data back out of the compute shader
        buffer_create(base->phys_dev, base->device, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      sizeof(struct ComputeOut), &physics->reader);

        // Descriptor shenanigans
        struct DescriptorInfo in_desc = {0};
        in_desc.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        in_desc.stage = VK_SHADER_STAGE_COMPUTE_BIT;

        struct DescriptorInfo out_desc = {0};
        out_desc.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        out_desc.stage = VK_SHADER_STAGE_COMPUTE_BIT;

        struct DescriptorInfo descs[] = {in_desc, out_desc};
        struct SetInfo set_info = {0};
        set_info.desc_ct = sizeof(descs) / sizeof(descs[0]);
        set_info.descs = descs;

        set_layout_create(base->device, &set_info, &physics->set_layout);

        // Make the actual sets
        for (int i = 0; i < CONCURRENT_FRAMES; i++) {
                // Set for compute
                union SetHandle buffers[2] = {0};
                buffers[0].buffer.buffer = physics->in_bufs[i].handle;
                buffers[0].buffer.range = sizeof(struct Scene);
                buffers[1].buffer.buffer = physics->out_bufs[i].handle;
                buffers[1].buffer.range = sizeof(struct ComputeOut);
                assert(sizeof(buffers) / sizeof(buffers[0]) == set_info.desc_ct);

                set_create(base->device, dpool, physics->set_layout, &set_info, buffers,
                           &physics->sets[i]);
        }

        // Pipeline
        // Layout
        VkPipelineLayoutCreateInfo pipe_layout_info = {0};
        pipe_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipe_layout_info.setLayoutCount = 1;
        pipe_layout_info.pSetLayouts = &physics->set_layout;

        VkResult res = vkCreatePipelineLayout(base->device, &pipe_layout_info, NULL,
                                              &physics->pipe_layout);
        assert(res == VK_SUCCESS);

        // Actual pipeline
        VkComputePipelineCreateInfo pipe_info = {0};
        pipe_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipe_info.layout = physics->pipe_layout;
        pipe_info.stage = shader_info;

        res = vkCreateComputePipelines(base->device, NULL, 1, &pipe_info, NULL, &physics->pipe);
        assert(res == VK_SUCCESS);

        vkDestroyShaderModule(base->device, shader, NULL);
}

// `idx` should be frame_ct % CONCURRENT_FRAMES, just like all the other per-frame resources. `in`
// should be a VkBuffer of `struct Scene`, which will be copied to compute shader input. Debug lines
// will be written into `debug`.
void physics_calc(VkDevice device, VkQueue queue, struct PhysicsEngine *physics, int idx,
                  VkBuffer scene, struct ComputeOut *out) {
        // Copy latest data to compute shader input
        buffer_copy(queue, physics->cbufs[idx], scene, physics->in_bufs[idx].handle,
                    sizeof(struct Scene));

        // Also reset compute buffer's output
        buffer_copy(queue, physics->cbufs[idx], physics->reset.handle,
                    physics->out_bufs[idx].handle, sizeof(struct ComputeOut));

        // Record compute dispatch
        vkResetCommandBuffer(physics->cbufs[idx], 0);
        cbuf_begin_onetime(physics->cbufs[idx]);
        vkCmdBindPipeline(physics->cbufs[idx], VK_PIPELINE_BIND_POINT_COMPUTE, physics->pipe);
        vkCmdBindDescriptorSets(physics->cbufs[idx], VK_PIPELINE_BIND_POINT_COMPUTE,
                                physics->pipe_layout, 0, 1, &physics->sets[idx], 0, NULL);
        vkCmdDispatch(physics->cbufs[idx], 20, 20, 20);
        VkResult res = vkEndCommandBuffer(physics->cbufs[idx]);
        assert(res == VK_SUCCESS);

        VkSubmitInfo compute_submit_info = {0};
        compute_submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        compute_submit_info.commandBufferCount = 1;
        compute_submit_info.pCommandBuffers = &physics->cbufs[idx];

        res = vkQueueSubmit(queue, 1, &compute_submit_info, physics->fences[idx]);
        assert(res == VK_SUCCESS);

        // Wait for compute to finish. This seems really wasteful.
        res = vkWaitForFences(device, 1, &physics->fences[idx], VK_TRUE, UINT64_MAX);
        assert(res == VK_SUCCESS);
        res = vkResetFences(device, 1, &physics->fences[idx]);
        assert(res == VK_SUCCESS);

        // Copy what the compute shader outputted to *out
        struct ComputeOut *mapped;
        static_assert(sizeof(*mapped) == sizeof(*out), "`out` and `mapped` must have same size!");
        buffer_copy(queue, physics->cbufs[idx], physics->out_bufs[idx].handle,
                    physics->reader.handle, sizeof(*mapped));
        vkMapMemory(device, physics->reader.mem, 0, sizeof(*mapped), 0, (void **)&mapped);
        memcpy(out, mapped, sizeof(*mapped));
        vkUnmapMemory(device, physics->reader.mem);
}

void physics_destroy(VkDevice device, struct PhysicsEngine *physics) {
        vkDestroyPipeline(device, physics->pipe, NULL);

        for (int i = 0; i < CONCURRENT_FRAMES; i++) {
                vkDestroyFence(device, physics->fences[i], NULL);
                buffer_destroy(device, &physics->in_bufs[i]);
                buffer_destroy(device, &physics->out_bufs[i]);
        }

        buffer_destroy(device, &physics->reader);
        buffer_destroy(device, &physics->reset);

        vkDestroyPipelineLayout(device, physics->pipe_layout, NULL);
        vkDestroyDescriptorSetLayout(device, physics->set_layout, NULL);
}

void physics_debug_create(struct Base *base, VkDescriptorPool dpool, struct PhysicsDebug *debug) {
        bzero(debug, sizeof(*debug));

        // Fence
        fence_create(base->device, 0, &debug->fence);

        // Command buffer
        cbuf_alloc(base->device, base->cpool, &debug->cbuf);

        // Shader
        VkShaderModule shader;
        VkPipelineShaderStageCreateInfo shader_info = {0};
        load_shader(base->device, "shaders_processed/compute_debug.spv", &shader,
                    VK_SHADER_STAGE_COMPUTE_BIT, &shader_info);

        // Buffers
        // Input
        buffer_create(base->phys_dev, base->device,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, sizeof(struct Scene), &debug->in_buf);

        // Output
        buffer_create(base->phys_dev, base->device,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, sizeof(struct ComputeDebugOut),
                      &debug->out_buf);

        // Reader
        buffer_create(base->phys_dev, base->device,
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      sizeof(struct ComputeDebugOut), &debug->reader);

        // Descriptor shenanigans
        struct DescriptorInfo in_desc = {0};
        in_desc.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        in_desc.stage = VK_SHADER_STAGE_COMPUTE_BIT;

        struct DescriptorInfo out_desc = {0};
        out_desc.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        out_desc.stage = VK_SHADER_STAGE_COMPUTE_BIT;

        struct DescriptorInfo descs[] = {in_desc, out_desc};
        struct SetInfo set_info = {0};
        set_info.desc_ct = sizeof(descs) / sizeof(descs[0]);
        set_info.descs = descs;

        set_layout_create(base->device, &set_info, &debug->set_layout);

        // Actual set
        union SetHandle buffers[2] = {0};
        buffers[0].buffer.buffer = debug->in_buf.handle;
        buffers[0].buffer.range = sizeof(struct Scene);
        buffers[1].buffer.buffer = debug->out_buf.handle;
        buffers[1].buffer.range = sizeof(struct ComputeDebugOut);
        assert(sizeof(buffers) / sizeof(buffers[0]) == set_info.desc_ct);

        set_create(base->device, dpool, debug->set_layout, &set_info, buffers, &debug->set);

        // Pipeline
        // Layout
        VkPipelineLayoutCreateInfo pipe_layout_info = {0};
        pipe_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipe_layout_info.setLayoutCount = 1;
        pipe_layout_info.pSetLayouts = &debug->set_layout;

        VkResult res =
                vkCreatePipelineLayout(base->device, &pipe_layout_info, NULL, &debug->pipe_layout);
        assert(res == VK_SUCCESS);

        // Actual pipeline
        VkComputePipelineCreateInfo pipe_info = {0};
        pipe_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipe_info.layout = debug->pipe_layout;
        pipe_info.stage = shader_info;

        res = vkCreateComputePipelines(base->device, NULL, 1, &pipe_info, NULL, &debug->pipe);
        assert(res == VK_SUCCESS);

        vkDestroyShaderModule(base->device, shader, NULL);
}

// For when it all goes horribly wrong, which is most of the time.
void physics_debug_calc(VkDevice device, VkQueue queue, struct PhysicsDebug *debug, VkBuffer scene,
                        struct ComputeDebugOut *out) {
        // Copy latest data to compute shader input
        buffer_copy(queue, debug->cbuf, scene, debug->in_buf.handle, sizeof(struct Scene));

        // Record compute dispatch
        vkResetCommandBuffer(debug->cbuf, 0);
        cbuf_begin_onetime(debug->cbuf);
        vkCmdBindPipeline(debug->cbuf, VK_PIPELINE_BIND_POINT_COMPUTE, debug->pipe);
        vkCmdBindDescriptorSets(debug->cbuf, VK_PIPELINE_BIND_POINT_COMPUTE, debug->pipe_layout, 0,
                                1, &debug->set, 0, NULL);
        vkCmdDispatch(debug->cbuf, 1, 1, 1);
        VkResult res = vkEndCommandBuffer(debug->cbuf);
        assert(res == VK_SUCCESS);

        VkSubmitInfo compute_submit_info = {0};
        compute_submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        compute_submit_info.commandBufferCount = 1;
        compute_submit_info.pCommandBuffers = &debug->cbuf;

        res = vkQueueSubmit(queue, 1, &compute_submit_info, debug->fence);
        assert(res == VK_SUCCESS);

        // Wait for compute to finish. This seems really wasteful.
        res = vkWaitForFences(device, 1, &debug->fence, VK_TRUE, UINT64_MAX);
        assert(res == VK_SUCCESS);
        res = vkResetFences(device, 1, &debug->fence);
        assert(res == VK_SUCCESS);

        // Copy what the compute shader outputted to *out
        struct ComputeDebugOut *mapped;
        static_assert(sizeof(*mapped) == sizeof(*out), "`out` and `mapped` must have same size!");
        buffer_copy(queue, debug->cbuf, debug->out_buf.handle, debug->reader.handle,
                    sizeof(*mapped));
        vkMapMemory(device, debug->reader.mem, 0, sizeof(*mapped), 0, (void **)&mapped);
        memcpy(out, mapped, sizeof(*mapped));
        vkUnmapMemory(device, debug->reader.mem);
}

void physics_debug_destroy(VkDevice device, struct PhysicsDebug *debug) {
        vkDestroyPipelineLayout(device, debug->pipe_layout, NULL);
        vkDestroyPipeline(device, debug->pipe, NULL);
        vkDestroyFence(device, debug->fence, NULL);
        buffer_destroy(device, &debug->in_buf);
        buffer_destroy(device, &debug->out_buf);
        buffer_destroy(device, &debug->reader);
        vkDestroyDescriptorSetLayout(device, debug->set_layout, NULL);
}

int main() {
        glfwInit();
        glfwSetErrorCallback(glfw_error_callback);
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        GLFWwindow *window = glfwCreateWindow(800, 600, "Vulkan", NULL, NULL);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        // Base
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_features = {0};
        atomic_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
        atomic_features.shaderBufferFloat32AtomicAdd = VK_TRUE;

        struct Base base;
        base_create(window, VK_API_VERSION_1_3, 1, 1, 0, NULL, DEVICE_EXT_CT, DEVICE_EXTS,
                    &atomic_features, &base);

        // Load font
        int font_width, font_height;
        uint8_t *font_bytes;
        load_font("font.txt", &font_width, &font_height, &font_bytes);

        // Copy font to image
        struct Image font_image;
        image_create(base.phys_dev, base.device, VK_FORMAT_R8_UNORM, VK_IMAGE_TYPE_2D, font_width,
                     font_height * FONT_CHAR_COUNT, 1, VK_IMAGE_TILING_OPTIMAL,
                     VK_IMAGE_ASPECT_COLOR_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 0, 1,
                     VK_SAMPLE_COUNT_1_BIT, &font_image);
        int font_byte_count = FONT_CHAR_COUNT * font_height * font_height;

        struct Buffer font_staging;
        buffer_create(base.phys_dev, base.device, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      font_byte_count, &font_staging);

        uint8_t *font_staging_mapped;
        VkResult res = vkMapMemory(base.device, font_staging.mem, 0, font_byte_count, 0,
                                   (void **)&font_staging_mapped);
        assert(res == VK_SUCCESS);
        memcpy(font_staging_mapped, font_bytes, font_byte_count);
        vkUnmapMemory(base.device, font_staging.mem);
        free(font_bytes);

        // Create sampler for font
        VkSamplerCreateInfo font_sampler_info = {0};
        font_sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        font_sampler_info.magFilter = VK_FILTER_NEAREST;
        font_sampler_info.minFilter = VK_FILTER_NEAREST;
        font_sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        font_sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        font_sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        font_sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        VkSampler font_sampler;
        res = vkCreateSampler(base.device, &font_sampler_info, 0, &font_sampler);
        assert(res == VK_SUCCESS);

        // Not sure if the access/stage flags are right
        // undefined -> transfer_dst_optimal
        image_trans(base.device, base.queue, base.cpool, font_image.handle,
                    VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 1);
        image_copy_from_buffer(base.device, base.queue, base.cpool, VK_IMAGE_ASPECT_COLOR_BIT,
                               font_staging.handle, font_image.handle, font_width,
                               font_height * FONT_CHAR_COUNT, 1);
        // transfer_dst_optimal -> attachment_optimal
        image_trans(base.device, base.queue, base.cpool, font_image.handle,
                    VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_INPUT_ATTACHMENT_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 1);

        buffer_destroy(base.device, &font_staging);

        // Swapchain
        struct Swapchain swapchain;
        swapchain_create(base.surface, base.phys_dev, base.device, SC_FORMAT_PREF,
                         SC_PRESENT_MODE_PREF, &swapchain);

        // Load shaders
        VkShaderModule graphics_vs, graphics_fs;
        VkPipelineShaderStageCreateInfo graphics_shaders[2] = {0};

        // Graphics shaders
        load_shader(base.device, "shaders_processed/graphics.vs.spv", &graphics_vs,
                    VK_SHADER_STAGE_VERTEX_BIT, &graphics_shaders[0]);
        load_shader(base.device, "shaders_processed/graphics.fs.spv", &graphics_fs,
                    VK_SHADER_STAGE_FRAGMENT_BIT, &graphics_shaders[1]);

        // Debug shaders
        VkShaderModule debug_vs, debug_fs;
        VkPipelineShaderStageCreateInfo debug_shaders[2] = {0};

        load_shader(base.device, "shaders_processed/debug.vs.spv", &debug_vs,
                    VK_SHADER_STAGE_VERTEX_BIT, &debug_shaders[0]);
        load_shader(base.device, "shaders_processed/debug.fs.spv", &debug_fs,
                    VK_SHADER_STAGE_FRAGMENT_BIT, &debug_shaders[1]);

        // Text shaders
        VkShaderModule text_vs, text_fs;
        VkPipelineShaderStageCreateInfo text_shaders[2] = {0};

        load_shader(base.device, "shaders_processed/text.vs.spv", &text_vs,
                    VK_SHADER_STAGE_VERTEX_BIT, &text_shaders[0]);
        load_shader(base.device, "shaders_processed/text.fs.spv", &text_fs,
                    VK_SHADER_STAGE_FRAGMENT_BIT, &text_shaders[1]);

        // Render pass
        VkRenderPass rpass;
        rpass_color_depth(base.device, swapchain.format, DEPTH_FORMAT, &rpass);

        // The debug arrows also need to be reset each frame
        struct Buffer debug_in_staging;
        buffer_create(base.phys_dev, base.device, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      sizeof(struct Debug), &debug_in_staging);
        struct Debug *debug_in_staging_mapped;
        vkMapMemory(base.device, debug_in_staging.mem, 0, sizeof(struct Debug), 0,
                    (void **)&debug_in_staging_mapped);
        bzero(debug_in_staging_mapped, sizeof(struct Debug));
        vkUnmapMemory(base.device, debug_in_staging.mem);

        // Staging buffer for graphics
        struct Buffer graphics_in_bufs[CONCURRENT_FRAMES];
        // Buffer for debug pass
        struct Buffer debug_in_buf;
        buffer_create(base.phys_dev, base.device,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, sizeof(struct Debug), &debug_in_buf);

        // Staging buffer for text pass (this is where we specify what characters we want to draw)
        struct Buffer text_in_bufs[CONCURRENT_FRAMES];
        struct Buffer text_staging;
        buffer_create(base.phys_dev, base.device, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      sizeof(struct Text), &text_staging);
        struct Text *text_staging_mapped;
        res = vkMapMemory(base.device, text_staging.mem, 0, sizeof(*text_staging_mapped), 0,
                          (void **)&text_staging_mapped);
        assert(res == VK_SUCCESS);
        text_staging_mapped->char_count = 4;
        text_staging_mapped->chars[4 * 0] = 1;
        text_staging_mapped->chars[4 * 1] = 2;
        text_staging_mapped->chars[4 * 2] = 3;
        text_staging_mapped->chars[4 * 3] = 4;

        // Initialize scene data
        struct Buffer scene_staging;
        buffer_create(base.phys_dev, base.device, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      sizeof(struct Scene), &scene_staging);

        struct Scene *scene_data;
        vkMapMemory(base.device, scene_staging.mem, 0, sizeof(struct Scene), 0,
                    (void **)&scene_data);
        get_init_data(scene_data);

        VkCommandBuffer copy_cbuf;
        cbuf_alloc(base.device, base.cpool, &copy_cbuf);

        // Actual storage buffers
        for (int i = 0; i < CONCURRENT_FRAMES; i++) {
                // Graphics shader takes Scene as input, we'll fill it with the latest data just
                // before drawing
                buffer_create(base.phys_dev, base.device,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, sizeof(struct Scene),
                              &graphics_in_bufs[i]);

                // We also might want to change text every frame
                buffer_create(base.phys_dev, base.device,
                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, sizeof(struct Text),
                              &text_in_bufs[i]);
        }

        // Descriptors
        struct DescriptorInfo graphics_in_desc = {0};
        graphics_in_desc.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        graphics_in_desc.stage = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        struct DescriptorInfo debug_in_desc = {0};
        debug_in_desc.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        debug_in_desc.stage = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT |
                              VK_SHADER_STAGE_FRAGMENT_BIT;

        struct DescriptorInfo text_image_desc = {0};
        text_image_desc.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        text_image_desc.stage = VK_SHADER_STAGE_FRAGMENT_BIT;

        struct DescriptorInfo text_uni_desc = {0};
        text_uni_desc.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        text_uni_desc.stage = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        // Create descriptor pool
        // Gotta be honest I have no idea what I'm doing here
        VkDescriptorPoolSize dpool_sizes[1] = {0};
        dpool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // I think it has to be 6 for each CONCURRENT_FRAME because there's the 4 descriptors for
        // the compute stage, 1 for the graphics stage, 1 for debug and 2 for text
        dpool_sizes[0].descriptorCount = CONCURRENT_FRAMES * 6;

        VkDescriptorPoolCreateInfo dpool_info = {0};
        dpool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpool_info.poolSizeCount = sizeof(dpool_sizes) / sizeof(dpool_sizes[0]);
        dpool_info.pPoolSizes = dpool_sizes;
        // Need 5 sets per CONCURRENT_FRAME because there's 2 for compute, 1 for graphics, 1 for
        // debug and 1 for text.
        dpool_info.maxSets = 5 * CONCURRENT_FRAMES;

        VkDescriptorPool dpool;
        res = vkCreateDescriptorPool(base.device, &dpool_info, NULL, &dpool);
        assert(res == VK_SUCCESS);

        // Now make the sets
        struct SetInfo graphics_set_info = {0};
        graphics_set_info.desc_ct = 1;
        graphics_set_info.descs = &graphics_in_desc;

        struct SetInfo debug_set_info = {0};
        debug_set_info.desc_ct = 1;
        debug_set_info.descs = &debug_in_desc;

        struct DescriptorInfo text_descs[] = {text_image_desc, text_uni_desc};
        struct SetInfo text_set_info = {0};
        text_set_info.desc_ct = sizeof(text_descs) / sizeof(text_descs[0]);
        text_set_info.descs = text_descs;

        VkDescriptorSetLayout graphics_set_layout;
        set_layout_create(base.device, &graphics_set_info, &graphics_set_layout);

        VkDescriptorSetLayout debug_set_layout;
        set_layout_create(base.device, &debug_set_info, &debug_set_layout);

        VkDescriptorSetLayout text_set_layout;
        set_layout_create(base.device, &text_set_info, &text_set_layout);

        VkDescriptorSet graphics_sets[CONCURRENT_FRAMES];
        VkDescriptorSet debug_sets[CONCURRENT_FRAMES];
        VkDescriptorSet text_sets[CONCURRENT_FRAMES];

        for (int i = 0; i < CONCURRENT_FRAMES; i++) {
                // Set for graphics
                union SetHandle graphics_buffers[1] = {0};
                graphics_buffers[0].buffer.buffer = graphics_in_bufs[i].handle;
                graphics_buffers[0].buffer.range = VK_WHOLE_SIZE;
                assert(sizeof(graphics_buffers) / sizeof(graphics_buffers[0]) ==
                       graphics_set_info.desc_ct);
                set_create(base.device, dpool, graphics_set_layout, &graphics_set_info,
                           graphics_buffers, &graphics_sets[i]);

                // Set for debug
                union SetHandle debug_buffers[1] = {0};
                debug_buffers[0].buffer.buffer = debug_in_buf.handle;
                debug_buffers[0].buffer.range = VK_WHOLE_SIZE;
                assert(sizeof(debug_buffers) / sizeof(debug_buffers[0]) == debug_set_info.desc_ct);
                set_create(base.device, dpool, debug_set_layout, &debug_set_info, debug_buffers,
                           &debug_sets[i]);

                // Sets for text
                union SetHandle text_handles[2] = {0};
                text_handles[0].image.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                text_handles[0].image.imageView = font_image.view;
                text_handles[0].image.sampler = font_sampler;
                text_handles[1].buffer.buffer = text_in_bufs[i].handle;
                text_handles[1].buffer.range = sizeof(struct Text);
                assert(sizeof(text_handles) / sizeof(text_handles[0]) == text_set_info.desc_ct);
                set_create(base.device, dpool, text_set_layout, &text_set_info, text_handles,
                           &text_sets[i]);
        }

        // Graphics pipeline
        // Layout
        VkPushConstantRange pushc_range = {0};
        pushc_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pushc_range.size = sizeof(struct PushConstants);

        VkPipelineLayoutCreateInfo graphics_pipe_layout_info = {0};
        graphics_pipe_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        graphics_pipe_layout_info.pushConstantRangeCount = 1;
        graphics_pipe_layout_info.pPushConstantRanges = &pushc_range;
        graphics_pipe_layout_info.setLayoutCount = 1;
        graphics_pipe_layout_info.pSetLayouts = &graphics_set_layout;

        VkPipelineLayout graphics_pipe_layout;
        res = vkCreatePipelineLayout(base.device, &graphics_pipe_layout_info, NULL,
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

        // Debug pipeline, almost exactly the same as graphics
        // Layout
        VkPipelineLayoutCreateInfo debug_pipe_layout_info = {0};
        debug_pipe_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        debug_pipe_layout_info.pushConstantRangeCount = 1;
        debug_pipe_layout_info.pPushConstantRanges = &pushc_range;
        debug_pipe_layout_info.setLayoutCount = 1;
        debug_pipe_layout_info.pSetLayouts = &debug_set_layout;

        VkPipelineLayout debug_pipe_layout;
        res = vkCreatePipelineLayout(base.device, &debug_pipe_layout_info, NULL,
                                     &debug_pipe_layout);
        assert(res == VK_SUCCESS);

        // Actual pipeline
        struct PipelineSettings debug_pipe_settings = PIPELINE_SETTINGS_DEFAULT;
        debug_pipe_settings.depth.depthTestEnable = VK_FALSE;
        debug_pipe_settings.rasterizer.cullMode = VK_CULL_MODE_NONE;
        debug_pipe_settings.input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

        VkPipeline debug_pipe;
        pipeline_create(base.device, &debug_pipe_settings,
                        sizeof(debug_shaders) / sizeof(debug_shaders[0]), debug_shaders,
                        debug_pipe_layout, rpass, 0, &debug_pipe);

        vkDestroyShaderModule(base.device, debug_vs, NULL);
        vkDestroyShaderModule(base.device, debug_fs, NULL);

        // Text pipeline
        // Layout
        VkPipelineLayoutCreateInfo text_pipe_layout_info = {0};
        text_pipe_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        text_pipe_layout_info.setLayoutCount = 1;
        text_pipe_layout_info.pSetLayouts = &text_set_layout;

        VkPipelineLayout text_pipe_layout;
        res = vkCreatePipelineLayout(base.device, &text_pipe_layout_info, NULL, &text_pipe_layout);
        assert(res == VK_SUCCESS);

        // Actual pipeline
        struct PipelineSettings text_pipe_settings = PIPELINE_SETTINGS_DEFAULT;
        text_pipe_settings.depth.depthTestEnable = VK_FALSE;
        text_pipe_settings.rasterizer.cullMode = VK_CULL_MODE_NONE;
        text_pipe_settings.input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipeline text_pipe;
        pipeline_create(base.device, &text_pipe_settings,
                        sizeof(text_shaders) / sizeof(text_shaders[0]), text_shaders,
                        text_pipe_layout, rpass, 0, &text_pipe);

        vkDestroyShaderModule(base.device, text_vs, NULL);
        vkDestroyShaderModule(base.device, text_fs, NULL);

        // Framebuffers, we'll create them later
        VkFramebuffer *framebuffers = malloc(swapchain.image_ct * sizeof(framebuffers[0]));
        bzero(framebuffers, swapchain.image_ct * sizeof(framebuffers[0]));
        int must_recreate = 1;

        struct Image depth_image = {0};

        // Command buffers
        VkCommandBuffer graphics_cbufs[CONCURRENT_FRAMES];
        for (int i = 0; i < CONCURRENT_FRAMES; i++) {
                cbuf_alloc(base.device, base.cpool, &graphics_cbufs[i]);
        }

        // Sync sets
        struct SyncSet sync_sets[CONCURRENT_FRAMES];
        for (int i = 0; i < CONCURRENT_FRAMES; i++) {
                sync_set_create(base.device, &sync_sets[i]);
        }

        // Image fences
        VkFence *image_fences = malloc(swapchain.image_ct * sizeof(image_fences[0]));
        for (int i = 0; i < swapchain.image_ct; i++) {
                image_fences[i] = VK_NULL_HANDLE;
        }

        // Physics
        struct PhysicsEngine physics = {0};
        physics_create(&base, dpool, &physics);

        struct PhysicsDebug physics_debug = {0};
        physics_debug_create(&base, dpool, &physics_debug);

        // Camera
        struct CameraFly camera;
        camera.pitch = 0.0F;
        camera.yaw = 0.0F;
        camera.eye[0] = 0.0F;
        camera.eye[1] = 0.0F;
        camera.eye[2] = -10.0F;
        double last_mouse_x, last_mouse_y;
        glfwGetCursorPos(window, &last_mouse_x, &last_mouse_y);

        // If this is true, debug console will open at the end of the frame
        int open_debug = 0;

        // For FPS running average
        float frametime_history[256] = {0};
        const int frametime_history_len = sizeof(frametime_history) / sizeof(frametime_history[0]);

        // Main loop
        int frame_ct = 0;
        int run_physics = 1;
        struct timespec start_time = timer_start();
        struct timespec last_frame_time = timer_start();

        while (!glfwWindowShouldClose(window)) {
                while (must_recreate) {
                        must_recreate = 0;
                        recreate_images(&base, rpass, &swapchain, sync_sets, &depth_image,
                                        framebuffers, image_fences);

                        int real_width, real_height;
                        glfwGetFramebufferSize(window, &real_width, &real_height);
                        if (real_width != swapchain.width || real_height != swapchain.height) {
                                must_recreate = 1;
                        }
                }

                // Handle input
                float delta = timer_get_elapsed(&last_frame_time);
                handle_input(window, &last_mouse_x, &last_mouse_y, &camera, scene_data, delta,
                             &open_debug);

                last_frame_time = timer_start();
                frametime_history[frame_ct % frametime_history_len] = delta;

                // Set up frame
                int frame_idx = frame_ct % CONCURRENT_FRAMES;
                struct SyncSet *sync_set = &sync_sets[frame_idx];

                VkCommandBuffer graphics_cbuf = graphics_cbufs[frame_idx];

                // Wait for the render process using these sync objects to finish rendering. There's
                // no need to explicitly wait for compute to finish because render waits for compute
                // anyway.
                res = vkWaitForFences(base.device, 1, &sync_set->render_fence, VK_TRUE, UINT64_MAX);
                assert(res == VK_SUCCESS);

                //////// begin physics
                int collision_count = 0;
                if (run_physics) {
                        struct ComputeOut compute_out;
                        physics_calc(base.device, base.queue, &physics, frame_idx,
                                     scene_staging.handle, &compute_out);

                        // Update object positions and debug input

                        // Apply impulse
                        collision_count = compute_out.collision_count;
                        // printf("col count: %u\n", collision_count);
                        if (collision_count > 0) {
                                scene_data->objects[0].linear_vel[0] +=
                                        compute_out.linear_impulse[0] / 1;
                                scene_data->objects[0].linear_vel[1] +=
                                        compute_out.linear_impulse[1] / 1;
                                scene_data->objects[0].linear_vel[2] +=
                                        compute_out.linear_impulse[2] / 1;

                                scene_data->objects[0].angular_vel[0] +=
                                        compute_out.angular_impulse[0] / 1;
                                scene_data->objects[0].angular_vel[1] +=
                                        compute_out.angular_impulse[1] / 1;
                                scene_data->objects[0].angular_vel[2] +=
                                        compute_out.angular_impulse[2] / 1;
                        }

                        // Integrate velocity to position
                        scene_data->objects[0].pos[0] += scene_data->objects[0].linear_vel[0];
                        scene_data->objects[0].pos[1] += scene_data->objects[0].linear_vel[1];
                        scene_data->objects[0].pos[2] += scene_data->objects[0].linear_vel[2];

                        // Apply angular velocity
                        vec3 omega;
                        memcpy(omega, scene_data->objects[0].angular_vel, sizeof(vec3));
                        mat4 omega_tilde = {{0, -omega[2], omega[1], 0},
                                            {omega[2], 0, -omega[0], 0},
                                            {-omega[1], omega[0], 0, 0},
                                            {0, 0, 0, 1}};
                        glm_mat4_transpose(omega_tilde);

                        mat4 derivative;
                        glm_mat4_mul(omega_tilde, scene_data->objects[0].transform, derivative);

                        for (int i = 0; i < 3; i++) {
                                for (int j = 0; j < 3; j++) {
                                        scene_data->objects[0].orientation[i][j] +=
                                                derivative[i][j];
                                }
                        }

                        reorthogonalize(scene_data->objects[0].orientation,
                                        scene_data->objects[0].orientation);

                        // Generate all transform matrices from position and orientation
                        for (int i = 0; i < scene_data->count[0]; i++) {
                                object_make_transform(&scene_data->objects[0]);
                        }
                }

                //////// end physics

                // Acquire an image
                uint32_t image_idx;
                res = vkAcquireNextImageKHR(base.device, swapchain.handle, UINT64_MAX,
                                            sync_set->acquire_sem, VK_NULL_HANDLE, &image_idx);
                if (res == VK_ERROR_OUT_OF_DATE_KHR) {
                        must_recreate = 1;
                        continue;
                } else {
                        assert(res == VK_SUCCESS);
                }

                // Copy new scene data to graphics input
                buffer_copy(base.queue, copy_cbuf, scene_staging.handle,
                            graphics_in_bufs[frame_idx].handle, sizeof(*scene_data));

                // Copy new text data to text input
                float frametime_sum = 0;
                for (int i = 0; i < frametime_history_len; i++) {
                        frametime_sum += frametime_history[i];
                }
                float fps_running_avg = frametime_history_len / frametime_sum;
                char text[256];
                sprintf(text, "FPS: %.1f | Collisions: %d", fps_running_avg, collision_count);
                text_write(text_staging_mapped, text, 0, -0.95, 0.02);
                buffer_copy(base.queue, copy_cbuf, text_staging.handle,
                            text_in_bufs[frame_idx].handle, sizeof(*text_staging_mapped));

                // Record graphics commands
                vkResetCommandBuffer(graphics_cbuf, 0);
                cbuf_begin_onetime(graphics_cbuf);

                VkClearValue clear_vals[] = {{{{0.0F, 0.0F, 0.0F, 1.0F}}}, {{{1.0F}}}};

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
                pushc_data.iMouse[0] = last_mouse_x;
                pushc_data.iMouse[1] = last_mouse_y;
                pushc_data.iMouse[2] =
                        glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
                pushc_data.iFrame = frame_ct;
                pushc_data.iTime = timer_get_elapsed(&start_time);
                memcpy(pushc_data.forward, camera.forward, sizeof(camera.forward));
                memcpy(pushc_data.eye, camera.eye, sizeof(camera.eye));
                pushc_data.dir[0] = camera.yaw;
                pushc_data.dir[1] = camera.pitch;
                memcpy(pushc_data.view, camera.view, sizeof(camera.view));
                glm_perspective(1.0F, (float)swapchain.width / (float)swapchain.height, 0.1F,
                                10000.0F, pushc_data.proj);

                vkCmdBindPipeline(graphics_cbuf, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipe);
                vkCmdPushConstants(graphics_cbuf, graphics_pipe_layout,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                   sizeof(struct PushConstants), &pushc_data);
                vkCmdBindDescriptorSets(graphics_cbuf, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        graphics_pipe_layout, 0, 1, &graphics_sets[frame_idx], 0,
                                        NULL);
                vkCmdDraw(graphics_cbuf, 36, MAX_OBJECTS, 0, 0);

                // Debug pass
                vkCmdBindPipeline(graphics_cbuf, VK_PIPELINE_BIND_POINT_GRAPHICS, debug_pipe);
                vkCmdPushConstants(graphics_cbuf, debug_pipe_layout,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                   sizeof(struct PushConstants), &pushc_data);
                vkCmdBindDescriptorSets(graphics_cbuf, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        debug_pipe_layout, 0, 1, &debug_sets[frame_idx], 0, NULL);

                vkCmdDraw(graphics_cbuf, 2, DEBUG_MAX_LINES, 0, 0);

                // Text pass
                vkCmdBindPipeline(graphics_cbuf, VK_PIPELINE_BIND_POINT_GRAPHICS, text_pipe);
                vkCmdBindDescriptorSets(graphics_cbuf, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        text_pipe_layout, 0, 1, &text_sets[frame_idx], 0, NULL);

                vkCmdDraw(graphics_cbuf, 6, 1, 0, 0);

                vkCmdEndRenderPass(graphics_cbuf);

                res = vkEndCommandBuffer(graphics_cbuf);
                assert(res == VK_SUCCESS);

                // Wait until whoever is rendering to the image is done
                if (image_fences[image_idx] != VK_NULL_HANDLE) {
                        vkWaitForFences(base.device, 1, &image_fences[image_idx], VK_TRUE,
                                        UINT64_MAX);
                }

                // Reset render fence
                res = vkResetFences(base.device, 1, &sync_set->render_fence);
                assert(res == VK_SUCCESS);

                // Mark it as in use by us
                image_fences[image_idx] = sync_set->render_fence;

                // Submit
                VkPipelineStageFlags wait_stages[] = {
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
                VkSemaphore wait_sems[] = {sync_set->acquire_sem};
                static_assert(sizeof(wait_stages) / sizeof(wait_stages[0]) ==
                                      sizeof(wait_sems) / sizeof(wait_sems[0]),
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
                if (res == VK_ERROR_OUT_OF_DATE_KHR) {
                        must_recreate = 1;
                } else {
                        assert(res == VK_SUCCESS);
                }

                frame_ct++;

                // Maybe open debug
                if (open_debug) {
                        printf("c: continue rendering\n");
                        printf("P: toggle physics\n");
                        printf("n: step to next frame\n");
                        printf("d: debug stuff\n");

                        while (1) {
                                char command;
                                if (scanf(" %c", &command) != 1) {
                                        printf("Couldn't read char\n");
                                        continue;
                                }

                                if (command == 'c') {
                                        printf("Continue\n");
                                        open_debug = 0;
                                        break;
                                } else if (command == 'n') {
                                        break;
                                } else if (command == 'P') {
                                        run_physics = !run_physics;
                                        printf("Run physics: %d\n", run_physics);
                                } else if (command == 'd') {
                                        struct ComputeDebugOut debug_out;
                                        physics_debug_calc(base.device, base.queue, &physics_debug,
                                                           scene_staging.handle, &debug_out);
                                        printf("A's COM: %5.2f %5.2f %5.2f\n", debug_out.a_com[0],
                                               debug_out.a_com[1], debug_out.a_com[2]);
                                        printf("B's COM: %5.2f %5.2f %5.2f\n", debug_out.b_com[0],
                                               debug_out.b_com[1], debug_out.b_com[2]);
                                } else {
                                        printf("Don't know what %c is\n", command);
                                }
                        }
                }

                glfwPollEvents();
        }

        double elapsed = timer_get_elapsed(&start_time);
        double fps = (double)frame_ct / elapsed;
        printf("FPS: %.2f (%.2fms), total frames: %d\n", fps, elapsed / frame_ct * 1000, frame_ct);

        vkDeviceWaitIdle(base.device);

        physics_destroy(base.device, &physics);
        physics_debug_destroy(base.device, &physics_debug);

        vkDestroySampler(base.device, font_sampler, 0);

        vkDestroyPipelineLayout(base.device, graphics_pipe_layout, NULL);
        vkDestroyPipelineLayout(base.device, debug_pipe_layout, NULL);
        vkDestroyPipelineLayout(base.device, text_pipe_layout, NULL);

        vkDestroyPipeline(base.device, graphics_pipe, NULL);
        vkDestroyPipeline(base.device, debug_pipe, NULL);
        vkDestroyPipeline(base.device, text_pipe, NULL);

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

        vkDestroyDescriptorSetLayout(base.device, graphics_set_layout, NULL);
        vkDestroyDescriptorSetLayout(base.device, debug_set_layout, NULL);
        vkDestroyDescriptorSetLayout(base.device, text_set_layout, NULL);

        buffer_destroy(base.device, &scene_staging);
        buffer_destroy(base.device, &debug_in_staging);
        buffer_destroy(base.device, &debug_in_buf);
        buffer_destroy(base.device, &text_staging);
        for (int i = 0; i < CONCURRENT_FRAMES; i++) {
                buffer_destroy(base.device, &graphics_in_bufs[i]);
                buffer_destroy(base.device, &text_in_bufs[i]);
        }

        image_destroy(base.device, &font_image);

        base_destroy(&base);

        glfwTerminate();

        free(framebuffers);
        free(image_fences);
}
