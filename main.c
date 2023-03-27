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
        uint32_t debug_out_idx;
};

// Gets passed to debug pass. Yes, I know I could have just used a vertex buffer, but this is more
// flexible if I want to include more stuff later. It's a debug layer, performance is less
// important.
struct __attribute__((packed, aligned(16))) Debug {
        vec4 line_poss[DEBUG_MAX_LINES]; // vec3[]
        vec4 line_dirs[DEBUG_MAX_LINES]; // vec3[]
};

// This stuff exists for every concurrent frame
struct SyncSet {
        VkFence render_fence;
        // Gotta wait for this before we can add up collision vectors
        VkFence compute_fence;
        VkSemaphore acquire_sem;
        // Signalled when compute shader finishes. Graphics submission waits on this and
        // acquire_sem.
        VkSemaphore compute_sem;
        VkSemaphore render_sem;
};

void sync_set_create(VkDevice device, struct SyncSet *sync_set) {
        fence_create(device, VK_FENCE_CREATE_SIGNALED_BIT, &sync_set->render_fence);
        // Shouldn't be signalled initially because we always dispatch before waiting, we never wait
        // on the compute from previous frame
        fence_create(device, 0, &sync_set->compute_fence);
        semaphore_create(device, &sync_set->acquire_sem);
        semaphore_create(device, &sync_set->compute_sem);
        semaphore_create(device, &sync_set->render_sem);
}

void sync_set_destroy(VkDevice device, struct SyncSet *sync_set) {
        vkDestroyFence(device, sync_set->render_fence, NULL);
        vkDestroyFence(device, sync_set->compute_fence, NULL);
        vkDestroySemaphore(device, sync_set->acquire_sem, NULL);
        vkDestroySemaphore(device, sync_set->compute_sem, NULL);
        vkDestroySemaphore(device, sync_set->render_sem, NULL);
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
        data->objects[0].type[0] = 5;
        data->objects[0].pos[1] = 4;
        glm_mat4_identity(data->objects[0].orientation);
        data->objects[0].angular_vel[2] = 0.005;
        data->objects[0].angular_vel[1] = 0.005;
        object_make_transform(&data->objects[0]);

        // Cube 2
        data->objects[1].type[0] = 2;
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
                     struct SyncSet sync_sets[CONCURRENT_FRAMES], struct Image* depth_image,
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

        // Compute shader
        VkShaderModule compute_shader;
        VkPipelineShaderStageCreateInfo compute_shader_info = {0};
        load_shader(base.device, "shaders_processed/compute.spv", &compute_shader,
                    VK_SHADER_STAGE_COMPUTE_BIT, &compute_shader_info);

        // Debug shaders
        VkShaderModule debug_vs, debug_fs;
        VkPipelineShaderStageCreateInfo debug_shaders[2] = {0};

        // Debug shaders
        load_shader(base.device, "shaders_processed/debug.vs.spv", &debug_vs,
                    VK_SHADER_STAGE_VERTEX_BIT, &debug_shaders[0]);
        load_shader(base.device, "shaders_processed/debug.fs.spv", &debug_fs,
                    VK_SHADER_STAGE_FRAGMENT_BIT, &debug_shaders[1]);

        // Render pass
        VkRenderPass rpass;
        rpass_color_depth(base.device, swapchain.format, DEPTH_FORMAT, &rpass);

        // Buffers for compute
        struct Buffer compute_in_bufs[CONCURRENT_FRAMES], compute_out_bufs[CONCURRENT_FRAMES],
                compute_buf_reader;
        // The accumulated force and torque needs to be reset each frame, and since all the compute
        // shaders run in parallel I think overwriting the old data with a zero buffer is the only
        // way to do it.
        struct Buffer compute_out_staging;
        buffer_create(base.phys_dev, base.device, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      sizeof(struct ComputeOut), &compute_out_staging);
        struct ComputeOut *compute_out_staging_mapped;
        vkMapMemory(base.device, compute_out_staging.mem, 0, sizeof(struct ComputeOut), 0,
                    (void **)&compute_out_staging_mapped);
        bzero(compute_out_staging_mapped, sizeof(struct ComputeOut));
        vkUnmapMemory(base.device, compute_out_staging.mem);

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

        // Buffer for graphics
        struct Buffer graphics_in_bufs[CONCURRENT_FRAMES];
        struct Buffer scene_staging;
        buffer_create(base.phys_dev, base.device, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      sizeof(struct Scene), &scene_staging);

        // Buffer for debug pass
        struct Buffer debug_in_buf;
        buffer_create(base.phys_dev, base.device,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, sizeof(struct Debug), &debug_in_buf);

        // Initialize scene staging
        struct Scene *scene_data;
        vkMapMemory(base.device, scene_staging.mem, 0, sizeof(struct Scene), 0,
                    (void **)&scene_data);
        get_init_data(scene_data);

        VkCommandBuffer copy_cbuf;
        cbuf_alloc(base.device, base.cpool, &copy_cbuf);

        // Actual storage buffers
        for (int i = 0; i < CONCURRENT_FRAMES; i++) {
                // Compute shader takes Scene as input...
                buffer_create(base.phys_dev, base.device,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, sizeof(struct Scene),
                              &compute_in_bufs[i]);
                // And outputs ComputeOut.
                buffer_create(base.phys_dev, base.device,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, sizeof(struct ComputeOut),
                              &compute_out_bufs[i]);

                // Graphics shader takes Scene as input, we'll fill it with the latest data just
                // before drawing
                buffer_create(base.phys_dev, base.device,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, sizeof(struct Scene),
                              &graphics_in_bufs[i]);
        }

        // This is so we can read data back out
        buffer_create(base.phys_dev, base.device, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      sizeof(struct ComputeOut), &compute_buf_reader);

        // Descriptors
        struct DescriptorInfo compute_in_desc = {0};
        compute_in_desc.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        compute_in_desc.stage = VK_SHADER_STAGE_COMPUTE_BIT;

        struct DescriptorInfo compute_out_desc = {0};
        compute_out_desc.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        compute_out_desc.stage = VK_SHADER_STAGE_COMPUTE_BIT;

        struct DescriptorInfo graphics_in_desc = {0};
        graphics_in_desc.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        graphics_in_desc.stage = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        struct DescriptorInfo debug_in_desc = {0};
        debug_in_desc.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        debug_in_desc.stage = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT |
                              VK_SHADER_STAGE_FRAGMENT_BIT;

        // Create descriptor pool
        // Gotta be honest I have no idea what I'm doing here
        VkDescriptorPoolSize dpool_sizes[1] = {0};
        dpool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // I think it has to be 5 for each CONCURRENT_FRAME because there's the 3 descriptors for
        // the compute stage, 1 for the graphics stage and 1 more for debug
        dpool_sizes[0].descriptorCount = CONCURRENT_FRAMES * 5;

        VkDescriptorPoolCreateInfo dpool_info = {0};
        dpool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpool_info.poolSizeCount = sizeof(dpool_sizes) / sizeof(dpool_sizes[0]);
        dpool_info.pPoolSizes = dpool_sizes;
        // Need 3 sets per CONCURRENT_FRAME because there's 1 for compute, 1 for graphics and 1 for
        // debug.
        dpool_info.maxSets = 3 * CONCURRENT_FRAMES;

        VkDescriptorPool dpool;
        VkResult res = vkCreateDescriptorPool(base.device, &dpool_info, NULL, &dpool);
        assert(res == VK_SUCCESS);

        // Now make the sets
        struct DescriptorInfo compute_descs[] = {compute_in_desc, compute_out_desc, debug_in_desc};
        struct SetInfo compute_set_info = {0};
        compute_set_info.desc_ct = sizeof(compute_descs) / sizeof(compute_descs[0]);
        compute_set_info.descs = compute_descs;

        struct SetInfo graphics_set_info = {0};
        graphics_set_info.desc_ct = 1;
        graphics_set_info.descs = &graphics_in_desc;

        struct SetInfo debug_set_info = {0};
        debug_set_info.desc_ct = 1;
        debug_set_info.descs = &debug_in_desc;

        VkDescriptorSetLayout compute_set_layout;
        set_layout_create(base.device, &compute_set_info, &compute_set_layout);

        VkDescriptorSetLayout graphics_set_layout;
        set_layout_create(base.device, &graphics_set_info, &graphics_set_layout);

        VkDescriptorSetLayout debug_set_layout;
        set_layout_create(base.device, &debug_set_info, &debug_set_layout);

        VkDescriptorSet compute_sets[CONCURRENT_FRAMES];
        VkDescriptorSet graphics_sets[CONCURRENT_FRAMES];
        VkDescriptorSet debug_sets[CONCURRENT_FRAMES];

        for (int i = 0; i < CONCURRENT_FRAMES; i++) {
                // Set for compute
                union SetHandle compute_buffers[3] = {0};
                compute_buffers[0].buffer.buffer = compute_in_bufs[i].handle;
                compute_buffers[0].buffer.range = VK_WHOLE_SIZE;
                compute_buffers[1].buffer.buffer = compute_out_bufs[i].handle;
                compute_buffers[1].buffer.range = VK_WHOLE_SIZE;
                compute_buffers[2].buffer.buffer = debug_in_buf.handle;
                compute_buffers[2].buffer.range = sizeof(struct Debug);
                assert(sizeof(compute_buffers) / sizeof(compute_buffers[0]) ==
                       compute_set_info.desc_ct);

                set_create(base.device, dpool, compute_set_layout, &compute_set_info,
                           compute_buffers, &compute_sets[i]);

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
        res = vkCreateComputePipelines(base.device, NULL, 1, &compute_pipe_info, NULL,
                                       &compute_pipe);
        assert(res == VK_SUCCESS);

        vkDestroyShaderModule(base.device, compute_shader, NULL);

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

        // Framebuffers, we'll create them later
        VkFramebuffer *framebuffers = malloc(swapchain.image_ct * sizeof(framebuffers[0]));
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
        struct SyncSet sync_sets[CONCURRENT_FRAMES];
        for (int i = 0; i < CONCURRENT_FRAMES; i++) {
                sync_set_create(base.device, &sync_sets[i]);
        }

        // Image fences
        VkFence *image_fences = malloc(swapchain.image_ct * sizeof(image_fences[0]));
        for (int i = 0; i < swapchain.image_ct; i++) {
                image_fences[i] = VK_NULL_HANDLE;
        }

        // Camera
        struct CameraFly camera;
        camera.pitch = 0.0F;
        camera.yaw = 0.0F;
        camera.eye[0] = 0.0F;
        camera.eye[1] = 0.0F;
        camera.eye[2] = -10.0F;
        double last_mouse_x, last_mouse_y;
        glfwGetCursorPos(window, &last_mouse_x, &last_mouse_y);

        // Every frame, we read the previous frame's compute output to integrate velocities and
        // positions
        struct ComputeOut *compute_out_mapped;
        res = vkMapMemory(base.device, compute_buf_reader.mem, 0, sizeof(struct ComputeOut), 0,
                          (void **)&compute_out_mapped);
        assert(res == VK_SUCCESS);
        // Should all be 0 initially
        bzero(compute_out_mapped, sizeof(struct ComputeOut));

        // Main loop
        int frame_ct = 0;
        struct timespec start_time = timer_start();
        struct timespec last_frame_time = timer_start();
        double total_collision_time = 0;

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
                // Mouse movement
                double new_mouse_x, new_mouse_y;
                glfwGetCursorPos(window, &new_mouse_x, &new_mouse_y);
                double d_mouse_x = new_mouse_x - last_mouse_x,
                       d_mouse_y = new_mouse_y - last_mouse_y;
                double delta = timer_get_elapsed(&last_frame_time);
                last_frame_time = timer_start(last_frame_time);
                last_mouse_x = new_mouse_x;
                last_mouse_y = new_mouse_y;

                // Camera keys
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

                if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
                        // Keys to rotate cube
                        if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS) {
                                glm_rotate(scene_data->objects[0].orientation,
                                           speed_multiplier * 0.001, (vec3){1, 0, 0});
                        }
                        if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS) {
                                glm_rotate(scene_data->objects[0].orientation,
                                           -speed_multiplier * 0.001, (vec3){1, 0, 0});
                        }
                        if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS) {
                                glm_rotate(scene_data->objects[0].orientation,
                                           speed_multiplier * 0.001, (vec3){0, 0, 1});
                        }
                        if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) {
                                glm_rotate(scene_data->objects[0].orientation,
                                           -speed_multiplier * 0.001, (vec3){0, 0, 1});
                        }
                        if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) {
                                glm_rotate(scene_data->objects[0].orientation,
                                           speed_multiplier * 0.001, (vec3){0, 1, 0});
                        }
                        if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS) {
                                glm_rotate(scene_data->objects[0].orientation,
                                           -speed_multiplier * 0.001, (vec3){0, 1, 0});
                        }
                } else {
                        // Keys to move cube around
                        if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS) {
                                scene_data->objects[0].pos[2] +=
                                        MOVEMENT_SPEED * speed_multiplier * 0.1;
                        }
                        if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS) {
                                scene_data->objects[0].pos[2] -=
                                        MOVEMENT_SPEED * speed_multiplier * 0.1;
                        }
                        if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS) {
                                scene_data->objects[0].pos[0] -=
                                        MOVEMENT_SPEED * speed_multiplier * 0.1;
                        }
                        if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) {
                                scene_data->objects[0].pos[0] +=
                                        MOVEMENT_SPEED * speed_multiplier * 0.1;
                        }
                        if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) {
                                scene_data->objects[0].pos[1] +=
                                        MOVEMENT_SPEED * speed_multiplier * 0.1;
                        }
                        if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS) {
                                scene_data->objects[0].pos[1] -=
                                        MOVEMENT_SPEED * speed_multiplier * 0.1;
                        }
                }

                // Update camera
                camera_fly_update(&camera, d_mouse_x * MOUSE_SENSITIVITY_FACTOR,
                                  d_mouse_y * MOUSE_SENSITIVITY_FACTOR, cam_movement, delta);

                // Set up frame
                int frame_idx = frame_ct % CONCURRENT_FRAMES;
                struct SyncSet *sync_set = &sync_sets[frame_idx];

                VkCommandBuffer graphics_cbuf = graphics_cbufs[frame_idx],
                                compute_cbuf = compute_cbufs[frame_idx];

                // Wait for the render process using these sync objects to finish rendering. There's
                // no need to explicitly wait for compute to finish because render waits for compute
                // anyway.
                res = vkWaitForFences(base.device, 1, &sync_set->render_fence, VK_TRUE, UINT64_MAX);
                assert(res == VK_SUCCESS);

                // Integrate acceleration to velocity
                vec3 linear_accel = {0, 0, 0};
                vec3 angular_accel = {0, 0, 0};
                memcpy(linear_accel, compute_out_mapped->force, sizeof(vec3));
                memcpy(angular_accel, compute_out_mapped->torque, sizeof(vec3));

                linear_accel[0] *= 0.000001;
                linear_accel[1] *= 0.000001;
                linear_accel[2] *= 0.000001;

                angular_accel[0] *= 0.00002;
                angular_accel[1] *= 0.00002;
                angular_accel[2] *= 0.00002;

                scene_data->objects[0].linear_vel[0] += linear_accel[0];
                scene_data->objects[0].linear_vel[1] += linear_accel[1];
                scene_data->objects[0].linear_vel[2] += linear_accel[2];

                scene_data->objects[0].angular_vel[0] += angular_accel[0];
                scene_data->objects[0].angular_vel[1] += angular_accel[1];
                scene_data->objects[0].angular_vel[2] += angular_accel[2];

                // Apply gravity, but only to first object
                scene_data->objects[0].linear_vel[1] -= 0.0001;

                // Apply world's simplest drag model
                /*
                scene_data->objects[0].linear_vel[0] *= 0.99;
                scene_data->objects[0].linear_vel[1] *= 0.99;
                scene_data->objects[0].linear_vel[2] *= 0.99;

                scene_data->objects[0].angular_vel[0] *= 0.99;
                scene_data->objects[0].angular_vel[1] *= 0.99;
                scene_data->objects[0].angular_vel[2] *= 0.99;
                */

                // Copy latest data to compute shader input
                buffer_copy(base.queue, copy_cbuf, scene_staging.handle,
                            compute_in_bufs[frame_idx].handle, sizeof(struct Scene));
                // Also reset compute buffer's output
                buffer_copy(base.queue, copy_cbuf, compute_out_staging.handle,
                            compute_out_bufs[frame_idx].handle, sizeof(struct ComputeOut));
                // And debug buffer's input, which compute outputs to. But only every once in a
                // while, otherwise the arrows don't appear because the framerate is so high.
                if (frame_ct % 16 == 0) {
                        buffer_copy(base.queue, copy_cbuf, debug_in_staging.handle,
                                    debug_in_buf.handle, sizeof(struct Debug));
                }

                // Record compute dispatch
                vkResetCommandBuffer(compute_cbuf, 0);
                cbuf_begin_onetime(compute_cbuf);
                vkCmdBindPipeline(compute_cbuf, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipe);
                vkCmdBindDescriptorSets(compute_cbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        compute_pipe_layout, 0, 1, &compute_sets[frame_idx], 0,
                                        NULL);
                vkCmdDispatch(compute_cbuf, 20, 20, 20);
                res = vkEndCommandBuffer(compute_cbuf);
                assert(res == VK_SUCCESS);

                VkSubmitInfo compute_submit_info = {0};
                compute_submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                compute_submit_info.commandBufferCount = 1;
                compute_submit_info.pCommandBuffers = &compute_cbuf;
                compute_submit_info.signalSemaphoreCount = 1;
                compute_submit_info.pSignalSemaphores = &sync_set->compute_sem;

                res = vkQueueSubmit(base.queue, 1, &compute_submit_info, sync_set->compute_fence);
                assert(res == VK_SUCCESS);

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

                // Before recording graphics, wait for compute to finish
                // I think I could combine this with the copy more efficiently with a barrier
                // This is so painfully inefficient it's not even funny
                res = vkWaitForFences(base.device, 1, &sync_set->compute_fence, VK_TRUE,
                                      UINT64_MAX);
                assert(res == VK_SUCCESS);
                res = vkResetFences(base.device, 1, &sync_set->compute_fence);
                assert(res == VK_SUCCESS);

                // Copy what the compute shader outputted to a CPU-visible buffer
                buffer_copy(base.queue, compute_cbuf, compute_out_bufs[frame_idx].handle,
                            compute_buf_reader.handle, sizeof(struct ComputeOut));
                // We only use the results at the beginning of next frame, apart from applying
                // impulse below

                // Update object positions and debug input
                struct timespec start_time = timer_start();

                // Apply impulse
                /*
                printf("Total linear impulse: %5.2f %5.2f %5.2f\n",
                       compute_out_mapped->linear_impulse[0], compute_out_mapped->linear_impulse[1],
                       compute_out_mapped->linear_impulse[2]);
                */
                uint32_t col_count = compute_out_mapped->collision_count;
                printf("col count: %u\n", col_count);
                if (col_count > 0) {
                        scene_data->objects[0].linear_vel[0] +=
                                compute_out_mapped->linear_impulse[0] / col_count;
                        scene_data->objects[0].linear_vel[1] +=
                                compute_out_mapped->linear_impulse[1] / col_count;
                        scene_data->objects[0].linear_vel[2] +=
                                compute_out_mapped->linear_impulse[2] / col_count;

                        scene_data->objects[0].angular_vel[0] +=
                                compute_out_mapped->angular_impulse[0] / col_count;
                        scene_data->objects[0].angular_vel[1] +=
                                compute_out_mapped->angular_impulse[1] / col_count;
                        scene_data->objects[0].angular_vel[2] +=
                                compute_out_mapped->angular_impulse[2] / col_count;
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

                mat4 derivative;
                glm_mat4_mul(omega_tilde, scene_data->objects[0].transform, derivative);

                for (int i = 0; i < 3; i++) {
                        for (int j = 0; j < 3; j++) {
                                scene_data->objects[0].orientation[i][j] -= derivative[i][j];
                        }
                }

                reorthogonalize(scene_data->objects[0].orientation,
                                scene_data->objects[0].orientation);

                // Generate all transform matrices from position and orientation
                for (int i = 0; i < scene_data->count[0]; i++) {
                        object_make_transform(&scene_data->objects[0]);
                }

                total_collision_time += timer_get_elapsed(&start_time);

                /*
                printf("New pos: %5.2f %5.2f %5.2f\n", scene_data->objects[0].transform[3][0],
                       scene_data->objects[0].transform[3][1],
                       scene_data->objects[0].transform[3][2]);
                */

                // Copy new scene data to graphics input
                buffer_copy(base.queue, copy_cbuf, scene_staging.handle,
                            graphics_in_bufs[frame_idx].handle, sizeof(struct Scene));

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
                pushc_data.iMouse[0] = new_mouse_x;
                pushc_data.iMouse[1] = new_mouse_y;
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
                VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                                      VK_PIPELINE_STAGE_VERTEX_SHADER_BIT};
                VkSemaphore wait_sems[] = {sync_set->acquire_sem, sync_set->compute_sem};
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
                glfwPollEvents();
        }

        double elapsed = timer_get_elapsed(&start_time);
        double fps = (double)frame_ct / elapsed;
        printf("FPS: %.2f (%.2fms), total frames: %d\n", fps, elapsed / frame_ct * 1000, frame_ct);
        printf("Average collision time: %.2f ms\n", total_collision_time / frame_ct * 1000);

        vkDeviceWaitIdle(base.device);

        vkDestroyPipelineLayout(base.device, graphics_pipe_layout, NULL);
        vkDestroyPipelineLayout(base.device, compute_pipe_layout, NULL);
        vkDestroyPipelineLayout(base.device, debug_pipe_layout, NULL);

        vkDestroyPipeline(base.device, graphics_pipe, NULL);
        vkDestroyPipeline(base.device, compute_pipe, NULL);
        vkDestroyPipeline(base.device, debug_pipe, NULL);

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
        vkDestroyDescriptorSetLayout(base.device, debug_set_layout, NULL);

        buffer_destroy(base.device, &scene_staging);
        buffer_destroy(base.device, &compute_buf_reader);
        buffer_destroy(base.device, &compute_out_staging);
        buffer_destroy(base.device, &debug_in_staging);
        buffer_destroy(base.device, &debug_in_buf);
        for (int i = 0; i < CONCURRENT_FRAMES; i++) {
                buffer_destroy(base.device, &compute_in_bufs[i]);
                buffer_destroy(base.device, &compute_out_bufs[i]);
                buffer_destroy(base.device, &graphics_in_bufs[i]);
        }

        base_destroy(&base);

        glfwTerminate();

        free(framebuffers);
        free(image_fences);
}
