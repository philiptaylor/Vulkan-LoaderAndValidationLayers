/* Copyright (c) 2016 Philip Taylor
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and/or associated documentation files (the "Materials"), to
 * deal in the Materials without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Materials, and to permit persons to whom the Materials
 * are furnished to do so, subject to the following conditions:
 *
 * The above copyright notice(s) and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 *
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE MATERIALS OR THE
 * USE OR OTHER DEALINGS IN THE MATERIALS
 */

#ifndef INCLUDED_VULKAN_SYNC_H
#define INCLUDED_VULKAN_SYNC_H

#include "vk_loader_platform.h"
#include "vk_layer.h"

#include <map>
#include <set>
#include <vector>
#include <memory>

enum sync_msg
{
    // General non-error messages
    SYNC_MSG_NONE,

    // Invalid parameters passed in by application
    SYNC_MSG_INVALID_PARAM,

    // Indicates a bug in this layer or in lower layers of the implementation
    SYNC_MSG_INTERNAL_ERROR,
};

enum class sync_command_buffer_state
{
    INITIAL,
    RECORDING,
    EXECUTABLE,
};

class sync_cmd_base
{
public:
    virtual ~sync_cmd_base() { }

    virtual bool is_draw() const { return false; }

    virtual void to_string(std::ostream &str) = 0;
};

class sync_cmd_bind_pipeline : public sync_cmd_base
{
public:
    sync_cmd_bind_pipeline(
        VkPipelineBindPoint pipelineBindPoint,
        VkPipeline pipeline) :
        pipelineBindPoint(pipelineBindPoint),
        pipeline(pipeline)
    {
    }

    virtual void to_string(std::ostream &str) override;

    VkPipelineBindPoint pipelineBindPoint;
    VkPipeline pipeline;
};

class sync_cmd_set_viewport : public sync_cmd_base
{
public:
    sync_cmd_set_viewport(
        uint32_t firstViewport,
        uint32_t viewportCount,
        const VkViewport *pViewports) :
        firstViewport(firstViewport),
        viewports(pViewports, pViewports + viewportCount)
    {
    }

    virtual void to_string(std::ostream &str) override;

    uint32_t firstViewport;
    std::vector<VkViewport> viewports;
};

class sync_cmd_set_scissor : public sync_cmd_base
{
public:
    sync_cmd_set_scissor(
        uint32_t firstScissor,
        uint32_t scissorCount,
        const VkRect2D *pScissors) :
        firstScissor(firstScissor),
        scissors(pScissors, pScissors + scissorCount)
    {
    }

    virtual void to_string(std::ostream &str) override;

    uint32_t firstScissor;
    std::vector<VkRect2D> scissors;
};

class sync_cmd_bind_descriptor_sets : public sync_cmd_base
{
public:
    sync_cmd_bind_descriptor_sets(
        VkPipelineBindPoint pipelineBindPoint,
        VkPipelineLayout layout,
        uint32_t firstSet,
        uint32_t descriptorSetCount,
        const VkDescriptorSet *pDescriptorSets,
        uint32_t dynamicOffsetCount,
        const uint32_t *pDynamicOffsets) :
        pipelineBindPoint(pipelineBindPoint),
        layout(layout),
        firstSet(firstSet),
        descriptorSets(pDescriptorSets, pDescriptorSets + descriptorSetCount),
        dynamicOffsets(pDynamicOffsets, pDynamicOffsets + dynamicOffsetCount)
    {
    }

    virtual void to_string(std::ostream &str) override;

    VkPipelineBindPoint pipelineBindPoint;
    VkPipelineLayout layout;
    uint32_t firstSet;
    std::vector<VkDescriptorSet> descriptorSets;
    std::vector<uint32_t> dynamicOffsets;
};

class sync_cmd_bind_vertex_buffers : public sync_cmd_base
{
public:
    sync_cmd_bind_vertex_buffers(
        uint32_t firstBinding,
        uint32_t bindingCount,
        const VkBuffer *pBuffers,
        const VkDeviceSize *pOffsets) :
        firstBinding(firstBinding),
        buffers(pBuffers, pBuffers + bindingCount),
        offsets(pOffsets, pOffsets + bindingCount)
    {
    }

    virtual void to_string(std::ostream &str) override;

    uint32_t firstBinding;
    std::vector<VkBuffer> buffers;
    std::vector<VkDeviceSize> offsets;
};

class sync_cmd_draw : public sync_cmd_base
{
public:
    sync_cmd_draw(
        uint32_t vertexCount,
        uint32_t instanceCount,
        uint32_t firstVertex,
        uint32_t firstInstance) :
        vertexCount(vertexCount),
        instanceCount(instanceCount),
        firstVertex(firstVertex),
        firstInstance(firstInstance)
    {
    }

    virtual bool is_draw() const override { return true; }

    virtual void to_string(std::ostream &str) override;

    uint32_t vertexCount;
    uint32_t instanceCount;
    uint32_t firstVertex;
    uint32_t firstInstance;
};

class sync_cmd_draw_indexed : public sync_cmd_base
{
public:
    sync_cmd_draw_indexed(
        uint32_t indexCount,
        uint32_t instanceCount,
        uint32_t firstIndex,
        int32_t vertexOffset,
        uint32_t firstInstance) :
        indexCount(indexCount),
        instanceCount(instanceCount),
        firstIndex(firstIndex),
        vertexOffset(vertexOffset),
        firstInstance(firstInstance)
    {
    }

    virtual bool is_draw() const override { return true; }

    virtual void to_string(std::ostream &str) override;

    uint32_t indexCount;
    uint32_t instanceCount;
    uint32_t firstIndex;
    int32_t vertexOffset;
    uint32_t firstInstance;
};

class sync_cmd_copy_image : public sync_cmd_base
{
public:
    sync_cmd_copy_image(
        VkImage srcImage,
        VkImageLayout srcImageLayout,
        VkImage dstImage,
        VkImageLayout dstImageLayout,
        uint32_t regionCount,
        const VkImageCopy *pRegions) :
        srcImage(srcImage),
        srcImageLayout(srcImageLayout),
        dstImage(dstImage),
        dstImageLayout(dstImageLayout),
        regions(pRegions, pRegions + regionCount)
    {
    }

    virtual void to_string(std::ostream &str) override;

    VkImage srcImage;
    VkImageLayout srcImageLayout;
    VkImage dstImage;
    VkImageLayout dstImageLayout;
    std::vector<VkImageCopy> regions;
};

class sync_cmd_pipeline_barrier : public sync_cmd_base
{
public:
    sync_cmd_pipeline_barrier(
        VkPipelineStageFlags srcStageMask,
        VkPipelineStageFlags dstStageMask,
        VkDependencyFlags dependencyFlags,
        uint32_t memoryBarrierCount,
        const VkMemoryBarrier *pMemoryBarriers,
        uint32_t bufferMemoryBarrierCount,
        const VkBufferMemoryBarrier *pBufferMemoryBarriers,
        uint32_t imageMemoryBarrierCount,
        const VkImageMemoryBarrier *pImageMemoryBarriers) :
        srcStageMask(srcStageMask),
        dstStageMask(dstStageMask),
        dependencyFlags(dependencyFlags),
        memoryBarriers(pMemoryBarriers, pMemoryBarriers + memoryBarrierCount),
        bufferMemoryBarriers(pBufferMemoryBarriers, pBufferMemoryBarriers + bufferMemoryBarrierCount),
        imageMemoryBarriers(pImageMemoryBarriers, pImageMemoryBarriers + imageMemoryBarrierCount)
    {
    }

    virtual void to_string(std::ostream &str) override;

    VkPipelineStageFlags srcStageMask;
    VkPipelineStageFlags dstStageMask;
    VkDependencyFlags dependencyFlags;
    std::vector<VkMemoryBarrier> memoryBarriers;
    std::vector<VkBufferMemoryBarrier> bufferMemoryBarriers;
    std::vector<VkImageMemoryBarrier> imageMemoryBarriers;
};

class sync_cmd_begin_render_pass : public sync_cmd_base
{
public:
    sync_cmd_begin_render_pass(
        const VkRenderPassBeginInfo *pRenderPassBegin,
        VkSubpassContents contents) :
        renderPass(pRenderPassBegin->renderPass),
        framebuffer(pRenderPassBegin->framebuffer),
        renderArea(pRenderPassBegin->renderArea),
        clearValues(pRenderPassBegin->pClearValues, pRenderPassBegin->pClearValues + pRenderPassBegin->clearValueCount),
        contents(contents)
    {
    }

    virtual void to_string(std::ostream &str) override;

    VkRenderPass renderPass;
    VkFramebuffer framebuffer;
    VkRect2D renderArea;
    std::vector<VkClearValue> clearValues;
    VkSubpassContents contents;
};

class sync_cmd_next_subpass : public sync_cmd_base
{
public:
    sync_cmd_next_subpass(VkSubpassContents contents) :
        contents(contents)
    {
    }

    virtual void to_string(std::ostream &str) override;

    VkSubpassContents contents;
};

class sync_cmd_end_render_pass : public sync_cmd_base
{
public:
    sync_cmd_end_render_pass()
    {
    }

    virtual void to_string(std::ostream &str) override;
};


/**
 * Internal state for a VkCommandBuffer.
 */
struct sync_command_buffer
{
    sync_command_buffer() { }

    void reset();

    VkCommandBuffer command_buffer;

    // Pool that this buffer belongs to
    VkCommandPool command_pool;

    // vkAllocateCommandBuffers state
    VkCommandBufferLevel level;

    sync_command_buffer_state state;

    // vkBeginCommandBuffer state
    VkCommandBufferUsageFlags flags;

    // vkBeginCommandBuffer pInheritanceInfo state
    VkRenderPass renderPass;
    uint32_t subpass;
    VkFramebuffer framebuffer;
    VkBool32 occlusionQueryEnable;
    VkQueryControlFlags queryFlags;
    VkQueryPipelineStatisticFlags pipelineStatistics;

    std::vector<std::unique_ptr<sync_cmd_base>> commands;
};

/**
 * Internal state for a VkCommandPool.
 */
struct sync_command_pool
{
    VkCommandPool command_pool;

    // All currently-existing VkCommandBuffers associated belonging to this pool
    std::set<VkCommandBuffer> command_buffers;
};

/**
 * Internal state for a VkRenderPass.
 */
struct sync_render_pass
{
    VkRenderPass render_pass;

    struct subpass_description
    {
        VkSubpassDescriptionFlags flags;
        VkPipelineBindPoint pipelineBindPoint;
        std::vector<VkAttachmentReference> inputAttachments;
        std::vector<VkAttachmentReference> colorAttachments;
        std::vector<VkAttachmentReference> resolveAttachments;
        std::vector<VkAttachmentReference> depthStencilAttachment;
        std::vector<uint32_t> preserveAttachments;
    };

    VkRenderPassCreateFlags flags;
    std::vector<VkAttachmentDescription> attachments;
    std::vector<subpass_description> subpasses;
    std::vector<VkSubpassDependency> dependencies;
};

/**
 * Internal state for a VkDevice.
 */
struct sync_device
{
    sync_device() { }

    // All currently-existing VkCommandPools
    std::map<VkCommandPool, sync_command_pool> command_pools;

    // All currently-existing VkCommandBuffers.
    // This must remain in sync with command_pools[].command_buffers
    // (every command buffer belongs to a single pool)
    std::map<VkCommandBuffer, sync_command_buffer> command_buffers;

    std::map<VkRenderPass, sync_render_pass> render_passes;
};

#endif // INCLUDED_VULKAN_SYNC_H
