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

#include "vk_loader_platform.h"
#include "vk_layer.h"

#include <map>
#include <set>

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

    virtual void to_string(std::ostream &str)
    {
        str << "vkCmdDraw {";
        str << " vertexCount=" << vertexCount;
        str << " instanceCount=" << instanceCount;
        str << " firstVertex=" << firstVertex;
        str << " firstInstance=" << firstInstance;
        str << " }";
    }

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

    virtual void to_string(std::ostream &str)
    {
        str << "vkCmdDrawIndexed {";
        str << " indexCount=" << indexCount;
        str << " instanceCount=" << instanceCount;
        str << " firstIndex=" << firstIndex;
        str << " vertexOffset=" << vertexOffset;
        str << " firstInstance=" << firstInstance;
        str << " }";
    }

    uint32_t indexCount;
    uint32_t instanceCount;
    uint32_t firstIndex;
    int32_t vertexOffset;
    uint32_t firstInstance;
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

    virtual void to_string(std::ostream &str)
    {
        str << "vkCmdPipelineBarrier {";
        str << " srcStageMask=0x" << std::hex << srcStageMask << std::dec;
        str << " dstStageMask=0x" << std::hex << dstStageMask << std::dec;
        str << " dependencyFlags=0x" << std::hex << dependencyFlags << std::dec;

        str << " memoryBarriers=[";
        for (auto &b : memoryBarriers)
        {
            str << " {";
            str << " srcAccessMask=0x" << std::hex << b.srcAccessMask << std::dec;
            str << " dstAccessMask=0x" << std::hex << b.dstAccessMask << std::dec;
            str << " }";
        }
        str << " ]";

        str << " bufferMemoryBarriers=[";
        for (auto &b : bufferMemoryBarriers)
        {
            str << " {";
            str << " srcAccessMask=0x" << std::hex << b.srcAccessMask << std::dec;
            str << " dstAccessMask=0x" << std::hex << b.dstAccessMask << std::dec;
            str << " srcQueueFamilyIndex=" << b.srcQueueFamilyIndex;
            str << " dstQueueFamilyIndex=" << b.dstQueueFamilyIndex;
            str << " buffer=" << (void *)b.buffer;
            str << " offset=" << b.offset;
            str << " size=" << b.size;
            str << " }";
        }
        str << " ]";

        str << " imageMemoryBarriers=[";
        for (auto &b : imageMemoryBarriers)
        {
            str << " {";
            str << " srcAccessMask=0x" << std::hex << b.srcAccessMask << std::dec;
            str << " dstAccessMask=0x" << std::hex << b.dstAccessMask << std::dec;
            str << " oldLayout=" << b.oldLayout;
            str << " newLayout=" << b.newLayout;
            str << " srcQueueFamilyIndex=" << b.srcQueueFamilyIndex;
            str << " dstQueueFamilyIndex=" << b.dstQueueFamilyIndex;
            str << " image=" << (void *)b.image;
            str << " subresourceRange={";
            str << " aspectMask=0x" << std::hex << b.subresourceRange.aspectMask << std::dec;
            str << " baseMipLevel=" << b.subresourceRange.baseMipLevel;
            str << " levelCount=" << b.subresourceRange.levelCount;
            str << " baseArrayLayer=" << b.subresourceRange.baseArrayLayer;
            str << " layerCount=" << b.subresourceRange.layerCount;
            str << " }";
            str << " }";
        }
        str << " ]";

        str << " }";
    }

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

    virtual void to_string(std::ostream &str)
    {
        str << "vkCmdBeginRenderPass {";
        str << " renderPass=" << (void *)renderPass;
        str << " framebuffer=" << (void *)framebuffer;
        str << " renderArea={";
        str << " offset=(" << renderArea.offset.x << ", " << renderArea.offset.y << ")";
        str << " extent=(" << renderArea.extent.width << ", " << renderArea.extent.height << ")";
        str << " }";

        str << " clearValues=[";
        for (auto &v : clearValues)
        {
            // Correct interpretation depends on attachment type, which we
            // don't know here, so just print all possibilities
            str << " {";
            str << " (" << v.color.float32[0] << ", " << v.color.float32[1] << ", " << v.color.float32[2] << ", " << v.color.float32[3] << ")";
            str << " |";
            str << " (" << v.color.int32[0] << ", " << v.color.int32[1] << ", " << v.color.int32[2] << ", " << v.color.int32[3] << ")";
            str << " |";
            str << " (" << v.color.uint32[0] << ", " << v.color.uint32[1] << ", " << v.color.uint32[2] << ", " << v.color.uint32[3] << ")";
            str << " |";
            str << " (" << v.depthStencil.depth << ", " << v.depthStencil.stencil << ")";
            str << " }";
        }
        str << " ]";

        str << " contents=" << contents;
        str << " }";
    }

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

    virtual void to_string(std::ostream &str)
    {
        str << "vkCmdNextSubpass {";
        str << " contents=" << contents;
        str << " }";
    }

    VkSubpassContents contents;
};

class sync_cmd_end_render_pass : public sync_cmd_base
{
public:
    sync_cmd_end_render_pass()
    {
    }

    virtual void to_string(std::ostream &str)
    {
        str << "vkCmdEndRenderPass {";
        str << " }";
    }
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
};
