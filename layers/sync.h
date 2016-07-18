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
#include "vk_layer_logging.h"

#include <map>
#include <set>
#include <vector>
#include <memory>

class sync_cmd_pipeline_barrier;
class sync_cmd_begin_render_pass;
class sync_cmd_next_subpass;
class sync_cmd_end_render_pass;
class sync_cmd_bind_pipeline;
class sync_cmd_bind_descriptor_sets;

class SyncValidator;

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
    sync_cmd_base();

    virtual ~sync_cmd_base() { }

    virtual bool is_draw() const { return false; }

    virtual const sync_cmd_pipeline_barrier *as_pipeline_barrier() const { return nullptr; }

    virtual const sync_cmd_begin_render_pass *as_begin_render_pass() const { return nullptr; }
    virtual const sync_cmd_next_subpass *as_next_subpass() const { return nullptr; }
    virtual const sync_cmd_end_render_pass *as_end_render_pass() const { return nullptr; }

    virtual const sync_cmd_bind_pipeline *as_bind_pipeline() const { return nullptr; }
    virtual const sync_cmd_bind_descriptor_sets *as_bind_descriptor_sets() const { return nullptr; }

    virtual void to_string(std::ostream &str) = 0;

    std::vector<std::string> get_backtrace();

    static const int BACKTRACE_SIZE = 8;
    void *mBackTrace[BACKTRACE_SIZE];
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
    virtual const sync_cmd_bind_pipeline *as_bind_pipeline() const override { return this; }

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
    virtual const sync_cmd_bind_descriptor_sets *as_bind_descriptor_sets() const override { return this; }

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
    virtual const sync_cmd_pipeline_barrier *as_pipeline_barrier() const { return this; }

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
    virtual const sync_cmd_begin_render_pass *as_begin_render_pass() const override { return this; }


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
    virtual const sync_cmd_next_subpass *as_next_subpass() const override { return nullptr; }

    VkSubpassContents contents;
};

class sync_cmd_end_render_pass : public sync_cmd_base
{
public:
    sync_cmd_end_render_pass()
    {
    }

    virtual void to_string(std::ostream &str) override;
    virtual const sync_cmd_end_render_pass *as_end_render_pass() const override { return nullptr; }
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

    // All currently-existing VkCommandBuffers belonging to this pool
    std::set<VkCommandBuffer> command_buffers;
};

/**
 * Internal state for a VkDescriptorSet.
 */
struct sync_descriptor_set
{
    sync_descriptor_set() { }

    VkDescriptorSet descriptor_set;

    struct descriptor
    {
        bool valid;
        VkDescriptorImageInfo imageInfo;
        VkDescriptorBufferInfo bufferInfo;
        VkBufferView bufferView;
    };

    struct descriptor_array
    {
        VkDescriptorType descriptorType;
        std::vector<descriptor> descriptors;
    };

    // Pool that this descriptor set belongs to
    VkDescriptorPool descriptor_pool;

    VkDescriptorSetLayout setLayout;

    std::map<uint32_t, descriptor_array> bindings;

    void to_string(std::ostream &str);
};

/**
 * Internal state for a VkDescriptorPool.
 */
struct sync_descriptor_pool
{
    VkDescriptorPool descriptor_pool;

    // All currently-existing VkDescriptorSets belonging to this pool
    std::set<VkDescriptorSet> descriptor_sets;
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
 * Internal state for a VkDescriptorSetLayout.
 */
struct sync_descriptor_set_layout
{
    VkDescriptorSetLayout descriptor_set_layout;

    struct descriptor_set_layout_binding
    {
        uint32_t binding;
        VkDescriptorType descriptorType;
        uint32_t descriptorCount;
        VkShaderStageFlags stageFlags;
        std::vector<VkSampler> immutableSamplers;
    };

    VkDescriptorSetLayoutCreateFlags flags;
    std::vector<descriptor_set_layout_binding> bindings;

    void to_string(std::ostream &str);
};

/**
 * Internal state for a VkPipelineLayout.
 */
struct sync_pipeline_layout
{
    VkPipelineLayout pipeline_layout;

    VkPipelineLayoutCreateFlags flags;
    std::vector<VkDescriptorSetLayout> setLayouts;
    std::vector<VkPushConstantRange> pushConstantRanges;

    void to_string(std::ostream &str);
};

/**
 * Internal state for a VkDeviceMemory.
 */
struct sync_device_memory
{
    VkDeviceMemory deviceMemory;

    // A globally-unique ID, used for tracking accesses to the same memory
    // object during the device's lifetime
    uint64_t uid;

    // VkMemoryAllocateInfo
    VkDeviceSize allocationSize;
    uint32_t memoryTypeIndex;

    // vkMapMemory
    bool isMapped;
    VkDeviceSize mapOffset;
    VkDeviceSize mapSize;
    VkMemoryMapFlags mapFlags;
    void *pMapData;

    void to_string(std::ostream &str);
};

/**
 * Internal state for a VkBuffer.
 */
struct sync_buffer
{
    VkBuffer buffer;

    VkBufferCreateFlags flags;
    VkDeviceSize size;
    VkBufferUsageFlags usage;
    VkSharingMode sharingMode;
    std::vector<uint32_t> queueFamilyIndices;

    // Got from vkGetBufferMemoryRequirements
    VkMemoryRequirements memoryRequirements;

    // Set by vkBindBufferMemory
    VkDeviceMemory memory;
    VkDeviceSize memoryOffset;

    void to_string(std::ostream &str);
};

/**
 * Internal state for a VkBufferView.
 */
struct sync_buffer_view
{
    VkBufferView buffer_view;

    VkBufferViewCreateFlags flags;
    VkBuffer buffer;
    VkFormat format;
    VkDeviceSize offset;
    VkDeviceSize range;

    void to_string(std::ostream &str);
};

/**
 * Internal state for a VkImage.
 */
struct sync_image
{
    VkImage image;

    bool isSwapchain;

    VkImageCreateFlags flags;
    VkImageType imageType;
    VkFormat format;
    VkExtent3D extent;
    uint32_t mipLevels;
    uint32_t arrayLayers;
    VkSampleCountFlagBits samples;
    VkImageTiling tiling;
    VkImageUsageFlags usage;
    VkSharingMode sharingMode;
    std::vector<uint32_t> queueFamilyIndices;
    VkImageLayout initialLayout;

    // Got from vkGetImageMemoryRequirements
    VkMemoryRequirements memoryRequirements;

    // Subresource layouts indexed by (mipLevel, arrayLayer, aspectMask)
    // (Only present if tiling=LINEAR)
    std::vector<VkSubresourceLayout> subresourceLayouts;

    // Set by vkBindImageMemory
    VkDeviceMemory memory;
    VkDeviceSize memoryOffset;

    void to_string(std::ostream &str);
};

/**
 * Internal state for a VkImageView.
 */
struct sync_image_view
{
    VkImageView image_view;

    VkImageViewCreateFlags flags;
    VkImage image;
    VkImageViewType viewType;
    VkFormat format;
    VkComponentMapping components;
    VkImageSubresourceRange subresourceRange;

    void to_string(std::ostream &str);
};

/**
 * Internal state for a graphics VkPipeline.
 */
struct sync_graphics_pipeline
{
    VkPipeline pipeline;

    struct shader_stage
    {
        VkPipelineShaderStageCreateFlags flags;
        VkShaderStageFlagBits stage;
        VkShaderModule module;
        std::string name;

        // TODO:
//         const VkSpecializationInfo*         pSpecializationInfo;
    };

    struct vertex_input_state
    {
        VkPipelineVertexInputStateCreateFlags flags;
        std::vector<VkVertexInputBindingDescription> vertexBindingDescriptions;
        std::vector<VkVertexInputAttributeDescription> vertexAttributeDescriptions;
    };

    struct input_assembly_state
    {
        VkPipelineInputAssemblyStateCreateFlags flags;
        VkPrimitiveTopology topology;
        VkBool32 primitiveRestartEnable;
    };

    VkPipelineCreateFlags flags;
    std::vector<shader_stage> stages;
    vertex_input_state vertexInputState;
    input_assembly_state inputAssemblyState;

    // TODO:
//     const VkPipelineTessellationStateCreateInfo*     pTessellationState;
//     const VkPipelineViewportStateCreateInfo*         pViewportState;
//     const VkPipelineRasterizationStateCreateInfo*    pRasterizationState;
//     const VkPipelineMultisampleStateCreateInfo*      pMultisampleState;
//     const VkPipelineDepthStencilStateCreateInfo*     pDepthStencilState;
//     const VkPipelineColorBlendStateCreateInfo*       pColorBlendState;
//     const VkPipelineDynamicStateCreateInfo*          pDynamicState;

    VkPipelineLayout layout;
    VkRenderPass renderPass;
    uint32_t subpass;

    void to_string(std::ostream &str);
};

/**
 * Internal state for a VkSwapchainKHR.
 */
struct sync_swapchain
{
    VkSwapchainKHR swapchain;

    std::vector<VkImage> images;

//     void to_string(std::ostream &str);
};

/**
 * Internal state for a VkDevice.
 */
struct sync_device
{
    sync_device() : nextMemoryUid(0) { }

    // All currently-existing VkCommandPools
    std::map<VkCommandPool, sync_command_pool> command_pools;

    // All currently-existing VkCommandBuffers.
    // This must remain in sync with command_pools[].command_buffers
    // (every command buffer belongs to a single pool)
    std::map<VkCommandBuffer, sync_command_buffer> command_buffers;

    // All currently-existing VkDescriptorPools
    std::map<VkDescriptorPool, sync_descriptor_pool> descriptor_pools;

    // All currently-existing VkDescriptorSets.
    // This must remain in sync with descriptor_pools[].descriptor_sets
    // (every descriptor set belongs to a single pool)
    std::map<VkDescriptorSet, sync_descriptor_set> descriptor_sets;

    std::map<VkRenderPass, sync_render_pass> render_passes;

    std::map<VkDescriptorSetLayout, sync_descriptor_set_layout> descriptor_set_layouts;

    std::map<VkPipelineLayout, sync_pipeline_layout> pipeline_layouts;

    std::map<VkDeviceMemory, sync_device_memory> device_memories;
    std::map<VkBuffer, sync_buffer> buffers;
    std::map<VkBufferView, sync_buffer_view> buffer_views;
    std::map<VkImage, sync_image> images;
    std::map<VkImageView, sync_image_view> image_views;

    std::map<VkPipeline, sync_graphics_pipeline> graphics_pipelines;

    std::map<VkSwapchainKHR, sync_swapchain> swapchains;

    uint64_t nextMemoryUid;

    std::unique_ptr<SyncValidator> mSyncValidator;
};



struct CommandId
{
    static const uint64_t SUBPASS_NONE = ~(uint64_t)0;

    uint64_t queueId;
    uint64_t subpassId;
    uint64_t sequenceId;

    CommandId();
    bool operator<(const CommandId &c) const;
};

struct MemRegion
{
    enum EType
    {
        INVALID,
        GLOBAL,
        BUFFER,
        IMAGE,
        SWAPCHAIN_IMAGE,
    };

    EType type;

    // If BUFFER:
    VkBuffer buffer;
    VkDeviceSize bufferOffset;
    VkDeviceSize bufferRange;

    // If IMAGE or SWAPCHAIN_IMAGE:
    VkImage image;
    VkImageSubresourceRange imageSubresourceRange;

    // If BUFFER or IMAGE:
    VkDeviceMemory deviceMemory;
    VkDeviceSize deviceMemoryOffset;

    MemRegion();
    bool operator<(const MemRegion &m) const;
    void to_string(std::ostream &str) const;
};

struct SyncNode
{
    enum ENodeType
    {
        INVALID,
        ACTION_CMD_STAGE,
        SYNC_CMD_SRC_STAGE,
        SYNC_CMD_DST_STAGE,
        SYNC_CMD_SRC,
        SYNC_CMD_DST,
        SYNC_CMD_POST_TRANS,
        SYNC_CMD_PRE_TRANS,
        TRANSITION, // XXX - not needed?
        MEM_READ,
        MEM_WRITE,
        MEM_FLUSH,
        MEM_INVALIDATE,
    };

    ENodeType type;

    CommandId commandId;
    VkPipelineStageFlags stages;
    VkAccessFlags accesses;
    MemRegion memory;

    SyncNode();
    bool operator<(const SyncNode &n) const;
    void to_string(std::ostream &str) const;
};

struct SyncEdge
{
    // Node IDs
    uint64_t a;
    uint64_t b;

    SyncEdge(uint64_t a, uint64_t b) : a(a), b(b) { }

    bool operator<(const SyncEdge &e) const;
};

struct SyncEdgeSet
{
    // Node ID of sync command node
    uint64_t sync;

    // Exclusive upper/lower bound of set of commands
    CommandId commandBound;
    VkPipelineStageFlagBits stage;

    SyncEdgeSet(uint64_t sync, CommandId commandBound, VkPipelineStageFlagBits stage) :
        sync(sync), commandBound(commandBound), stage(stage)
    {
    }

    bool operator<(const SyncEdgeSet &e) const;
};

class SyncValidator
{
public:
    SyncValidator(sync_device &syncDevice, debug_report_data *reportData);

    bool submitCmdBuffer(VkQueue queue, const sync_command_buffer &buf);

private:
    sync_device &mSyncDevice;
    debug_report_data *mReportData;

    CommandId mNextCommandId;
    uint64_t mNextSubpassId;
    uint64_t mNextNodeId;

    std::map<SyncNode, uint64_t> mNodeIds;
    std::map<uint64_t, SyncNode> mNodesById;
    std::set<SyncEdge> mEdges;
    std::set<SyncEdgeSet> mPrecedingEdges;
    std::set<SyncEdgeSet> mFollowingEdges;

    uint64_t addNode(const SyncNode &node);

    bool findPath(uint64_t srcNodeId, uint64_t dstNodeId);
};

#endif // INCLUDED_VULKAN_SYNC_H
