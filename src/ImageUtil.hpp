#ifndef IMAGEUTIL_H_
#define IMAGEUTIL_H_

#include "Device.hpp"

VkImageView createImageView(Device& device, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels);

void transitionImageLayout(Device& device, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels);

bool hasStencilComponent(VkFormat format);

void createImage(Device& device, uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& memory);

#endif // IMAGEUTIL_H_
