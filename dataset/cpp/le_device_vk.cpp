#include "le_backend_vk.h"
#include "le_backend_types_internal.h"
#include "le_log.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <cstring> // for memcpy
#include <cassert>
#include <bitset>

static constexpr auto LOGGER_LABEL = "le_backend";

#ifdef _WIN32
#	define __PRETTY_FUNCTION__ __FUNCSIG__
#endif //

struct le_device_o {

	VkDevice         vkDevice         = nullptr;
	VkPhysicalDevice vkPhysicalDevice = nullptr;

	struct Properties {
		VkPhysicalDeviceProperties2                     device_properties;
		VkPhysicalDeviceRayTracingPipelinePropertiesKHR raytracing_properties;
		VkPhysicalDeviceVulkan11Properties              vk_11_physical_device_properties;
		VkPhysicalDeviceVulkan12Properties              vk_12_physical_device_properties;
		VkPhysicalDeviceVulkan13Properties              vk_13_physical_device_properties;
		//
		VkPhysicalDeviceMemoryProperties2 memory_properties;
		//
		std::vector<VkQueueFamilyProperties2> queue_family_properties;

	} properties;

	std::vector<VkQueue>      queues;              // queues as created with device
	std::vector<VkQueueFlags> queues_flags;        // per-queue capability flags
	std::vector<uint32_t>     queues_family_index; // per-queue family index

	std::set<std::string> requestedDeviceExtensions;

	VkFormat defaultDepthStencilFormat;
	VkFormat defaultColorAttachmentFormat;
	VkFormat defaultSampledImageFormat;

	uint32_t referenceCount = 0;
};

// ----------------------------------------------------------------------

static std::string le_queue_flags_to_string( le::QueueFlags const& flags ) {
	std::string result;

	uint64_t f = flags;
	for ( int i = 0; ( f > 0 ); i++ ) {
		if ( f & 1 ) {
			result.append(
			    result.empty()
			        ? std::string( le::to_str( le::QueueFlagBits( 1 << i ) ) )
			        : " | " + std::string( le::to_str( le::QueueFlagBits( 1 << i ) ) ) );
		}
		f >>= 1;
	}
	return result;
}

// ----------------------------------------------------------------------

uint32_t findClosestMatchingQueueIndex( const std::vector<VkQueueFlags>& haystack, const VkQueueFlags& needle ) {

	// Find out the queue family index for a queue best matching the given flags.
	// We use this to find out the index of the Graphics Queue for example.

	for ( uint32_t i = 0; i != haystack.size(); i++ ) {
		if ( haystack[ i ] == needle ) {
			// First perfect match
			return i;
		}
	}

	for ( uint32_t i = 0; i != haystack.size(); i++ ) {
		if ( haystack[ i ] & needle ) {
			// First multi-function queue match
			return i;
		}
	}

	// ---------| invariant: no queue found

	if ( needle & VK_QUEUE_GRAPHICS_BIT ) {
		static auto logger = LeLog( LOGGER_LABEL );
		logger.error( "Could not find queue family index matching: '%d'", needle );
	}

	return ~( uint32_t( 0 ) );
}

// ----------------------------------------------------------------------
/// \brief Find best match for a vector or queues defined by queueFamiliyProperties flags
/// \note  For each entry in the result vector the tuple values represent:
///        0.. best matching queue family
///        1.. index within queue family
///        2.. index of queue from props vector (used to keep queue indices
//             consistent between requested queues and queues you will render to)

struct QueueQueryResult {
	uint32_t queue_family_index;     // Queue Family Index
	uint32_t queue_index;            // Offset within queue family.
	                                 // If we're picking the 1st queue of a family, the offset will be 0,
	                                 // If we're picking the 3rd queue of a family, the offset will be 2.
	VkQueueFlags queue_family_flags; // Full flags that this queue supports
};

std::vector<QueueQueryResult> findBestMatchForRequestedQueues( const std::vector<VkQueueFamilyProperties2>& props, const std::vector<VkQueueFlags>& reqProps ) {
	std::vector<QueueQueryResult> result;

	static auto logger = LeLog( LOGGER_LABEL );

	std::vector<uint32_t> usedQueues( props.size(), 0 ); // number of used queues per queue family

	size_t reqIdx = 0; // original index for requested queue
	for ( const auto& flags : reqProps ) {

		bool     foundMatch            = false;
		uint32_t foundFamily           = 0;
		uint32_t lowest_num_extra_bits = uint32_t( ~0 );

		// Best match is a queue that has the fewest extra bits set after it covers all the bits
		// that are mandatory.
		// We should be able to find this in one pass per requested queue.

		for ( uint32_t familyIndex = 0; familyIndex != props.size(); familyIndex++ ) {
			VkQueueFlags available_flags = props[ familyIndex ].queueFamilyProperties.queueFlags;
			VkQueueFlags requested_flags = flags;

			if ( ( available_flags & requested_flags ) == requested_flags ) {
				// requested_flags are contained in available flags
				VkQueueFlags leftover_flags = ( available_flags & ( ~requested_flags ) ); // flags which only appear in available_flags
				size_t       num_extra_bits = std::bitset<sizeof( VkQueueFlags ) * 8>( leftover_flags ).count();
				if ( num_extra_bits < lowest_num_extra_bits ) {
					// Check if queue would be available
					if ( usedQueues[ familyIndex ] < props[ familyIndex ].queueFamilyProperties.queueCount ) {
						if ( foundMatch ) {
							// we must release the previously used queue so that it becomes available again
							usedQueues[ foundFamily ]--;
						}
						usedQueues[ familyIndex ]++; // mark this queue as used
						foundFamily           = familyIndex;
						lowest_num_extra_bits = num_extra_bits;
						foundMatch            = true;
					}
				}
			}
		}

		if ( foundMatch ) {

			logger.info( "Found queue { %s } matching requirement: { %s }.",
			             le_queue_flags_to_string( le::QueueFlagBits( props[ foundFamily ].queueFamilyProperties.queueFlags ) ).c_str(),
			             le_queue_flags_to_string( le::QueueFlagBits( flags ) ).c_str() );

			assert( usedQueues[ foundFamily ] > 0 && "must have at least one used queue at index" );

			result.push_back( {
			    .queue_family_index = foundFamily,                                           // queue family index
			    .queue_index        = usedQueues[ foundFamily ] - 1,                         // index within queue family
			    .queue_family_flags = props[ foundFamily ].queueFamilyProperties.queueFlags, // queue capability flags for this family
			} );

		} else {
			logger.error( "No queue available matching requirement: { %d }", flags );
		}

		++reqIdx;
	}

	return result;
}

// ----------------------------------------------------------------------

static le_device_o* device_create( le_backend_vk_instance_o* backend_instance, const char** extension_names, uint32_t extension_names_count ) {

	static auto logger = LeLog( LOGGER_LABEL );

	le_device_o* self = new le_device_o{};

	{
		self->properties.device_properties = {
		    .sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
		    .pNext      = &self->properties.vk_11_physical_device_properties,
		    .properties = {},
		};

		self->properties.vk_11_physical_device_properties = {
		    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES,
		    .pNext = &self->properties.vk_12_physical_device_properties,
		};

		self->properties.vk_12_physical_device_properties = {
		    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES,
		    .pNext = &self->properties.vk_13_physical_device_properties,
		};

		self->properties.vk_13_physical_device_properties = {
		    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES,
		    .pNext = &self->properties.raytracing_properties,
		};

		self->properties.raytracing_properties = {
		    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
		    .pNext = nullptr,
		};

		self->properties.memory_properties = {
		    .sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
		    .pNext            = nullptr, // optional
		    .memoryProperties = {},
		};
	}

	using namespace le_backend_vk;

	VkInstance instance = vk_instance_i.get_vk_instance( backend_instance );

	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices( instance, &deviceCount, nullptr );
	std::vector<VkPhysicalDevice> deviceList( deviceCount );
	vkEnumeratePhysicalDevices( instance, &deviceCount, deviceList.data() );

	if ( deviceCount == 0 ) {
		logger.error( "No physical Vulkan device found. Quitting." );
		exit( 1 );
	}
	// ---------| invariant: there is at least one physical device

	{
		// Find the first device which is a dedicated GPU, if none of these can be found,
		// fall back to the first physical device.

		self->vkPhysicalDevice = deviceList.front(); // select the first device as a fallback

		for ( auto d = deviceList.begin(); d != deviceList.end(); d++ ) {
			VkPhysicalDeviceProperties device_properties{};
			vkGetPhysicalDeviceProperties( *d, &device_properties );
			if ( device_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ) {
				self->vkPhysicalDevice = *d;
				break;
			}
		}

		// Fetch extended device properties for the currently selected physical device
		vkGetPhysicalDeviceProperties2( self->vkPhysicalDevice, &self->properties.device_properties );

		logger.info( "Selected GPU: %s", self->properties.device_properties.properties.deviceName );
	}

	// Let's find out the devices' memory properties
	vkGetPhysicalDeviceMemoryProperties2( self->vkPhysicalDevice, &self->properties.memory_properties );

	{
		uint32_t numQueueFamilyProperties = 0;
		vkGetPhysicalDeviceQueueFamilyProperties2( self->vkPhysicalDevice, &numQueueFamilyProperties, nullptr );
		self->properties.queue_family_properties.resize(
		    numQueueFamilyProperties,
		    {
		        .sType                 = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2,
		        .pNext                 = nullptr, // optional
		        .queueFamilyProperties = {},
		    } );
		vkGetPhysicalDeviceQueueFamilyProperties2( self->vkPhysicalDevice, &numQueueFamilyProperties, self->properties.queue_family_properties.data() );
	}

	std::vector<VkQueueFlags> requested_queues;
	{
		// fetch user-requested queues
		uint32_t num_requested_queues = 0;
		le_backend_vk::settings_i.get_requested_queue_capabilities( nullptr, &num_requested_queues );
		requested_queues.resize( num_requested_queues );
		le_backend_vk::settings_i.get_requested_queue_capabilities( requested_queues.data(), &num_requested_queues );

		if ( requested_queues.empty() ) {
			VkQueueFlags default_queue_flags = { VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT };
			le::Log( LOGGER_LABEL ).info( "No queues explicitly requested, requesting default: { %s }", le_queue_flags_to_string( le::QueueFlagBits( default_queue_flags ) ).c_str() );
			requested_queues.push_back( default_queue_flags );
		}
	}
	std::vector<QueueQueryResult> available_queues =
	    findBestMatchForRequestedQueues( self->properties.queue_family_properties, requested_queues );

	// Create queues based on available_queues
	std::vector<VkDeviceQueueCreateInfo> device_queue_creation_infos;
	// Consolidate queues by queue family type - this will also sort by queue family type.
	std::map<uint32_t, uint32_t> queueCountPerFamily; // queueFamily -> count

	for ( const auto& q : available_queues ) {
		// Attempt to insert family to map

		auto insertResult = queueCountPerFamily.insert( { q.queue_family_index, 1 } );
		if ( false == insertResult.second ) {
			// Increment count if family entry already existed in map.
			insertResult.first->second++;
		}
	}

	device_queue_creation_infos.reserve( available_queues.size() );

	// We must store this in a map so that the pointer stays
	// alive until we call the api.
	std::map<uint32_t, std::vector<float>> prioritiesPerFamily;

	for ( auto& q : queueCountPerFamily ) {
		VkDeviceQueueCreateInfo queueCreateInfo;
		const auto&             queueFamily = q.first;
		const auto&             queueCount  = q.second;
		prioritiesPerFamily[ queueFamily ].resize( queueCount, 1.f ); // all queues have the same priority, 1.

		queueCreateInfo = {
		    .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		    .pNext            = nullptr,
		    .flags            = 0,
		    .queueFamilyIndex = queueFamily,
		    .queueCount       = queueCount,
		    .pQueuePriorities = prioritiesPerFamily[ queueFamily ].data(),
		};

		device_queue_creation_infos.emplace_back( std::move( queueCreateInfo ) );
	}

	std::vector<const char*> enabledDeviceExtensionNames;

	{
		// We consolidate all requested device extensions in a set,
		// so that we can be sure that each of the names requested
		// is unique.

		auto const extensions_names_end = extension_names + extension_names_count;

		for ( auto ext = extension_names; ext != extensions_names_end; ++ext ) {
			self->requestedDeviceExtensions.insert( *ext );
		}

		// We then copy the strings with the names for requested extensions
		// into this object's storage, so that we can be sure the pointers
		// will not go stale.

		enabledDeviceExtensionNames.reserve( self->requestedDeviceExtensions.size() );

		logger.info( "Enabled Device Extensions:" );
		for ( auto const& ext : self->requestedDeviceExtensions ) {
			enabledDeviceExtensionNames.emplace_back( ext.c_str() );
			logger.info( "\t + %s", ext.c_str() );
		}
	}

	auto features = le_backend_vk::settings_i.get_requested_physical_device_features_chain();

	VkDeviceCreateInfo deviceCreateInfo = {
	    .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
	    .pNext                   = features, // this contains the features chain from settings
	    .flags                   = 0,
	    .queueCreateInfoCount    = uint32_t( device_queue_creation_infos.size() ),
	    .pQueueCreateInfos       = device_queue_creation_infos.data(), // note that queues are created with device, once the device is created its number of queues is immutable.
	    .enabledLayerCount       = 0,
	    .ppEnabledLayerNames     = 0,
	    .enabledExtensionCount   = uint32_t( enabledDeviceExtensionNames.size() ),
	    .ppEnabledExtensionNames = enabledDeviceExtensionNames.data(),
	    .pEnabledFeatures        = nullptr, // must be nullptr, as we're using pNext for the features chain
	};

	// Create device
	vkCreateDevice( self->vkPhysicalDevice, &deviceCreateInfo, nullptr, &self->vkDevice );

	// load device pointers directly, to bypass the device dispatcher for better performance
	volkLoadDevice( self->vkDevice );

	// Store queue flags, and queue family index per queue into renderer properties,
	// so that queue capabilities and family index may be queried thereafter.

	self->queues.resize( available_queues.size() );
	self->queues_family_index.resize( available_queues.size() );
	self->queues_flags.resize( available_queues.size() );

	// Fetch queue handle into self->queues[i], matching indices with available_queues
	for ( size_t i = 0; i != available_queues.size(); i++ ) {
		vkGetDeviceQueue( self->vkDevice, available_queues[ i ].queue_family_index, available_queues[ i ].queue_index, &self->queues[ i ] );
		self->queues_family_index[ i ] = available_queues[ i ].queue_family_index;
		self->queues_flags[ i ]        = available_queues[ i ].queue_family_flags;
	}

	auto find_first_supported_format = []( le_device_o* device, std::vector<VkFormat> const& formats, VkFormatFeatureFlags const& required_flags ) {
		for ( auto& format : formats ) {
			VkFormatProperties2 props2 = {
			    .sType            = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
			    .pNext            = nullptr, // optional
			    .formatProperties = {},
			};
			vkGetPhysicalDeviceFormatProperties2( device->vkPhysicalDevice, format, &props2 );
			// Format must support optimal tiling
			if ( props2.formatProperties.optimalTilingFeatures & required_flags ) {
				return VkFormat( format );
				break;
			}
		}
		return VkFormat( VK_FORMAT_UNDEFINED );
	};

	self->defaultDepthStencilFormat =
	    find_first_supported_format(
	        self,
	        {
	            VK_FORMAT_D32_SFLOAT_S8_UINT,
	            VK_FORMAT_D32_SFLOAT,
	            VK_FORMAT_D24_UNORM_S8_UINT,
	            VK_FORMAT_D16_UNORM,
	            VK_FORMAT_D16_UNORM_S8_UINT,
	        },
	        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT );

	if ( self->defaultDepthStencilFormat == VK_FORMAT_UNDEFINED ) {
		le::Log( LOGGER_LABEL ).error( "Could not figure out default depth stencil image format" );
		exit( 1 );
	}

	self->defaultColorAttachmentFormat =
	    find_first_supported_format(
	        self,
	        {
	            VK_FORMAT_B8G8R8A8_SRGB,
	            VK_FORMAT_R8G8B8A8_SRGB,
	            VK_FORMAT_B8G8R8A8_UNORM,
	            VK_FORMAT_R8G8B8A8_UNORM,
	        },
	        VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT );

	if ( self->defaultColorAttachmentFormat == VK_FORMAT_UNDEFINED ) {
		le::Log( LOGGER_LABEL ).error( "Could not figure out default color attachment image format" );
		exit( 1 );
	}

	self->defaultSampledImageFormat =
	    find_first_supported_format(
	        self,
	        {
	            VK_FORMAT_R8G8B8A8_UNORM,
	            VK_FORMAT_B8G8R8A8_UNORM,
	            VK_FORMAT_B8G8R8A8_SRGB,
	            VK_FORMAT_R8G8B8A8_SRGB,
	        },
	        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT );

	if ( self->defaultSampledImageFormat == VK_FORMAT_UNDEFINED ) {
		le::Log( LOGGER_LABEL ).error( "Could not figure out default sampled image format" );
		exit( 1 );
	}

	return self;
};

// ----------------------------------------------------------------------

static le_device_o* device_increase_reference_count( le_device_o* self ) {
	++self->referenceCount;
	return self;
}

// ----------------------------------------------------------------------

static void device_destroy( le_device_o* self ) {
	self->queues.clear();
	vkDestroyDevice( self->vkDevice, nullptr );
	delete ( self );
};

// ----------------------------------------------------------------------

static le_device_o* device_decrease_reference_count( le_device_o* self ) {

	--self->referenceCount;

	if ( self->referenceCount == 0 ) {
		device_destroy( self );
		return nullptr;
	} else {
		return self;
	}
}

// ----------------------------------------------------------------------

static uint32_t device_get_reference_count( le_device_o* self ) {
	return self->referenceCount;
}

// ----------------------------------------------------------------------

static VkDevice device_get_vk_device( le_device_o* self_ ) {
	return self_->vkDevice;
}

// ----------------------------------------------------------------------

static VkPhysicalDevice device_get_vk_physical_device( le_device_o* self_ ) {
	return self_->vkPhysicalDevice;
}

// ----------------------------------------------------------------------

static const VkPhysicalDeviceProperties* device_get_vk_physical_device_properties( le_device_o* self ) {
	return &self->properties.device_properties.properties;
}

// ----------------------------------------------------------------------

static const VkPhysicalDeviceMemoryProperties* device_get_vk_physical_device_memory_properties( le_device_o* self ) {
	return &self->properties.memory_properties.memoryProperties;
}

// ----------------------------------------------------------------------

static void device_get_physical_device_ray_tracing_properties( le_device_o* self, VkPhysicalDeviceRayTracingPipelinePropertiesKHR* properties ) {
	*properties = self->properties.raytracing_properties;
}

// ----------------------------------------------------------------------

static void device_get_queue_family_indices( le_device_o* self, uint32_t* family_indices, uint32_t* num_family_indices ) {
	if ( num_family_indices && *num_family_indices == self->queues.size() ) {
		memcpy( family_indices, self->queues_family_index.data(), sizeof( uint32_t ) * self->queues_family_index.size() );
	} else {
		if ( num_family_indices ) {
			*num_family_indices = self->queues.size();
		}
	}
}

static void le_device_get_queues_info( le_device_o* self, uint32_t* queue_count, VkQueue* queues, uint32_t* queues_family_index, VkQueueFlags* queues_flags ) {
	if ( queue_count ) {
		*queue_count = self->queues.size();
	}

	assert( self->queues.size() == self->queues_family_index.size() &&
	        self->queues.size() == self->queues_flags.size() &&
	        "Queue SOA members must have equal length." );

	if ( queues ) {
		memcpy( queues, self->queues.data(), sizeof( VkQueue ) * self->queues.size() );
	}
	if ( queues_family_index ) {
		memcpy( queues_family_index, self->queues_family_index.data(), sizeof( uint32_t ) * self->queues_family_index.size() );
	}
	if ( queues_flags ) {
		memcpy( queues_flags, self->queues_flags.data(), sizeof( VkQueueFlags ) * self->queues_flags.size() );
	}
}
// ----------------------------------------------------------------------
LE_WRAP_ENUM_IN_STRUCT( VkFormat, VkFormatEnum ); // define wrapper struct `VkFormatEnum`

static void device_get_default_image_formats( le_device_o* self, VkFormatEnum* format_color_attachment, VkFormatEnum* format_depth_stencil_attachment, VkFormatEnum* format_sampled_image ) {
	*format_depth_stencil_attachment = reinterpret_cast<VkFormatEnum&>( self->defaultDepthStencilFormat );
	*format_color_attachment         = reinterpret_cast<VkFormatEnum&>( self->defaultColorAttachmentFormat );
	*format_sampled_image            = reinterpret_cast<VkFormatEnum&>( self->defaultSampledImageFormat );
}

// ----------------------------------------------------------------------

// get memory allocation info for best matching memory type that matches any of the type bits and flags
static bool device_get_memory_allocation_info( le_device_o*                self,
                                               const VkMemoryRequirements& memReqs,
                                               const VkFlags&              memPropsRef,
                                               VkMemoryAllocateInfo*       pMemoryAllocationInfo ) {

	if ( !memReqs.size ) {
		pMemoryAllocationInfo->allocationSize  = 0;
		pMemoryAllocationInfo->memoryTypeIndex = ~0u;
		return true;
	}

	const auto& physicalMemProperties = self->properties.memory_properties.memoryProperties;

	const VkMemoryPropertyFlags memProps{ reinterpret_cast<const VkMemoryPropertyFlags&>( memPropsRef ) };

	// Find an available memory type that satisfies the requested properties.
	uint32_t memoryTypeIndex;
	for ( memoryTypeIndex = 0; memoryTypeIndex < physicalMemProperties.memoryTypeCount; ++memoryTypeIndex ) {
		if ( ( memReqs.memoryTypeBits & ( 1 << memoryTypeIndex ) ) &&
		     ( physicalMemProperties.memoryTypes[ memoryTypeIndex ].propertyFlags & memProps ) == memProps ) {
			break;
		}
	}
	if ( memoryTypeIndex >= physicalMemProperties.memoryTypeCount ) {
		static auto logger = LeLog( LOGGER_LABEL );
		logger.error( "%s: MemoryTypeIndex not found", __PRETTY_FUNCTION__ );
		return false;
	}

	pMemoryAllocationInfo->allocationSize  = memReqs.size;
	pMemoryAllocationInfo->memoryTypeIndex = memoryTypeIndex;

	return true;
}

// ----------------------------------------------------------------------

static bool device_is_extension_available( le_device_o* self, char const* extension_name ) {
	return self->requestedDeviceExtensions.find( extension_name ) != self->requestedDeviceExtensions.end();
}

// ----------------------------------------------------------------------

void register_le_device_vk_api( void* api_ ) {
	auto  api_i    = static_cast<le_backend_vk_api*>( api_ );
	auto& device_i = api_i->vk_device_i;

	device_i.create                                        = device_create;
	device_i.destroy                                       = device_destroy;
	device_i.decrease_reference_count                      = device_decrease_reference_count;
	device_i.increase_reference_count                      = device_increase_reference_count;
	device_i.get_reference_count                           = device_get_reference_count;
	device_i.get_queue_family_indices                      = device_get_queue_family_indices;
	device_i.get_queues_info                               = le_device_get_queues_info;
	device_i.get_default_image_formats                     = device_get_default_image_formats;
	device_i.get_vk_physical_device                        = device_get_vk_physical_device;
	device_i.get_vk_device                                 = device_get_vk_device;
	device_i.get_vk_physical_device_properties             = device_get_vk_physical_device_properties;
	device_i.get_vk_physical_device_memory_properties      = device_get_vk_physical_device_memory_properties;
	device_i.get_vk_physical_device_ray_tracing_properties = device_get_physical_device_ray_tracing_properties;
	device_i.get_memory_allocation_info                    = device_get_memory_allocation_info;
	device_i.is_extension_available                        = device_is_extension_available;
}
