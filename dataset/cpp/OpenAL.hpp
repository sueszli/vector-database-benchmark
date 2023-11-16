// Copyright (C) 2023 Jérôme "Lynix" Leclercq (lynix680@gmail.com)
// This file is part of the "Nazara Engine - Audio module"
// For conditions of distribution and use, see copyright notice in Config.hpp

#pragma once

#ifndef NAZARA_AUDIO_OPENAL_HPP
#define NAZARA_AUDIO_OPENAL_HPP

#if defined(NAZARA_AUDIO_OPENAL) || defined(NAZARA_AUDIO_BUILD)

// Inclusion of OpenAL headers
#include <AL/al.h>
#include <AL/alc.h>

#ifndef NAZARA_PLATFORM_WEB
// Emscripten ships its own set of OpenAL headers which does not include alext.h
#include <AL/alext.h>
#endif

#endif // NAZARA_AUDIO_OPENAL

#endif // NAZARA_AUDIO_OPENAL_HPP
