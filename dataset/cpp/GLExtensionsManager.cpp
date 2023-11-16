/****************************************************************************
 * MeshLab                                                           o o     *
 * An extendible mesh processor                                    o     o   *
 *                                                                _   O  _   *
 * Copyright(C) 2005, 2006                                          \/)\/    *
 * Visual Computing Lab                                            /\/|      *
 * ISTI - Italian National Research Council                           |      *
 *                                                                    \      *
 * All rights reserved.                                                      *
 *                                                                           *
 * This program is free software; you can redistribute it and/or modify      *
 * it under the terms of the GNU General Public License as published by      *
 * the Free Software Foundation; either version 2 of the License, or         *
 * (at your option) any later version.                                       *
 *                                                                           *
 * This program is distributed in the hope that it will be useful,           *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of            *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             *
 * GNU General Public License (http://www.gnu.org/licenses/gpl.txt)          *
 * for more details.                                                         *
 *                                                                           *
 ****************************************************************************/

#include "GLExtensionsManager.h"

bool GLExtensionsManager::glewInitialized;

void GLExtensionsManager::init()
{
    glewInitialized = false;
}

/**
 * @brief initializes the GL extensions context, without thorwing any exception
 * @return true if the context has been correctly initializes, false otherwise
 */
bool GLExtensionsManager::initializeGLextensions_notThrowing()
{
    if (!glewInitialized) {
        glewExperimental = GL_TRUE;
        GLenum err = glewInit();
        if ((err == GLEW_OK) ||
            (err == GLEW_ERROR_NO_GLX_DISPLAY)) {
            glewInitialized = true;
        }
        else {
            qWarning("GLEW initialization failed: %s",
                     (const char *)glewGetErrorString(err));
        }
    }
    return glewInitialized;

}

/**
 * @brief initializes the GL extensions context, thorwing a MLException if something
 * wrong happens during the initialization.
 */
void GLExtensionsManager::initializeGLextensions()
{
    if (!glewInitialized) {
        glewExperimental = GL_TRUE;
        GLenum err = glewInit();
        if ((err == GLEW_OK) ||
            (err == GLEW_ERROR_NO_GLX_DISPLAY)) {
            glewInitialized = true;
        }
        else {
            throw MLException(QString("GLEW initialization failed: %1\n")
                                  .arg((const char *)glewGetErrorString(err)));
        }
    }
}
