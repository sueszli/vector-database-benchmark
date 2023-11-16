#include <zeno/utils/nowarn.h>
#include "MyShader.hpp"
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/geometric.hpp"
#include "stdafx.hpp"
#include "main.hpp"
#include "IGraphic.hpp"
#include <Hg/FPSCounter.hpp>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <array>
#include <stb_image_write.h>
#include <Hg/OpenGL/stdafx.hpp>
#include <unordered_map>
#include "zenvisapi.hpp"
#include <spdlog/spdlog.h>
#include <zeno/types/MaterialObject.h>
#include <zeno/types/TextureObject.h>

#ifdef _WIN32
    #ifdef near
        #undef near
    #endif
    #ifdef far
        #undef far
    #endif
#endif

namespace zenvis{
unsigned int reflectFBO = 0;
unsigned int reflectRBO = 0;
unsigned int reflectResolution = 2048;
std::vector<glm::mat4> reflectViews;
std::vector<glm::mat4> reflectMVPs;
std::vector<unsigned int> reflectiveMaps;
extern std::vector<unsigned int> getReflectMaps()
{
    return reflectiveMaps;
}
extern void setReflectMVP(int i, glm::mat4 mvp)
{
    reflectMVPs[i] = mvp;
}
extern glm::mat4 reflectView(glm::vec3 camPos, glm::vec3 viewDir, glm::vec3 up, glm::vec3 planeCenter, glm::vec3 planeNormal)
{
    glm::vec3 v = glm::normalize(viewDir);
    glm::vec3 R = v - 2.0f*glm::dot(v, planeNormal) * planeNormal;
    glm::vec3 RC = camPos + planeNormal * 2.0f * glm::dot((planeCenter - camPos), planeNormal);
    glm::vec3 Ru = up - 2.0f * glm::dot(up, planeNormal) * planeNormal;
    return glm::lookAt(RC, RC+R, Ru);

}

struct ReflectivePlane{
    glm::vec3 n;
    glm::vec3 c;
};
extern glm::mat4 getReflectViewMat(int i)
{
    return reflectViews[i];
}
extern glm::mat4 getReflectMVP(int i)
{
    return reflectMVPs[i];
}
std::vector<int> reflectMask;
std::vector<ReflectivePlane> ReflectivePlanes;
extern std::vector<ReflectivePlane> getReflectivePlanes()
{
    return ReflectivePlanes;
}
extern glm::vec3 getReflectiveNormal(int i)
{
    return ReflectivePlanes[i].n;
}
extern glm::vec3 getReflectiveCenter(int i)
{
    return ReflectivePlanes[i].c;
}
extern void setReflectivePlane(int i, glm::vec3 n, glm::vec3 c)
{
    
    if(i<0||i>=16)
        return;
    reflectMask[i] = 1;
    ReflectivePlane p;
    p.n = n; p.c = c;
    ReflectivePlanes[i] = p;

}
extern void clearReflectMask()
{
    reflectMask.assign(16,0);
}
extern bool renderReflect(int i)
{
    return reflectMask[i] == 1 ;
}
extern int getReflectivePlaneCount()
{ 
  return ReflectivePlanes.size();  
}
extern void setReflectivePlane(int i, glm::vec3 n, glm::vec3 c, glm::vec3 camPos, glm::vec3 camView, glm::vec3 camUp)
{
    reflectMask[i]=1;
    ReflectivePlane p;
    p.n = n; p.c = c;
    ReflectivePlanes[i] = p;
    reflectViews[i] = reflectView(camPos, camView, camUp, p.c, p.n);
}
extern void setReflectivePlane(int i, glm::vec3 camPos, glm::vec3 camView, glm::vec3 camUp)
{
    reflectMask[i] = 1;
    setReflectivePlane(i, ReflectivePlanes[i].n, ReflectivePlanes[i].c, camPos, camView, camUp);
}
static int reflectionID = -1;
extern void setReflectionViewID(int id)
{
    reflectionID = id;
}
extern int getReflectionViewID()
{
    return reflectionID;
}
static int mnx, mny;
static int moldnx, moldny;
extern void initReflectiveMaps(int nx, int ny)
{
    mnx = nx, mny = ny;
    moldnx = nx, moldny = ny;
    reflectiveMaps.resize(16);
    ReflectivePlanes.resize(16);
    reflectViews.resize(16);
    reflectMVPs.resize(16);
    reflectMask.assign(16,0);
    if(reflectFBO==0 && reflectRBO==0){
        CHECK_GL(glGenFramebuffers(1, &reflectFBO));
        CHECK_GL(glGenRenderbuffers(1, &reflectRBO));

        CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, reflectFBO));
        CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, reflectRBO));
        CHECK_GL(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, reflectResolution, reflectResolution));
        CHECK_GL(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, reflectRBO));
        
        for(int i=0;i<reflectiveMaps.size();i++)
        {
            CHECK_GL(glGenTextures(1, &(reflectiveMaps[i])));
            CHECK_GL(glBindTexture(GL_TEXTURE_RECTANGLE, reflectiveMaps[i]));
            CHECK_GL(glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGB16F, nx, ny, 
            0, GL_RGB, GL_FLOAT, 0));
            CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
            CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
            CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
            CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
            

        }
    }
}
extern void updateReflectTexture(int nx, int ny)
{
    if(moldnx!=nx || moldny!=ny){
        CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
        
        for(int i=0;i<reflectiveMaps.size();i++)
        {
            CHECK_GL(glBindTexture(GL_TEXTURE_RECTANGLE, reflectiveMaps[i]));
            CHECK_GL(glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGB16F, nx, ny, 
            0, GL_RGB, GL_FLOAT, 0));
            CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
            CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
            CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
            CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
            

        }
        moldnx = nx;
        moldny = ny;
    }
}
extern void BeginReflective(int i, int nx, int ny)
{
    CHECK_GL(glDisable(GL_BLEND));
    CHECK_GL(glDisable(GL_DEPTH_TEST));
    CHECK_GL(glDisable(GL_PROGRAM_POINT_SIZE));///????ZHXX???
    CHECK_GL(glDisable(GL_MULTISAMPLE));
    CHECK_GL(glEnable(GL_DEPTH_TEST));
    CHECK_GL(glDepthFunc(GL_LESS));
    

    CHECK_GL(glBindTexture(GL_TEXTURE_RECTANGLE, reflectiveMaps[i]));
    CHECK_GL(glViewport(0, 0, nx, ny));
    CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, reflectFBO));
    CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, reflectRBO));
    CHECK_GL(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, nx, ny));
    CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, reflectiveMaps[i], 0));
    

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
extern void BeginSecondReflective(int i, int nx, int ny)
{
    CHECK_GL(glDisable(GL_BLEND));
    CHECK_GL(glDisable(GL_DEPTH_TEST));
    CHECK_GL(glDisable(GL_PROGRAM_POINT_SIZE));///????ZHXX???
    CHECK_GL(glDisable(GL_MULTISAMPLE));
    CHECK_GL(glEnable(GL_DEPTH_TEST));
    CHECK_GL(glDepthFunc(GL_LESS));
    

    CHECK_GL(glBindTexture(GL_TEXTURE_RECTANGLE, reflectiveMaps[i+8]));
    CHECK_GL(glViewport(0, 0, nx, ny));
    CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, reflectFBO));
    CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, reflectRBO));
    CHECK_GL(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, nx, ny));
    CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, reflectiveMaps[i+8], 0));
    

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
extern void EndReflective()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(0);
    CHECK_GL(glEnable(GL_BLEND));
    CHECK_GL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    CHECK_GL(glEnable(GL_DEPTH_TEST));
    CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE));
    //CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE_ARB));
    //CHECK_GL(glEnable(GL_POINT_SPRITE_ARB));
    //CHECK_GL(glEnable(GL_SAMPLE_COVERAGE));
    //CHECK_GL(glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE));
    //CHECK_GL(glEnable(GL_SAMPLE_ALPHA_TO_ONE));
    CHECK_GL(glEnable(GL_MULTISAMPLE));

}
extern void EndSecondReflective()
{
    for(int i=0;i<8;i++)
    {

        auto temp = reflectiveMaps[i];
        reflectiveMaps[i]=reflectiveMaps[i+8];
        reflectiveMaps[i+8]=temp;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(0);
    CHECK_GL(glEnable(GL_BLEND));
    CHECK_GL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    CHECK_GL(glEnable(GL_DEPTH_TEST));
    CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE));
    //CHECK_GL(glEnable(GL_PROGRAM_POINT_SIZE_ARB));
    //CHECK_GL(glEnable(GL_POINT_SPRITE_ARB));
    //CHECK_GL(glEnable(GL_SAMPLE_COVERAGE));
    //CHECK_GL(glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE));
    //CHECK_GL(glEnable(GL_SAMPLE_ALPHA_TO_ONE));
    CHECK_GL(glEnable(GL_MULTISAMPLE));

}

void printMat4(glm::mat4 &m)
{
    for(int j=0;j<4;j++)
    {
        for(int i=0;i<4;i++)
        {
            std::cout<<m[j][i]<<" ";
        }
        std::cout<<std::endl;
    }
}

}
