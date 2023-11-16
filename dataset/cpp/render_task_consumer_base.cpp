﻿//
// Created by captainchen on 2022/2/7.
//

#include "render_task_consumer_base.h"
#include <iostream>
#include <glm/gtc/matrix_transform.hpp>
#include "timetool/stopwatch.h"
#include "utils/debug.h"
#include "render_task_type.h"
#include "render_command.h"
#include "gpu_resource_mapper.h"
#include "render_task_queue.h"
#include "utils/screen.h"
#include "render_device/uniform_buffer_object_manager.h"

void RenderTaskConsumerBase::Init() {
    render_thread_ = std::thread(&RenderTaskConsumerBase::ProcessTask,this);
    render_thread_.detach();
}

void RenderTaskConsumerBase::InitGraphicsLibraryFramework() {
    
}

void RenderTaskConsumerBase::Exit() {
    exit_=true;
    if (render_thread_.joinable()) {
        render_thread_.join();//等待渲染线程结束
    }
}

void RenderTaskConsumerBase::GetFramebufferSize(int& width,int& height) {
    
}

void RenderTaskConsumerBase::SwapBuffer() {
    
}

/// 更新游戏画面尺寸
void RenderTaskConsumerBase::UpdateScreenSize(RenderTaskBase* task_base) {
    RenderTaskUpdateScreenSize* task= dynamic_cast<RenderTaskUpdateScreenSize*>(task_base);
    int width, height;
    GetFramebufferSize(width, height);
    glViewport(0, 0, width, height);
    Screen::set_width_height(width,height);
}

/// 编译、链接Shader
/// \param task_base
void RenderTaskConsumerBase::CompileShader(RenderTaskBase* task_base){
    RenderTaskCompileShader* task= dynamic_cast<RenderTaskCompileShader*>(task_base);
    const char* vertex_shader_text=task->vertex_shader_source_;
    const char* fragment_shader_text=task->fragment_shader_source_;

    //创建顶点Shader
    unsigned int vertex_shader = glCreateShader(GL_VERTEX_SHADER);__CHECK_GL_ERROR__
    //指定Shader源码
    glShaderSource(vertex_shader, 1, &vertex_shader_text, NULL);__CHECK_GL_ERROR__
    //编译Shader
    glCompileShader(vertex_shader);__CHECK_GL_ERROR__
    //获取编译结果
    GLint compile_status=GL_FALSE;
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &compile_status);
    if (compile_status == GL_FALSE)
    {
        GLchar message[256];
        glGetShaderInfoLog(vertex_shader, sizeof(message), 0, message);
        DEBUG_LOG_ERROR("compile vertex shader error:{}",message);
        return;
    }

    //创建片段Shader
    unsigned int fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);__CHECK_GL_ERROR__
    //指定Shader源码
    glShaderSource(fragment_shader, 1, &fragment_shader_text, NULL);__CHECK_GL_ERROR__
    //编译Shader
    glCompileShader(fragment_shader);__CHECK_GL_ERROR__
    //获取编译结果
    compile_status=GL_FALSE;
    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &compile_status);
    if (compile_status == GL_FALSE)
    {
        GLchar message[256];
        glGetShaderInfoLog(fragment_shader, sizeof(message), 0, message);
        DEBUG_LOG_ERROR("compile fragment shader error:{}",message);
        return;
    }

    //创建Shader程序
    GLuint shader_program = glCreateProgram();__CHECK_GL_ERROR__
    //附加Shader
    glAttachShader(shader_program, vertex_shader);__CHECK_GL_ERROR__
    glAttachShader(shader_program, fragment_shader);__CHECK_GL_ERROR__
    //Link
    glLinkProgram(shader_program);__CHECK_GL_ERROR__
    //获取编译结果
    GLint link_status=GL_FALSE;
    glGetProgramiv(shader_program, GL_LINK_STATUS, &link_status);
    if (link_status == GL_FALSE)
    {
        GLchar message[256];
        glGetProgramInfoLog(shader_program, sizeof(message), 0, message);
        DEBUG_LOG_ERROR("link shader error:{}",message);
        return;
    }
    //将主线程中产生的Shader程序句柄 映射到 Shader程序
    GPUResourceMapper::MapShaderProgram(task->shader_program_handle_, shader_program);
}

void RenderTaskConsumerBase::ConnectUniformBlockInstanceAndBindingPoint(RenderTaskBase *task_base) {
    RenderTaskConnectUniformBlockInstanceAndBindingPoint* task= dynamic_cast<RenderTaskConnectUniformBlockInstanceAndBindingPoint*>(task_base);
    GLuint shader_program = GPUResourceMapper::GetShaderProgram(task->shader_program_handle_);

    std::vector<UniformBlockInstanceBindingInfo> uniform_block_instance_binding_info_array= UniformBufferObjectManager::UniformBlockInstanceBindingInfoArray();
    for (int i = 0; i < uniform_block_instance_binding_info_array.size(); ++i) {
        //找到UniformBlock在当前Shader程序的index。(注意这里是用uniform_block_name_，而不是uniform_block_instance_name_)
        std::string uniform_block_name=uniform_block_instance_binding_info_array[i].uniform_block_name_;
        GLuint uniform_block_index = glGetUniformBlockIndex(shader_program, uniform_block_name.c_str());__CHECK_GL_ERROR__
        if(uniform_block_index==GL_INVALID_INDEX){//当前Shader程序没有这个UniformBlock
            continue;
        }
        //关联当前Shader的UniformBlock到BindingPoint，这样间接与UniformBufferObject有了联系。
        GLuint uniform_block_binding_point=uniform_block_instance_binding_info_array[i].binding_point_;
        glUniformBlockBinding(shader_program, uniform_block_index, uniform_block_binding_point);__CHECK_GL_ERROR__
    }
}

void RenderTaskConsumerBase::UseShaderProgram(RenderTaskBase *task_base) {
    RenderTaskUseShaderProgram* task= dynamic_cast<RenderTaskUseShaderProgram*>(task_base);
    GLuint shader_program = GPUResourceMapper::GetShaderProgram(task->shader_program_handle_);
    glUseProgram(shader_program);__CHECK_GL_ERROR__
}

void RenderTaskConsumerBase::CreateCompressedTexImage2D(RenderTaskBase *task_base) {
    RenderTaskCreateCompressedTexImage2D* task= dynamic_cast<RenderTaskCreateCompressedTexImage2D*>(task_base);

    GLuint texture_id;

    //1. 通知显卡创建纹理对象，返回句柄;
    glGenTextures(1, &texture_id);__CHECK_GL_ERROR__

    //2. 将纹理绑定到特定纹理目标;
    glBindTexture(GL_TEXTURE_2D, texture_id);__CHECK_GL_ERROR__

    //3. 将压缩纹理数据上传到GPU;
    glCompressedTexImage2D(GL_TEXTURE_2D, 0, task->texture_format_, task->width_, task->height_, 0, task->compress_size_, task->data_);
    __CHECK_GL_ERROR__

    //4. 指定放大，缩小滤波方式，线性滤波，即放大缩小的插值方式;
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);__CHECK_GL_ERROR__
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);__CHECK_GL_ERROR__

    //将主线程中产生的压缩纹理句柄 映射到 纹理
    GPUResourceMapper::MapTexture(task->texture_handle_, texture_id);
}

void RenderTaskConsumerBase::CreateTexImage2D(RenderTaskBase *task_base) {
    RenderTaskCreateTexImage2D* task= dynamic_cast<RenderTaskCreateTexImage2D*>(task_base);

    GLuint texture_id;

    //1. 通知显卡创建纹理对象，返回句柄;
    glGenTextures(1, &texture_id);__CHECK_GL_ERROR__

    //2. 将纹理绑定到特定纹理目标;
    glBindTexture(GL_TEXTURE_2D, texture_id);__CHECK_GL_ERROR__

    //3. 将图片rgb数据上传到GPU;
    glTexImage2D(GL_TEXTURE_2D, 0, task->gl_texture_format_, task->width_, task->height_, 0, task->client_format_, task->data_type_, task->data_);__CHECK_GL_ERROR__

    //4. 指定放大，缩小滤波方式，线性滤波，即放大缩小的插值方式;
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);__CHECK_GL_ERROR__
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);__CHECK_GL_ERROR__

    //将主线程中产生的纹理句柄 映射到 纹理
    GPUResourceMapper::MapTexture(task->texture_handle_, texture_id);
}

/// 删除Textures
/// \param task_base
void RenderTaskConsumerBase::DeleteTextures(RenderTaskBase *task_base) {
    RenderTaskDeleteTextures* task= dynamic_cast<RenderTaskDeleteTextures*>(task_base);
    //从句柄转换到纹理对象
    GLuint* texture_id_array=new GLuint[task->texture_count_];
    for (int i = 0; i < task->texture_count_; ++i) {
        texture_id_array[i]=GPUResourceMapper::GetTexture(task->texture_handle_array_[i]);
    }
    glDeleteTextures(task->texture_count_,texture_id_array);__CHECK_GL_ERROR__
    delete [] texture_id_array;
}

/// 局部更新纹理
/// \param task_base
void RenderTaskConsumerBase::UpdateTextureSubImage2D(RenderTaskBase *task_base) {
    RenderTaskUpdateTextureSubImage2D* task= dynamic_cast<RenderTaskUpdateTextureSubImage2D*>(task_base);
    GLuint texture=GPUResourceMapper::GetTexture(task->texture_handle_);
    glBindTexture(GL_TEXTURE_2D, texture);__CHECK_GL_ERROR__
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);__CHECK_GL_ERROR__
    glTexSubImage2D(GL_TEXTURE_2D,0,task->x_,task->y_,task->width_,task->height_,task->client_format_,task->data_type_,task->data_);__CHECK_GL_ERROR__
}

void RenderTaskConsumerBase::CreateVAO(RenderTaskBase *task_base) {
    RenderTaskCreateVAO* task=dynamic_cast<RenderTaskCreateVAO*>(task_base);
    GLuint shader_program=GPUResourceMapper::GetShaderProgram(task->shader_program_handle_);
    GLint attribute_pos_location = glGetAttribLocation(shader_program, "a_pos");__CHECK_GL_ERROR__
    GLint attribute_color_location = glGetAttribLocation(shader_program, "a_color");__CHECK_GL_ERROR__
    GLint attribute_uv_location = glGetAttribLocation(shader_program, "a_uv");__CHECK_GL_ERROR__
    GLint attribute_normal_location = glGetAttribLocation(shader_program, "a_normal");__CHECK_GL_ERROR__

    GLuint vertex_buffer_object,element_buffer_object,vertex_array_object;
    //在GPU上创建缓冲区对象
    glGenBuffers(1,&vertex_buffer_object);__CHECK_GL_ERROR__
    //将缓冲区对象指定为顶点缓冲区对象
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object);__CHECK_GL_ERROR__
    //上传顶点数据到缓冲区对象
    glBufferData(GL_ARRAY_BUFFER, task->vertex_data_size_, task->vertex_data_, GL_DYNAMIC_DRAW);__CHECK_GL_ERROR__
    //将主线程中产生的VBO句柄 映射到 VBO
    GPUResourceMapper::MapVBO(task->vbo_handle_, vertex_buffer_object);

    //在GPU上创建缓冲区对象
    glGenBuffers(1,&element_buffer_object);__CHECK_GL_ERROR__
    //将缓冲区对象指定为顶点索引缓冲区对象
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_object);__CHECK_GL_ERROR__
    //上传顶点索引数据到缓冲区对象
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, task->vertex_index_data_size_, task->vertex_index_data_, GL_STATIC_DRAW);__CHECK_GL_ERROR__

    glGenVertexArrays(1,&vertex_array_object);__CHECK_GL_ERROR__

    //设置VAO
    glBindVertexArray(vertex_array_object);__CHECK_GL_ERROR__
    {
        //指定当前使用的VBO
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object);__CHECK_GL_ERROR__
        //将Shader变量(a_pos)和顶点坐标VBO句柄进行关联，最后的0表示数据偏移量。
        glVertexAttribPointer(attribute_pos_location, 3, GL_FLOAT, false, task->vertex_data_stride_, 0);__CHECK_GL_ERROR__
        //启用顶点Shader属性(a_color)，指定与顶点颜色数据进行关联
        if(attribute_color_location>=0){
            glVertexAttribPointer(attribute_color_location, 4, GL_FLOAT, false, task->vertex_data_stride_, (void*)(sizeof(float) * 3));__CHECK_GL_ERROR__
        }
        //将Shader变量(a_uv)和顶点UV坐标VBO句柄进行关联，最后的0表示数据偏移量。
        glVertexAttribPointer(attribute_uv_location, 2, GL_FLOAT, false, task->vertex_data_stride_, (void*)(sizeof(float) * (3 + 4)));__CHECK_GL_ERROR__
        //将Shader变量(a_normal)和顶点法线VBO句柄进行关联，最后的0表示数据偏移量。
        if(attribute_normal_location>=0) {
            glVertexAttribPointer(attribute_normal_location, 3, GL_FLOAT, false, task->vertex_data_stride_,(void *) (sizeof(float) * (3 + 4 + 2)));__CHECK_GL_ERROR__
        }

        glEnableVertexAttribArray(attribute_pos_location);__CHECK_GL_ERROR__
        if(attribute_color_location>=0){
            glEnableVertexAttribArray(attribute_color_location);__CHECK_GL_ERROR__
        }
        glEnableVertexAttribArray(attribute_uv_location);__CHECK_GL_ERROR__
        if(attribute_normal_location>=0){
            glEnableVertexAttribArray(attribute_normal_location);__CHECK_GL_ERROR__
        }

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_object);__CHECK_GL_ERROR__
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);__CHECK_GL_ERROR__
    //将主线程中产生的VAO句柄 映射到 VAO
    GPUResourceMapper::MapVAO(task->vao_handle_, vertex_array_object);
}

void RenderTaskConsumerBase::UpdateVBOSubData(RenderTaskBase *task_base) {
    RenderTaskUpdateVBOSubData* task=dynamic_cast<RenderTaskUpdateVBOSubData*>(task_base);
    GLuint vbo=GPUResourceMapper::GetVBO(task->vbo_handle_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);__CHECK_GL_ERROR__
    timetool::StopWatch stopwatch;
    stopwatch.start();
    //更新Buffer数据
    glBufferSubData(GL_ARRAY_BUFFER,0,task->vertex_data_size_,task->vertex_data_);__CHECK_GL_ERROR__
    stopwatch.stop();
//    DEBUG_LOG_INFO("glBufferSubData cost {}",stopwatch.microseconds());
}

void RenderTaskConsumerBase::CreateUBO(RenderTaskBase *task_base) {
    RenderTaskCreateUBO* task=dynamic_cast<RenderTaskCreateUBO*>(task_base);
    GLuint shader_program=GPUResourceMapper::GetShaderProgram(task->shader_program_handle_);

    //将UniformBlock索引 和 point 绑定
    GLchar* uniform_name=task->uniform_block_name_;
    GLuint uniform_block_index = glGetUniformBlockIndex(shader_program, uniform_name);
    GLuint uniform_block_binding_point=1;
    glUniformBlockBinding(shader_program, uniform_block_index, uniform_block_binding_point);

    //创建与绑定缓冲区
    GLuint uniform_buffer_object;
    glGenBuffers(1, &uniform_buffer_object);
    glBindBuffer(GL_UNIFORM_BUFFER, uniform_buffer_object);
    //向缓冲区中赋值
    glBufferData(GL_UNIFORM_BUFFER, task->uniform_block_data_size_, task->uniform_block_data_, GL_DYNAMIC_DRAW);
    //将UBO 和 point 绑定
    glBindBufferBase(GL_UNIFORM_BUFFER, uniform_block_binding_point, uniform_buffer_object);

    //将主线程中产生的UBO句柄 映射到 UBO
    GPUResourceMapper::MapUBO(task->ubo_handle_, uniform_buffer_object);
}

/// 更新UBO
/// \param task_base
void RenderTaskConsumerBase::UpdateUBOSubData(RenderTaskBase* task_base){
    RenderTaskUpdateUBOSubData* task=dynamic_cast<RenderTaskUpdateUBOSubData*>(task_base);

    std::vector<UniformBlockInstanceBindingInfo>& uniform_block_instance_binding_info_array= UniformBufferObjectManager::UniformBlockInstanceBindingInfoArray();
    std::unordered_map<std::string,UniformBlock>& uniform_block_map= UniformBufferObjectManager::UniformBlockMap();

    for (int i = 0; i < uniform_block_instance_binding_info_array.size(); ++i) {
        //找到UniformBlock实例信息
        UniformBlockInstanceBindingInfo& uniform_block_instance_binding_info=uniform_block_instance_binding_info_array[i];
        if(uniform_block_instance_binding_info.uniform_block_instance_name_ != task->uniform_block_instance_name_){
            continue;
        }

        glBindBuffer(GL_UNIFORM_BUFFER, uniform_block_instance_binding_info.uniform_buffer_object_);__CHECK_GL_ERROR__

        //以UniformBlock实例对应的UniformBlock名，获取对应的结构信息。
        UniformBlock uniform_block=uniform_block_map[uniform_block_instance_binding_info.uniform_block_name_];
        for (auto& uniform_block_member:uniform_block.uniform_block_member_vec_) {
            //找到对应的成员变量。
            if(uniform_block_member.member_name_!=task->uniform_block_member_name_){
                continue;
            }

            //更新对应成员变量数据。
            glBufferSubData(GL_UNIFORM_BUFFER, uniform_block_member.offset_, uniform_block_member.data_size_, task->data);__CHECK_GL_ERROR__
        }

        glBindBuffer(GL_UNIFORM_BUFFER, 0);__CHECK_GL_ERROR__
    }
}

void RenderTaskConsumerBase::SetEnableState(RenderTaskBase *task_base) {
    RenderTaskSetEnableState* task= dynamic_cast<RenderTaskSetEnableState*>(task_base);
    if(task->enable_){
        glEnable(task->state_);__CHECK_GL_ERROR__
    }else{
        glDisable(task->state_);__CHECK_GL_ERROR__
    }
}

void RenderTaskConsumerBase::SetBlendFunc(RenderTaskBase *task_base) {
    RenderTaskSetBlenderFunc* task= dynamic_cast<RenderTaskSetBlenderFunc*>(task_base);
    glBlendFunc(task->source_blending_factor_, task->destination_blending_factor_);__CHECK_GL_ERROR__
}

void RenderTaskConsumerBase::SetUniformMatrix4fv(RenderTaskBase *task_base) {
    RenderTaskSetUniformMatrix4fv* task=dynamic_cast<RenderTaskSetUniformMatrix4fv*>(task_base);
    //上传mvp矩阵
    GLuint shader_program=GPUResourceMapper::GetShaderProgram(task->shader_program_handle_);
    GLint uniform_location=glGetUniformLocation(shader_program, task->uniform_name_);__CHECK_GL_ERROR__
    glUniformMatrix4fv(uniform_location, 1, task->transpose_?GL_TRUE:GL_FALSE, &task->matrix_[0][0]);__CHECK_GL_ERROR__
}

void RenderTaskConsumerBase::ActiveAndBindTexture(RenderTaskBase *task_base) {
    RenderTaskActiveAndBindTexture* task=dynamic_cast<RenderTaskActiveAndBindTexture*>(task_base);
    //激活纹理单元
    glActiveTexture(task->texture_uint_);__CHECK_GL_ERROR__
    //将加载的图片纹理句柄，绑定到纹理单元0的Texture2D上。
    GLuint texture=GPUResourceMapper::GetTexture(task->texture_handle_);
    glBindTexture(GL_TEXTURE_2D, texture);__CHECK_GL_ERROR__
}

void RenderTaskConsumerBase::SetUniform1i(RenderTaskBase *task_base) {
    RenderTaskSetUniform1i* task=dynamic_cast<RenderTaskSetUniform1i*>(task_base);
    GLuint shader_program=GPUResourceMapper::GetShaderProgram(task->shader_program_handle_);
    GLint uniform_location= glGetUniformLocation(shader_program, task->uniform_name_);__CHECK_GL_ERROR__
    glUniform1i(uniform_location, task->value_);__CHECK_GL_ERROR__
}

void RenderTaskConsumerBase::SetUniform1f(RenderTaskBase *task_base) {
    RenderTaskSetUniform1f* task=dynamic_cast<RenderTaskSetUniform1f*>(task_base);
    GLuint shader_program=GPUResourceMapper::GetShaderProgram(task->shader_program_handle_);
    GLint uniform_location= glGetUniformLocation(shader_program, task->uniform_name_);__CHECK_GL_ERROR__
    glUniform1f(uniform_location, task->value_);__CHECK_GL_ERROR__
}

void RenderTaskConsumerBase::SetUniform3f(RenderTaskBase *task_base) {
    RenderTaskSetUniform3f* task=dynamic_cast<RenderTaskSetUniform3f*>(task_base);
    GLuint shader_program=GPUResourceMapper::GetShaderProgram(task->shader_program_handle_);
    GLint uniform_location= glGetUniformLocation(shader_program, task->uniform_name_);__CHECK_GL_ERROR__
    glUniform3f(uniform_location, task->value_.x,task->value_.y,task->value_.z);__CHECK_GL_ERROR__
}

void RenderTaskConsumerBase::BindVAOAndDrawElements(RenderTaskBase *task_base) {
    RenderTaskBindVAOAndDrawElements* task=dynamic_cast<RenderTaskBindVAOAndDrawElements*>(task_base);
    GLuint vao=GPUResourceMapper::GetVAO(task->vao_handle_);
    glBindVertexArray(vao);__CHECK_GL_ERROR__
    {
        glDrawElements(GL_TRIANGLES,task->vertex_index_num_,GL_UNSIGNED_SHORT,0);__CHECK_GL_ERROR__//使用顶点索引进行绘制，最后的0表示数据偏移量。
    }
    glBindVertexArray(0);__CHECK_GL_ERROR__
}

/// 清除
/// \param task_base
void RenderTaskConsumerBase::SetClearFlagAndClearColorBuffer(RenderTaskBase* task_base){
    RenderTaskClear* task=dynamic_cast<RenderTaskClear*>(task_base);
    glClear(task->clear_flag_);__CHECK_GL_ERROR__
    glClearColor(task->clear_color_r_,task->clear_color_g_,task->clear_color_b_,task->clear_color_a_);__CHECK_GL_ERROR__
}

/// 设置模板测试函数
void RenderTaskConsumerBase::SetStencilFunc(RenderTaskBase* task_base){
    RenderTaskSetStencilFunc* task=dynamic_cast<RenderTaskSetStencilFunc*>(task_base);
    glStencilFunc(task->stencil_func_, task->stencil_ref_, task->stencil_mask_);__CHECK_GL_ERROR__
}

/// 设置模板操作
void RenderTaskConsumerBase::SetStencilOp(RenderTaskBase* task_base){
    RenderTaskSetStencilOp* task=dynamic_cast<RenderTaskSetStencilOp*>(task_base);
    glStencilOp(task->fail_op_, task->z_test_fail_op_, task->z_test_pass_op_);__CHECK_GL_ERROR__
}

void RenderTaskConsumerBase::SetStencilBufferClearValue(RenderTaskBase* task_base){
    RenderTaskSetStencilBufferClearValue* task=dynamic_cast<RenderTaskSetStencilBufferClearValue*>(task_base);
    glClearStencil(task->clear_value_);__CHECK_GL_ERROR__
}

/// 结束一帧
/// \param task_base
void RenderTaskConsumerBase::EndFrame(RenderTaskBase* task_base) {
    RenderTaskEndFrame *task = dynamic_cast<RenderTaskEndFrame *>(task_base);
    SwapBuffer();
    task->return_result_set_=true;
}

void RenderTaskConsumerBase::ProcessTask() {
    //渲染相关的API调用需要放到渲染线程中。
    InitGraphicsLibraryFramework();

    //初始化UBO
    UniformBufferObjectManager::Init();
    UniformBufferObjectManager::CreateUniformBufferObject();

    while (!exit_)
    {
        while(true){
            if(RenderTaskQueue::Empty()){//渲染线程一直等待主线程发出任务。
                std::this_thread::sleep_for(std::chrono::nanoseconds(1));//没有任务休息一下。
                continue;
            }
            RenderTaskBase* render_task = RenderTaskQueue::Front();
            RenderCommand render_command=render_task->render_command_;
            bool need_return_result=render_task->need_return_result_;
            switch (render_command) {//根据主线程发来的命令，做不同的处理
                case RenderCommand::NONE:break;
                case RenderCommand::UPDATE_SCREEN_SIZE:{
                    UpdateScreenSize(render_task);
                    break;
                }
                case RenderCommand::COMPILE_SHADER:{
                    CompileShader(render_task);
                    break;
                }
                case RenderCommand::CONNECT_UNIFORM_BLOCK_INSTANCE_AND_BINDING_POINT:{
                    ConnectUniformBlockInstanceAndBindingPoint(render_task);
                    break;
                }
                case RenderCommand::USE_SHADER_PROGRAM:{
                    UseShaderProgram(render_task);
                    break;
                }
                case RenderCommand::CREATE_COMPRESSED_TEX_IMAGE2D:{
                    CreateCompressedTexImage2D(render_task);
                    break;
                }
                case RenderCommand::CREATE_TEX_IMAGE2D:{
                    CreateTexImage2D(render_task);
                    break;
                }
                case RenderCommand::DELETE_TEXTURES:{
                    DeleteTextures(render_task);
                    break;
                }
                case RenderCommand::UPDATE_TEXTURE_SUB_IMAGE2D:{
                    UpdateTextureSubImage2D(render_task);
                    break;
                }
                case RenderCommand::CREATE_VAO:{
                    CreateVAO(render_task);
                    break;
                }
                case RenderCommand::UPDATE_VBO_SUB_DATA:{
                    UpdateVBOSubData(render_task);
                    break;
                }
                case RenderCommand::CREATE_UBO:{
                    CreateUBO(render_task);
                    break;
                }
                case RenderCommand::UPDATE_UBO_SUB_DATA:{
                    UpdateUBOSubData(render_task);
                    break;
                }
                case RenderCommand::SET_ENABLE_STATE:{
                    SetEnableState(render_task);
                    break;
                }
                case RenderCommand::SET_BLENDER_FUNC:{
                    SetBlendFunc(render_task);
                    break;
                }
                case RenderCommand::SET_UNIFORM_MATRIX_4FV:{
                    SetUniformMatrix4fv(render_task);
                    break;
                }
                case RenderCommand::ACTIVE_AND_BIND_TEXTURE:{
                    ActiveAndBindTexture(render_task);
                    break;
                }
                case RenderCommand::SET_UNIFORM_1I:{
                    SetUniform1i(render_task);
                    break;
                }
                case RenderCommand::SET_UNIFORM_1F:{
                    SetUniform1f(render_task);
                    break;
                }
                case RenderCommand::SET_UNIFORM_3F:{
                    SetUniform3f(render_task);
                    break;
                }
                case RenderCommand::SET_CLEAR_FLAG_AND_CLEAR_COLOR_BUFFER:{
                    SetClearFlagAndClearColorBuffer(render_task);
                    break;
                }
                case RenderCommand::BIND_VAO_AND_DRAW_ELEMENTS:{
                    BindVAOAndDrawElements(render_task);
                    break;
                }
                case RenderCommand::SET_STENCIL_FUNC:{
                    SetStencilFunc(render_task);
                    break;
                }
                case RenderCommand::SET_STENCIL_OP:{
                    SetStencilOp(render_task);
                    break;
                }
                case RenderCommand::SET_STENCIL_BUFFER_CLEAR_VALUE:{
                    SetStencilBufferClearValue(render_task);
                    break;
                }
                case RenderCommand::END_FRAME:{
                    EndFrame(render_task);
                    break;
                }
            }

            RenderTaskQueue::Pop();

            //如果这个任务不需要返回参数，那么用完就删掉。
            if(need_return_result==false){
                delete render_task;
            }
            //如果是帧结束任务，就交换缓冲区。
            if(render_command==RenderCommand::END_FRAME){
                break;
            }
        }
        std::cout<<"task in queue:"<<RenderTaskQueue::Size()<<std::endl;
    }
}