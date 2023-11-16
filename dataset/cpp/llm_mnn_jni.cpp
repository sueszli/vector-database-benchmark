#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <jni.h>
#include <string>
#include <vector>
#include <sstream>
#include <thread>

#include "llm.hpp"

static Llm* llm;
static std::stringstream response_buffer;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "MNN_DEBUG", "JNI_OnLoad");
    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "MNN_DEBUG", "JNI_OnUnload");
}

JNIEXPORT jboolean JNICALL Java_com_mnn_llm_Chat_Init(JNIEnv* env, jobject thiz, jstring modelDir) {
    if (llm->load_progress() < 100) {
        const char* model_dir = env->GetStringUTFChars(modelDir, 0);
        llm = Llm::createLLM(model_dir);
        llm->load(model_dir);
    }
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_mnn_llm_Chat_Ready(JNIEnv* env, jobject thiz) {
    if (llm->load_progress() >= 100) {
        return JNI_TRUE;
    }
    return JNI_FALSE;
}

JNIEXPORT jfloat JNICALL Java_com_mnn_llm_Chat_Progress(JNIEnv* env, jobject thiz) {
    return jfloat(llm->load_progress());
}

JNIEXPORT jstring JNICALL Java_com_mnn_llm_Chat_Submit(JNIEnv* env, jobject thiz, jstring inputStr) {
    if (llm->load_progress() < 100) {
        return env->NewStringUTF("Failed, Chat is not ready!");
    }
    const char* input_str = env->GetStringUTFChars(inputStr, 0);
    auto chat = [&](std::string str) {
        llm->response(str, &response_buffer);
    };
    std::thread chat_thread(chat, input_str);
    chat_thread.detach();
    jstring result = env->NewStringUTF("Submit success!");
    return result;
}

/*
JNIEXPORT jstring JNICALL Java_com_mnn_chatglm_Chat_Response(JNIEnv* env, jobject thiz) {
    jstring result = env->NewStringUTF(response_buffer.str().c_str());
    return result;
}
*/

JNIEXPORT jbyteArray JNICALL Java_com_mnn_llm_Chat_Response(JNIEnv* env, jobject thiz) {
    auto len = response_buffer.str().size();
    jbyteArray res = env->NewByteArray(len);
    env->SetByteArrayRegion(res, 0, len, (const jbyte*)response_buffer.str().c_str());
    return res;
}

JNIEXPORT void JNICALL Java_com_mnn_llm_Chat_Done(JNIEnv* env, jobject thiz) {
    response_buffer.str("");
}

JNIEXPORT void JNICALL Java_com_mnn_llm_Chat_Reset(JNIEnv* env, jobject thiz) {
    llm->reset();
}

} // extern "C"