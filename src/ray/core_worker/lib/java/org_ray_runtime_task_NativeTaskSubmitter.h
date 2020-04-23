// Copyright 2017 The Ray Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_ray_runtime_task_NativeTaskSubmitter */

#ifndef _Included_org_ray_runtime_task_NativeTaskSubmitter
#define _Included_org_ray_runtime_task_NativeTaskSubmitter
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     org_ray_runtime_task_NativeTaskSubmitter
 * Method:    nativeSubmitTask
 * Signature:
 * (JLorg/ray/runtime/functionmanager/FunctionDescriptor;Ljava/util/List;ILorg/ray/api/options/CallOptions;)Ljava/util/List;
 */
JNIEXPORT jobject JNICALL Java_org_ray_runtime_task_NativeTaskSubmitter_nativeSubmitTask(
    JNIEnv *, jclass, jlong, jobject, jobject, jint, jobject);

/*
 * Class:     org_ray_runtime_task_NativeTaskSubmitter
 * Method:    nativeCreateActor
 * Signature:
 * (JLorg/ray/runtime/functionmanager/FunctionDescriptor;Ljava/util/List;Lorg/ray/api/options/ActorCreationOptions;)[B
 */
JNIEXPORT jbyteArray JNICALL
Java_org_ray_runtime_task_NativeTaskSubmitter_nativeCreateActor(JNIEnv *, jclass, jlong,
                                                                jobject, jobject,
                                                                jobject);

/*
 * Class:     org_ray_runtime_task_NativeTaskSubmitter
 * Method:    nativeSubmitActorTask
 * Signature:
 * (J[BLorg/ray/runtime/functionmanager/FunctionDescriptor;Ljava/util/List;ILorg/ray/api/options/CallOptions;)Ljava/util/List;
 */
JNIEXPORT jobject JNICALL
Java_org_ray_runtime_task_NativeTaskSubmitter_nativeSubmitActorTask(JNIEnv *, jclass,
                                                                    jlong, jbyteArray,
                                                                    jobject, jobject,
                                                                    jint, jobject);

#ifdef __cplusplus
}
#endif
#endif