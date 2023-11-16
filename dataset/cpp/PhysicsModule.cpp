// Copyright(c) 2019 - 2020, #Momo
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met :
// 
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and /or other materials provided with the distribution.
// 
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "PhysicsModule.h"
#include "Utilities/Memory/Memory.h"
#include "Utilities/Profiler/Profiler.h"
#include "Platform/Bullet3/Bullet3Utils.h"

namespace MxEngine
{
    // defined in Core/Application/Physics.cpp
    void OnCollisionCallback();

    void PhysicsModule::Init()
    {
        data = Alloc<PhysicsModuleData>();
        data->CollisionConfiguration = Alloc<btDefaultCollisionConfiguration>();
        data->Dispatcher = Alloc<btCollisionDispatcher>(data->CollisionConfiguration);
        data->Broadphase = Alloc<btDbvtBroadphase>();
        data->Solver = Alloc<btSequentialImpulseConstraintSolver>();
        data->Solver->reset();
        data->World = Alloc<btDiscreteDynamicsWorld>(
            data->Dispatcher, data->Broadphase, data->Solver, data->CollisionConfiguration
        );

        data->World->setGravity(btVector3(0.0f, -9.8f, 0.0f));
    }

    void PhysicsModule::Destroy()
    {
        Free(data->World);
        Free(data->Solver);
        Free(data->Broadphase);
        Free(data->Dispatcher);
        Free(data->CollisionConfiguration);
        Free(data);
    }

    void PhysicsModule::OnUpdate(float dt)
    {
        if (data->simulationStep != 0.0f)
        {
            PhysicsModule::PerformSimulationStep(Min(dt, data->simulationStep));
            OnCollisionCallback();
        }
    }

    void PhysicsModule::PerformSimulationStep(float dt)
    {
        MAKE_SCOPE_PROFILER("Physics::SimulationStep()");
        constexpr int maxSubSteps = 10;
        data->World->stepSimulation(dt, maxSubSteps);
    }

    void PhysicsModule::SetSimulationStep(float timedelta)
    {
        data->simulationStep = timedelta;
    }

    float PhysicsModule::GetSimulationStep()
    {
        return data->simulationStep;
    }

    PhysicsModuleData* PhysicsModule::GetImpl()
    {
        return PhysicsModule::data;
    }

    void PhysicsModule::Clone(PhysicsModuleData* impl)
    {
        PhysicsModule::data = impl;
    }
}