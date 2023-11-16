/*
 * Xsc.cpp
 * 
 * This file is part of the XShaderCompiler project (Copyright (c) 2014-2018 by Lukas Hermanns)
 * See "LICENSE.txt" for license information.
 */

#include <Xsc/Xsc.h>
#include "Compiler.h"
#include "ReportIdents.h"
#include <algorithm>

#ifdef XSC_ENABLE_SPIRV
#   include "SPIRVDisassembler.h"
#endif


namespace Xsc
{


XSC_EXPORT bool CompileShader(
    const ShaderInput&          inputDesc,
    const ShaderOutput&         outputDesc,
    Log*                        log,
    Reflection::ReflectionData* reflectionData)
{
    /* Compile shader with compiler driver */
    Compiler::StageTimePoints timePoints;

    Compiler compiler(log);

    auto result = compiler.CompileShader(
        inputDesc,
        outputDesc,
        reflectionData,
        &timePoints
    );

    /* Show timings */
    if (outputDesc.options.showTimes && log)
    {
        using TimePoint = Compiler::TimePoint;

        auto PrintTiming = [log](const std::string& processName, const TimePoint startTime, const TimePoint endTime)
        {
            long long duration = 0ll;

            if (endTime > startTime)
            {
                duration =
                (
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::duration<float>(endTime - startTime)
                    ).count()
                );
            }

            log->SubmitReport(
                Report(
                    ReportTypes::Info,
                    "timing " + processName + std::to_string(duration) + " ms"
                )
            );
        };

        PrintTiming( "pre-processing:   ", timePoints.preprocessor, timePoints.parser     );
        PrintTiming( "parsing:          ", timePoints.parser,       timePoints.analyzer   );
        PrintTiming( "context analysis: ", timePoints.analyzer,     timePoints.optimizer  );
        PrintTiming( "optimization:     ", timePoints.optimizer,    timePoints.generation );
        PrintTiming( "code generation:  ", timePoints.generation,   timePoints.reflection );
    }

    return result;
}

XSC_EXPORT void DisassembleShader(
    std::istream&               streamIn,
    std::ostream&               streamOut,
    const AssemblyDescriptor&   desc)
{
    switch (desc.intermediateLanguage)
    {
        case IntermediateLanguage::SPIRV:
        {
            #ifdef XSC_ENABLE_SPIRV

            /* Disassemble SPIR-V module */
            SPIRVDisassembler disassembler;
            disassembler.Parse(streamIn);
            disassembler.Print(streamOut, desc);

            #else

            throw std::invalid_argument(R_NotBuildWithSPIRV);

            #endif
        }
        break;

        default:
        {
            throw std::invalid_argument(R_InvalidILForDisassembling);
        }
        break;
    }
}


} // /namespace Xsc



// ================================================================================
