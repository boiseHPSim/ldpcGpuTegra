set(PROJECTNAME "ldpcARM")
project(${PROJECTNAME})
cmake_minimum_required(VERSION 2.8)
# set(CMAKE_VERBOSE_MAKEFILE OFF)

find_package(CUDA QUIET REQUIRED)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -fopenmp -O3 -march=native -mtune=cortex-a15 -marm -mfpu=neon -finline -funroll-loops -flax-vector-conversions" )




include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
${CUDA_INCLUDE_DIRS}

)

link_directories(
"/usr/local/cuda/lib/"
)

set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
    -O2
    -arch=sm_32
)

set(srcs
 CBitGenerator/CBitGenerator.cpp 
 CChanel/CChanel.cpp 
 CChanel/CChanelAWGN_x86.cpp
 CChanel/CChanelForFixed.cpp
 CDecoder/template/CDecoder.cpp 
 CDecoder/template/CDecoder_fixed.cpp 
 CDecoder/template/CDecoder_fixed_x86.cpp 
 CDecoder/template/CDecoder_fixed_SSE.cpp 
 CDecoder/OMS/CDecoder_OMS_fixed_x86.cpp
 CDecoder/OMS/CDecoder_OMS_fixed_SSE.cpp 
 CDecoder/OMS/CDecoder_OMS_fixed_NEON16_v2.cpp
 CEncoder/CFakeEncoder.cpp 
 CEncoder/Encoder.cpp 
 CEncoder/GenericEncoder.cpp 
 CErrorAnalyzer/CErrorAnalyzer.cpp 
 CFixPointConversion/CFastFixConversion.cpp 
 CFixPointConversion/CFixConversion.cpp 
 CTerminal/CTerminal.cpp 
 CTimer/CTimer.cpp 
 CTrame/CTrame.cpp 
 CTools/transpose_neon.cpp
 cuda/matrix/constantes_decoder.cpp 
 cuda/tools/debug_fx.cpp
 cuda/decoder_template/CGPUDecoder.cpp
 cuda/matrix/constantes_decoder.cpp
 main.cpp
)

set(hdrs
cuda/test.h	
cuda/decoder_oms_v2/CGPU_Decoder_MS_SIMD_v2.h
cuda/decoder_oms_v2/cuda/CUDA_OMS_SIMD_v2.h  
cuda/decoder_oms_v2/cuda/simd_functions.h

cuda/decoder_template/CGPUDecoder.h
cuda/decoder_template/GPU_Scheduled_functions.h
cuda/custom_api/custom_cuda.h

cuda/transpose/GPU_Transpose.h
cuda/transpose/GPU_Transpose_uint8.h
cuda/transpose/simd_functions.h

cuda/tools/debug_fx.h
	)
	
set(kernels
cuda/test.cu

cuda/transpose/GPU_Transpose.cu
cuda/transpose/GPU_Transpose_uint8.cu

cuda/custom_api/custom_cuda.cu

cuda/decoder_oms_v2/CGPU_Decoder_MS_SIMD_v2.cu
cuda/decoder_oms_v2/cuda/CUDA_OMS_SIMD_v2.cu
cuda/decoder_template/GPU_Scheduled_functions.cu
)
	
cuda_add_executable(${PROJECTNAME} ${srcs} ${hdrs} ${kernels})
#add_executable(${PROJECTNAME} ${srcs} ${hdrs})

target_link_libraries(${PROJECTNAME}  -lpthread -lm -lcurand)

install(TARGETS ${PROJECTNAME} RUNTIME DESTINATION bin) 
