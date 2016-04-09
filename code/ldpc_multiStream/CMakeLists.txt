set(PROJECTNAME ldpcGpuFixed)
project(${PROJECTNAME})
cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

include_directories(
${CUDA_INCLUDE_DIRS}
${CMAKE_SOURCE_DIR}
)

link_directories(
"/usr/local/cuda/lib64/"
)

# set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-Wall")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall  -fopenmp -O3 -march=native -m64")

# Pass options to NVCC
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
    -arch=sm_50
    )

set(srcs
	main.cpp
	awgn_channel/CChanel.cpp
	ber_analyzer/CErrorAnalyzer.cpp
	decoder_template/CGPUDecoder.cpp
	terminal/CTerminal.cpp
	tools/debug_fx.cpp
	trame/CTrame.cpp
	matrix/constantes_decoder.cpp
	queue/thread.cpp
	queue/handler.cpp
)

set(hdrs
	awgn_channel/CChanel.h 
	awgn_channel/CChanel_AWGN_SIMD.h
	
	ber_analyzer/CErrorAnalyzer.h
	custom_api/custom_cuda.h
	
	decoder_2nms/CGPU_Decoder_2NMS_SIMD.h
	decoder_2nms/cuda/CUDA_2NMS_SIMD.h  
	decoder_2nms/cuda/simd_functions.h
	
	decoder_ms/CGPU_Decoder_MS_SIMD.h
	decoder_ms/cuda/CUDA_MS_SIMD.h  
	decoder_ms/cuda/simd_functions.h
	
	decoder_nms/CGPU_Decoder_NMS_SIMD.h
	decoder_nms/cuda/CUDA_NMS_SIMD.h  
	decoder_nms/cuda/simd_functions.h
	
	decoder_oms/CGPU_Decoder_OMS_SIMD.h
	decoder_oms/cuda/CUDA_OMS_SIMD.h  
	decoder_oms/cuda/simd_functions.h
	
	decoder_oms_v2/CGPU_Decoder_MS_SIMD_v2.h
	decoder_oms_v2/cuda/CUDA_OMS_SIMD_v2.h  
	decoder_oms_v2/cuda/simd_functions.h
	
	decoder_template/CGPUDecoder.h
	decoder_template/GPU_Scheduled_functions.h
	
	transpose/GPU_Transpose.h
	transpose/GPU_Transpose_uint8.h
	transpose/simd_functions.h
	
	terminal/CTerminal.h
	tools/debug_fx.h
	trame/CTrame.h
	
	queue/thread.h 
	queue/wqueue.h
	queue/handler.h
)

set(kernels
	awgn_channel/CChanel_AWGN_SIMD.cu
	custom_api/custom_cuda.cu
	
	decoder_2nms/CGPU_Decoder_2NMS_SIMD.cu 
	decoder_2nms/cuda/CUDA_2NMS_SIMD.cu
	
	decoder_ms/CGPU_Decoder_MS_SIMD.cu
	decoder_ms/cuda/CUDA_MS_SIMD.cu
	
	decoder_nms/CGPU_Decoder_NMS_SIMD.cu
	decoder_nms/cuda/CUDA_NMS_SIMD.cu
	
	decoder_oms/CGPU_Decoder_OMS_SIMD.cu
	decoder_oms/cuda/CUDA_OMS_SIMD.cu
	
	decoder_oms_v2/CGPU_Decoder_MS_SIMD_v2.cu
	decoder_oms_v2/cuda/CUDA_OMS_SIMD_v2.cu
	
	decoder_template/GPU_Scheduled_functions.cu
	
	transpose/GPU_Transpose.cu
	transpose/GPU_Transpose_uint8.cu
	
	timer/CTimer.cu
)
cuda_add_executable(${PROJECTNAME} ${srcs} ${hdrs} ${kernels})

target_link_libraries(${PROJECTNAME} -lcurand -lpthread)

install(TARGETS ${PROJECTNAME} RUNTIME DESTINATION bin)
