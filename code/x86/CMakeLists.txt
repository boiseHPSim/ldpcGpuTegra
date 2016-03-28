set(PROJECTNAME "ldpcX86")
project(${PROJECTNAME})
cmake_minimum_required(VERSION 2.8)
# set(CMAKE_VERBOSE_MAKEFILE OFF)

# set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-Wall")
# set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -fopenmp -O3 -march=native -fast -ansi-alias 
# 			-finline -funroll-loops -no-prec-div -opt-prefetch -unroll-aggressive 
# 			-m64 -auto-ilp32  -msse -fma -mkl  -DMKL_ILP64 -I${MKLROOT}/include")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -fopenmp -O3 -march=native -DMKL_ILP64 -msse -m64" )

set(MKL_INCLUDE_DIRS "/opt/intel/mkl/include/")
set(MKL_LIBRARIES "/opt/intel/mkl/lib/intel64/")
include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
${MKL_INCLUDE_DIRS}
)

link_directories(${MKL_LIBRARIES})

set(srcs
 CBitGenerator/CBitGenerator.cpp 
 CChanel/CChanel.cpp 
 CChanel/CChanelAWGN_MKL.cpp 
 CDecoder/template/CDecoder.cpp 
 CDecoder/template/CDecoder_fixed.cpp 
#  CDecoder/template/CDecoder_fixed_AVX.cpp 
 CDecoder/template/CDecoder_fixed_SSE.cpp 
 CDecoder/OMS/CDecoder_OMS_fixed_SSE.cpp 
#  CDecoder/OMS/CDecoder_OMS_fixed_AVX.cpp 
 CDecoder/NMS/CDecoder_NMS_fixed_SSE.cpp 
#  CDecoder/NMS/CDecoder_NMS_fixed_AVX.cpp 
 CEncoder/CFakeEncoder.cpp 
 CEncoder/Encoder.cpp 
 CEncoder/GenericEncoder.cpp 
 CErrorAnalyzer/CErrorAnalyzer.cpp 
 CFixPointConversion/CFastFixConversion.cpp 
 CFixPointConversion/CFixConversion.cpp 
 CTerminal/CTerminal.cpp 
 CTimer/CTimer.cpp 
 CTools/CTools.cpp 
#  CTools/transpose_avx.cpp 
 CTrame/CTrame.cpp 
 main_p.cpp
)

set(hdrs
	
	)
	
add_executable(${PROJECTNAME} ${srcs} ${hdrs})

target_link_libraries(${PROJECTNAME}  
-lboost_system  -lboost_timer
# -lPcmMsr  -Wl  --no-as-needed 
mkl_intel_lp64 mkl_sequential mkl_core 
-lpthread -lm -ldl)

install(TARGETS ${PROJECTNAME} RUNTIME DESTINATION bin) 
