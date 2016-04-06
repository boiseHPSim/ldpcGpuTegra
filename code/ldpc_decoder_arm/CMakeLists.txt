set(PROJECTNAME "ldpcARM")
project(${PROJECTNAME})
cmake_minimum_required(VERSION 2.8)
# set(CMAKE_VERBOSE_MAKEFILE OFF)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -fopenmp -O3 -march=native -std=c++11" )

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
)

set(srcs
 CBitGenerator/CBitGenerator.cpp 
 CChanel/CChanel.cpp 
 CChanel/CChanelAWGN_x86.cpp
 CChanel/CChanelForFixed.cpp
 CDecoder/template/CDecoder.cpp 
 CDecoder/template/CDecoder_fixed.cpp 
 CDecoder/template/CDecoder_fixed_x86.cpp 
#  CDecoder/template/CDecoder_fixed_SSE.cpp 
 CDecoder/OMS/CDecoder_OMS_fixed_x86.cpp
#  CDecoder/OMS/CDecoder_OMS_fixed_SSE.cpp
#  CDecoder/OMS/CDecoder_OMS_fixed_NEON16_v2.cpp
 CEncoder/CFakeEncoder.cpp 
 CEncoder/Encoder.cpp 
 CEncoder/GenericEncoder.cpp 
 CErrorAnalyzer/CErrorAnalyzer.cpp 
 CFixPointConversion/CFastFixConversion.cpp 
 CFixPointConversion/CFixConversion.cpp 
 CTerminal/CTerminal.cpp 
 CTimer/CTimer.cpp 
 CTrame/CTrame.cpp 
#  CTools/transpose_neon.cpp 
 main.cpp
)

set(hdrs
	
	)
	
add_executable(${PROJECTNAME} ${srcs} ${hdrs})

target_link_libraries(${PROJECTNAME}  
-lboost_system  -lboost_timer
-lpthread -lm -ldl)

install(TARGETS ${PROJECTNAME} RUNTIME DESTINATION bin) 
