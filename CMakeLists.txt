cmake_minimum_required(VERSION 2.8)
project(FlPcnn)

if(UNIX)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++0x")
	add_definitions(-L /usr/local/ -l cnpy)
endif()

add_executable(	FlPcnn 
		npLoad.cpp
		coreFunc.cpp
		coreFunc.h
		utilities.cpp
		utilities.h
		parameters.h
)

target_link_libraries(FlPcnn cnpy opencv_core opencv_imgproc opencv_highgui opencv_objdetect pthread)
target_link_libraries (FlPcnn ${CMAKE_THREAD_LIBS_INIT}
)
