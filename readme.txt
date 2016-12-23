link opencv with g++

g++ -o conv2d npLoad.cpp -L /usr/local/ -l cnpy -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect

g++ -o conv2d npLoad.cpp coreFunc.cpp -L /usr/local/ -l cnpy -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect

threading
g++ -o conv2d npLoad.cpp coreFunc.cpp -L /usr/local/ -l cnpy -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect -std=gnu++11 -pthread

utilities adding
g++ -o conv2d npLoad.cpp coreFunc.cpp utilities.cpp -L /usr/local/ -l cnpy -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect -std=gnu++11 -pthread


