# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/vortex/zhou_temp_test/faceRecognition

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/vortex/zhou_temp_test/faceRecognition/build

# Include any dependencies generated for this target.
include src/CMakeFiles/FACE.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/FACE.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/FACE.dir/flags.make

src/CMakeFiles/FACE.dir/Covar_Eigen.cpp.o: src/CMakeFiles/FACE.dir/flags.make
src/CMakeFiles/FACE.dir/Covar_Eigen.cpp.o: ../src/Covar_Eigen.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/vortex/zhou_temp_test/faceRecognition/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/FACE.dir/Covar_Eigen.cpp.o"
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FACE.dir/Covar_Eigen.cpp.o -c /home/vortex/zhou_temp_test/faceRecognition/src/Covar_Eigen.cpp

src/CMakeFiles/FACE.dir/Covar_Eigen.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FACE.dir/Covar_Eigen.cpp.i"
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vortex/zhou_temp_test/faceRecognition/src/Covar_Eigen.cpp > CMakeFiles/FACE.dir/Covar_Eigen.cpp.i

src/CMakeFiles/FACE.dir/Covar_Eigen.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FACE.dir/Covar_Eigen.cpp.s"
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vortex/zhou_temp_test/faceRecognition/src/Covar_Eigen.cpp -o CMakeFiles/FACE.dir/Covar_Eigen.cpp.s

src/CMakeFiles/FACE.dir/Covar_Eigen.cpp.o.requires:

.PHONY : src/CMakeFiles/FACE.dir/Covar_Eigen.cpp.o.requires

src/CMakeFiles/FACE.dir/Covar_Eigen.cpp.o.provides: src/CMakeFiles/FACE.dir/Covar_Eigen.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/FACE.dir/build.make src/CMakeFiles/FACE.dir/Covar_Eigen.cpp.o.provides.build
.PHONY : src/CMakeFiles/FACE.dir/Covar_Eigen.cpp.o.provides

src/CMakeFiles/FACE.dir/Covar_Eigen.cpp.o.provides.build: src/CMakeFiles/FACE.dir/Covar_Eigen.cpp.o


src/CMakeFiles/FACE.dir/EigenFaces.cpp.o: src/CMakeFiles/FACE.dir/flags.make
src/CMakeFiles/FACE.dir/EigenFaces.cpp.o: ../src/EigenFaces.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/vortex/zhou_temp_test/faceRecognition/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/FACE.dir/EigenFaces.cpp.o"
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FACE.dir/EigenFaces.cpp.o -c /home/vortex/zhou_temp_test/faceRecognition/src/EigenFaces.cpp

src/CMakeFiles/FACE.dir/EigenFaces.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FACE.dir/EigenFaces.cpp.i"
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vortex/zhou_temp_test/faceRecognition/src/EigenFaces.cpp > CMakeFiles/FACE.dir/EigenFaces.cpp.i

src/CMakeFiles/FACE.dir/EigenFaces.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FACE.dir/EigenFaces.cpp.s"
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vortex/zhou_temp_test/faceRecognition/src/EigenFaces.cpp -o CMakeFiles/FACE.dir/EigenFaces.cpp.s

src/CMakeFiles/FACE.dir/EigenFaces.cpp.o.requires:

.PHONY : src/CMakeFiles/FACE.dir/EigenFaces.cpp.o.requires

src/CMakeFiles/FACE.dir/EigenFaces.cpp.o.provides: src/CMakeFiles/FACE.dir/EigenFaces.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/FACE.dir/build.make src/CMakeFiles/FACE.dir/EigenFaces.cpp.o.provides.build
.PHONY : src/CMakeFiles/FACE.dir/EigenFaces.cpp.o.provides

src/CMakeFiles/FACE.dir/EigenFaces.cpp.o.provides.build: src/CMakeFiles/FACE.dir/EigenFaces.cpp.o


src/CMakeFiles/FACE.dir/FaceRecognizer.cpp.o: src/CMakeFiles/FACE.dir/flags.make
src/CMakeFiles/FACE.dir/FaceRecognizer.cpp.o: ../src/FaceRecognizer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/vortex/zhou_temp_test/faceRecognition/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/FACE.dir/FaceRecognizer.cpp.o"
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FACE.dir/FaceRecognizer.cpp.o -c /home/vortex/zhou_temp_test/faceRecognition/src/FaceRecognizer.cpp

src/CMakeFiles/FACE.dir/FaceRecognizer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FACE.dir/FaceRecognizer.cpp.i"
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vortex/zhou_temp_test/faceRecognition/src/FaceRecognizer.cpp > CMakeFiles/FACE.dir/FaceRecognizer.cpp.i

src/CMakeFiles/FACE.dir/FaceRecognizer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FACE.dir/FaceRecognizer.cpp.s"
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vortex/zhou_temp_test/faceRecognition/src/FaceRecognizer.cpp -o CMakeFiles/FACE.dir/FaceRecognizer.cpp.s

src/CMakeFiles/FACE.dir/FaceRecognizer.cpp.o.requires:

.PHONY : src/CMakeFiles/FACE.dir/FaceRecognizer.cpp.o.requires

src/CMakeFiles/FACE.dir/FaceRecognizer.cpp.o.provides: src/CMakeFiles/FACE.dir/FaceRecognizer.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/FACE.dir/build.make src/CMakeFiles/FACE.dir/FaceRecognizer.cpp.o.provides.build
.PHONY : src/CMakeFiles/FACE.dir/FaceRecognizer.cpp.o.provides

src/CMakeFiles/FACE.dir/FaceRecognizer.cpp.o.provides.build: src/CMakeFiles/FACE.dir/FaceRecognizer.cpp.o


src/CMakeFiles/FACE.dir/FisherFaces.cpp.o: src/CMakeFiles/FACE.dir/flags.make
src/CMakeFiles/FACE.dir/FisherFaces.cpp.o: ../src/FisherFaces.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/vortex/zhou_temp_test/faceRecognition/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/FACE.dir/FisherFaces.cpp.o"
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FACE.dir/FisherFaces.cpp.o -c /home/vortex/zhou_temp_test/faceRecognition/src/FisherFaces.cpp

src/CMakeFiles/FACE.dir/FisherFaces.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FACE.dir/FisherFaces.cpp.i"
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vortex/zhou_temp_test/faceRecognition/src/FisherFaces.cpp > CMakeFiles/FACE.dir/FisherFaces.cpp.i

src/CMakeFiles/FACE.dir/FisherFaces.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FACE.dir/FisherFaces.cpp.s"
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vortex/zhou_temp_test/faceRecognition/src/FisherFaces.cpp -o CMakeFiles/FACE.dir/FisherFaces.cpp.s

src/CMakeFiles/FACE.dir/FisherFaces.cpp.o.requires:

.PHONY : src/CMakeFiles/FACE.dir/FisherFaces.cpp.o.requires

src/CMakeFiles/FACE.dir/FisherFaces.cpp.o.provides: src/CMakeFiles/FACE.dir/FisherFaces.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/FACE.dir/build.make src/CMakeFiles/FACE.dir/FisherFaces.cpp.o.provides.build
.PHONY : src/CMakeFiles/FACE.dir/FisherFaces.cpp.o.provides

src/CMakeFiles/FACE.dir/FisherFaces.cpp.o.provides.build: src/CMakeFiles/FACE.dir/FisherFaces.cpp.o


src/CMakeFiles/FACE.dir/LBPHFaces.cpp.o: src/CMakeFiles/FACE.dir/flags.make
src/CMakeFiles/FACE.dir/LBPHFaces.cpp.o: ../src/LBPHFaces.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/vortex/zhou_temp_test/faceRecognition/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/CMakeFiles/FACE.dir/LBPHFaces.cpp.o"
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FACE.dir/LBPHFaces.cpp.o -c /home/vortex/zhou_temp_test/faceRecognition/src/LBPHFaces.cpp

src/CMakeFiles/FACE.dir/LBPHFaces.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FACE.dir/LBPHFaces.cpp.i"
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vortex/zhou_temp_test/faceRecognition/src/LBPHFaces.cpp > CMakeFiles/FACE.dir/LBPHFaces.cpp.i

src/CMakeFiles/FACE.dir/LBPHFaces.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FACE.dir/LBPHFaces.cpp.s"
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vortex/zhou_temp_test/faceRecognition/src/LBPHFaces.cpp -o CMakeFiles/FACE.dir/LBPHFaces.cpp.s

src/CMakeFiles/FACE.dir/LBPHFaces.cpp.o.requires:

.PHONY : src/CMakeFiles/FACE.dir/LBPHFaces.cpp.o.requires

src/CMakeFiles/FACE.dir/LBPHFaces.cpp.o.provides: src/CMakeFiles/FACE.dir/LBPHFaces.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/FACE.dir/build.make src/CMakeFiles/FACE.dir/LBPHFaces.cpp.o.provides.build
.PHONY : src/CMakeFiles/FACE.dir/LBPHFaces.cpp.o.provides

src/CMakeFiles/FACE.dir/LBPHFaces.cpp.o.provides.build: src/CMakeFiles/FACE.dir/LBPHFaces.cpp.o


src/CMakeFiles/FACE.dir/PCA.cpp.o: src/CMakeFiles/FACE.dir/flags.make
src/CMakeFiles/FACE.dir/PCA.cpp.o: ../src/PCA.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/vortex/zhou_temp_test/faceRecognition/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/CMakeFiles/FACE.dir/PCA.cpp.o"
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FACE.dir/PCA.cpp.o -c /home/vortex/zhou_temp_test/faceRecognition/src/PCA.cpp

src/CMakeFiles/FACE.dir/PCA.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FACE.dir/PCA.cpp.i"
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vortex/zhou_temp_test/faceRecognition/src/PCA.cpp > CMakeFiles/FACE.dir/PCA.cpp.i

src/CMakeFiles/FACE.dir/PCA.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FACE.dir/PCA.cpp.s"
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vortex/zhou_temp_test/faceRecognition/src/PCA.cpp -o CMakeFiles/FACE.dir/PCA.cpp.s

src/CMakeFiles/FACE.dir/PCA.cpp.o.requires:

.PHONY : src/CMakeFiles/FACE.dir/PCA.cpp.o.requires

src/CMakeFiles/FACE.dir/PCA.cpp.o.provides: src/CMakeFiles/FACE.dir/PCA.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/FACE.dir/build.make src/CMakeFiles/FACE.dir/PCA.cpp.o.provides.build
.PHONY : src/CMakeFiles/FACE.dir/PCA.cpp.o.provides

src/CMakeFiles/FACE.dir/PCA.cpp.o.provides.build: src/CMakeFiles/FACE.dir/PCA.cpp.o


src/CMakeFiles/FACE.dir/elbp.cpp.o: src/CMakeFiles/FACE.dir/flags.make
src/CMakeFiles/FACE.dir/elbp.cpp.o: ../src/elbp.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/vortex/zhou_temp_test/faceRecognition/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object src/CMakeFiles/FACE.dir/elbp.cpp.o"
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FACE.dir/elbp.cpp.o -c /home/vortex/zhou_temp_test/faceRecognition/src/elbp.cpp

src/CMakeFiles/FACE.dir/elbp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FACE.dir/elbp.cpp.i"
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vortex/zhou_temp_test/faceRecognition/src/elbp.cpp > CMakeFiles/FACE.dir/elbp.cpp.i

src/CMakeFiles/FACE.dir/elbp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FACE.dir/elbp.cpp.s"
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vortex/zhou_temp_test/faceRecognition/src/elbp.cpp -o CMakeFiles/FACE.dir/elbp.cpp.s

src/CMakeFiles/FACE.dir/elbp.cpp.o.requires:

.PHONY : src/CMakeFiles/FACE.dir/elbp.cpp.o.requires

src/CMakeFiles/FACE.dir/elbp.cpp.o.provides: src/CMakeFiles/FACE.dir/elbp.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/FACE.dir/build.make src/CMakeFiles/FACE.dir/elbp.cpp.o.provides.build
.PHONY : src/CMakeFiles/FACE.dir/elbp.cpp.o.provides

src/CMakeFiles/FACE.dir/elbp.cpp.o.provides.build: src/CMakeFiles/FACE.dir/elbp.cpp.o


src/CMakeFiles/FACE.dir/predict_collector.cpp.o: src/CMakeFiles/FACE.dir/flags.make
src/CMakeFiles/FACE.dir/predict_collector.cpp.o: ../src/predict_collector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/vortex/zhou_temp_test/faceRecognition/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object src/CMakeFiles/FACE.dir/predict_collector.cpp.o"
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FACE.dir/predict_collector.cpp.o -c /home/vortex/zhou_temp_test/faceRecognition/src/predict_collector.cpp

src/CMakeFiles/FACE.dir/predict_collector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FACE.dir/predict_collector.cpp.i"
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vortex/zhou_temp_test/faceRecognition/src/predict_collector.cpp > CMakeFiles/FACE.dir/predict_collector.cpp.i

src/CMakeFiles/FACE.dir/predict_collector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FACE.dir/predict_collector.cpp.s"
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vortex/zhou_temp_test/faceRecognition/src/predict_collector.cpp -o CMakeFiles/FACE.dir/predict_collector.cpp.s

src/CMakeFiles/FACE.dir/predict_collector.cpp.o.requires:

.PHONY : src/CMakeFiles/FACE.dir/predict_collector.cpp.o.requires

src/CMakeFiles/FACE.dir/predict_collector.cpp.o.provides: src/CMakeFiles/FACE.dir/predict_collector.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/FACE.dir/build.make src/CMakeFiles/FACE.dir/predict_collector.cpp.o.provides.build
.PHONY : src/CMakeFiles/FACE.dir/predict_collector.cpp.o.provides

src/CMakeFiles/FACE.dir/predict_collector.cpp.o.provides.build: src/CMakeFiles/FACE.dir/predict_collector.cpp.o


# Object files for target FACE
FACE_OBJECTS = \
"CMakeFiles/FACE.dir/Covar_Eigen.cpp.o" \
"CMakeFiles/FACE.dir/EigenFaces.cpp.o" \
"CMakeFiles/FACE.dir/FaceRecognizer.cpp.o" \
"CMakeFiles/FACE.dir/FisherFaces.cpp.o" \
"CMakeFiles/FACE.dir/LBPHFaces.cpp.o" \
"CMakeFiles/FACE.dir/PCA.cpp.o" \
"CMakeFiles/FACE.dir/elbp.cpp.o" \
"CMakeFiles/FACE.dir/predict_collector.cpp.o"

# External object files for target FACE
FACE_EXTERNAL_OBJECTS =

../lib/libFACE.so: src/CMakeFiles/FACE.dir/Covar_Eigen.cpp.o
../lib/libFACE.so: src/CMakeFiles/FACE.dir/EigenFaces.cpp.o
../lib/libFACE.so: src/CMakeFiles/FACE.dir/FaceRecognizer.cpp.o
../lib/libFACE.so: src/CMakeFiles/FACE.dir/FisherFaces.cpp.o
../lib/libFACE.so: src/CMakeFiles/FACE.dir/LBPHFaces.cpp.o
../lib/libFACE.so: src/CMakeFiles/FACE.dir/PCA.cpp.o
../lib/libFACE.so: src/CMakeFiles/FACE.dir/elbp.cpp.o
../lib/libFACE.so: src/CMakeFiles/FACE.dir/predict_collector.cpp.o
../lib/libFACE.so: src/CMakeFiles/FACE.dir/build.make
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_stitching3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_superres3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_videostab3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_aruco3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_bgsegm3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_bioinspired3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_ccalib3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_cvv3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_datasets3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_dpm3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_face3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_fuzzy3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_hdf3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_line_descriptor3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_optflow3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_plot3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_reg3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_saliency3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_stereo3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_structured_light3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_surface_matching3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_text3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_xfeatures2d3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_ximgproc3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_xobjdetect3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_xphoto3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_shape3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_video3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_viz3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_phase_unwrapping3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_rgbd3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_calib3d3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_features2d3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_flann3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_objdetect3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_ml3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_highgui3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_photo3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_videoio3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_imgcodecs3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_imgproc3.so.3.2.0
../lib/libFACE.so: /opt/ros/kinetic/lib/libopencv_core3.so.3.2.0
../lib/libFACE.so: src/CMakeFiles/FACE.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/vortex/zhou_temp_test/faceRecognition/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX shared library ../../lib/libFACE.so"
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FACE.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/FACE.dir/build: ../lib/libFACE.so

.PHONY : src/CMakeFiles/FACE.dir/build

src/CMakeFiles/FACE.dir/requires: src/CMakeFiles/FACE.dir/Covar_Eigen.cpp.o.requires
src/CMakeFiles/FACE.dir/requires: src/CMakeFiles/FACE.dir/EigenFaces.cpp.o.requires
src/CMakeFiles/FACE.dir/requires: src/CMakeFiles/FACE.dir/FaceRecognizer.cpp.o.requires
src/CMakeFiles/FACE.dir/requires: src/CMakeFiles/FACE.dir/FisherFaces.cpp.o.requires
src/CMakeFiles/FACE.dir/requires: src/CMakeFiles/FACE.dir/LBPHFaces.cpp.o.requires
src/CMakeFiles/FACE.dir/requires: src/CMakeFiles/FACE.dir/PCA.cpp.o.requires
src/CMakeFiles/FACE.dir/requires: src/CMakeFiles/FACE.dir/elbp.cpp.o.requires
src/CMakeFiles/FACE.dir/requires: src/CMakeFiles/FACE.dir/predict_collector.cpp.o.requires

.PHONY : src/CMakeFiles/FACE.dir/requires

src/CMakeFiles/FACE.dir/clean:
	cd /home/vortex/zhou_temp_test/faceRecognition/build/src && $(CMAKE_COMMAND) -P CMakeFiles/FACE.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/FACE.dir/clean

src/CMakeFiles/FACE.dir/depend:
	cd /home/vortex/zhou_temp_test/faceRecognition/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vortex/zhou_temp_test/faceRecognition /home/vortex/zhou_temp_test/faceRecognition/src /home/vortex/zhou_temp_test/faceRecognition/build /home/vortex/zhou_temp_test/faceRecognition/build/src /home/vortex/zhou_temp_test/faceRecognition/build/src/CMakeFiles/FACE.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/FACE.dir/depend

