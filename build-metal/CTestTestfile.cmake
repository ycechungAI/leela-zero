# CMake generated Testfile for 
# Source directory: /Users/neolinux/dev/leela-zero
# Build directory: /Users/neolinux/dev/leela-zero/build-metal
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test("unit_tests" "/Users/neolinux/dev/leela-zero/build-metal/tests")
set_tests_properties("unit_tests" PROPERTIES  WORKING_DIRECTORY "/Users/neolinux/dev/leela-zero/src" _BACKTRACE_TRIPLES "/Users/neolinux/dev/leela-zero/CMakeLists.txt;192;add_test;/Users/neolinux/dev/leela-zero/CMakeLists.txt;0;")
subdirs("gtest")
subdirs("autogtp")
subdirs("validation")
