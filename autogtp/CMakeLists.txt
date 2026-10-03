add_executable(autogtp
	Game.h Order.h Management.h Worker.h Job.h Result.h Console.h
	Worker.cpp Management.cpp Job.cpp main.cpp Game.cpp Order.cpp)
set_target_properties(autogtp PROPERTIES AUTOMOC ON)
target_link_libraries(autogtp PRIVATE Qt6::Core)

install(TARGETS autogtp DESTINATION ${CMAKE_INSTALL_BINDIR})
