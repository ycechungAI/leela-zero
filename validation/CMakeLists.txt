add_executable(validation
        main.cpp ../autogtp/Game.cpp SPRT.cpp Validation.cpp Results.cpp
        ../autogtp/Game.h SPRT.h Validation.h Results.h ../autogtp/Console.h)
set_target_properties(validation PROPERTIES AUTOMOC ON)
target_link_libraries(validation PRIVATE Qt6::Core)

install(TARGETS validation DESTINATION ${CMAKE_INSTALL_BINDIR})
