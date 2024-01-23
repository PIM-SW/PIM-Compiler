get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
	${dialect_libs}
	${conversion_libs}
	MLIROptLib

	${PIMLIBS}
	)
add_executable(pim-compiler 
	pim-compiler.cpp
	)
add_dependencies(pim-compiler 
	PIMOpsInc
	PNMOpsInc
	APIMOpsInc
	DPIMOpsInc
	)
add_dependencies(pim-compiler 
	${PIMLIBS}
	)
target_include_directories(pim-compiler
	PRIVATE
	${PROJECT_SOURCE_DIR}
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}
	${PROJECT_BINARY_DIR}/include
	)
llvm_update_compile_flags(pim-compiler)
target_link_libraries(pim-compiler 
	PRIVATE 
	MLIRIR
	${LIBS} 
	)

mlir_check_all_link_libraries(pim-compiler)
