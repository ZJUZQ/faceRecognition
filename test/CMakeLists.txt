add_executable( facerec_eigenfaces facerec_eigenfaces.cpp )
target_link_libraries( facerec_eigenfaces FACE )

add_executable( facerec_fisherfaces facerec_fisherfaces.cpp )
target_link_libraries( facerec_fisherfaces FACE )

add_executable( facerec_lbph facerec_lbph.cpp )
target_link_libraries( facerec_lbph FACE )