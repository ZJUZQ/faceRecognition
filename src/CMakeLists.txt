add_library( FACE SHARED
        EigenFaces.cpp
        FaceRecognizer.cpp
        FisherFaces.cpp
        predict_collector.cpp
         )

target_link_libraries( FACE ${THIRD_PARTY_LIBS} )