add_library( FACE SHARED
        Covar_Eigen.cpp
        EigenFaces.cpp
        FaceRecognizer.cpp
        FisherFaces.cpp
        LBPHFaces.cpp
        PCA.cpp
        elbp.cpp
        predict_collector.cpp
         )

target_link_libraries( FACE ${THIRD_PARTY_LIBS} )