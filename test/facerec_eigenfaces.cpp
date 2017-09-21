#include <iostream>
#include <fstream> 
#include <sstream> // String streams
#include <stdlib.h> /* atoi */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "FACE/BasicFaceRecognizer.hpp"

using namespace std;

static cv::Mat norm_0_255( cv::InputArray _src ){ 
    // create and return normalized image

    cv::Mat dst;
    cv::Mat src = _src.getMat();

    switch( src.channels() ){
    case 1:
        cv::normalize( _src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1 );
        break;
    case 3:
        cv::normalize( _src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC3 );
        break;
    default:       
        src.copyTo( dst );
        break;
    }

    return dst;
}

/*  CSV: 逗号分隔值（Comma-Separated Values，CSV，有时也称为字符分隔值，因为分隔字符也可以不是逗号）

    在程序中，我决定从一个非常简单的CSV文件中读取图像。 为什么？ 因为它是最简单的平台无关的方法，我可以想到。 
    基本上所有的CSV文件都需要包含一个文件名，后跟一个; 后面是标签label（作为整数），组成一行如下：

            /path/to/image.ext;0
*/
static void read_csv( const std::string& filename, std::vector<cv::Mat>& images, std::vector<int>& labels, char separator = ';' ) {
    std::ifstream file_in( filename.c_str(), std::ifstream::in ); // ifstream constructor

    if ( !file_in ) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error( cv::Error::StsBadArg, error_message );
    }
    string line, path, classlabel;

    while ( getline( file_in, line ) ) {   // istream& getline (istream&& is, string& str);
        stringstream liness( line ); // Stream class to operate on strings.

        getline(liness, path, separator);   // istream& getline (istream&& is, string& str, char delim)
        getline(liness, classlabel);        // istream& getline (istream&& is, string& str);    until: newline character,'\n'

        if( !path.empty() && !classlabel.empty() ) {
            images.push_back( cv::imread( path, 0 ) ); // read gray image
            labels.push_back( atoi( classlabel.c_str() ) ); // atoi: Convert string to integer
        }
    }
}

int main( int argc, const char** argv ){
    // check for valid command line arguments, print usage if no arguments were given
    if( argc < 2 ){
        cout << "usage: " << argv[0] << " <csv.ext> <output_folder> " << endl;
        exit( 1 );
    }
    string output_folder = ".";
    if( argc == 3 )
        output_folder = string( argv[2] );

    string fn_csv = string( argv[1] ); // get the path to your CSV

    std::vector<cv::Mat> images; // These vectors hold the images and corresponding labels.
    std::vector<int> labels;

    // Read in the data. This can fail if no valid input filename is given.
    try{
        read_csv( fn_csv, images, labels );
    }
    catch( cv::Exception& e ){
        cerr << "Error opening file \"" << fn_csv << "\".  \nReason: " << e.msg << endl;
        // nothing more we can do
        exit( 1 ); 
    }

    // quit if there are not enough images for this demo
    if( images.size() <= 1 ){
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error( cv::Error::StsError, error_message );
    }

    // Get the height from the first image. We'll need this later in code to reshape the images to their original size
    int height = images[0].rows;

    cv::Mat testSample = images[ images.size() - 1 ]; //最后一张图片作为测试图片
    int testLabel = labels[ labels.size() - 1 ];
    images.pop_back(); // remove it from the vector, because the last image is chosen as test image
    labels.pop_back();

    cv::Ptr<FACE::BasicFaceRecognizer> model = FACE::createEigenFaceRecognizer(); // will use full PCA
    // cv::createEigenFaceRecognizer( 10 ); // 保留10个主成分
    // cv::createEigenFaceRecognizer( 10, 123.0 ); // create a FaceRecognizer with a confidence threshold (e.g. 123.0)

    model->train( images, labels ); // training an Eigenfaces model

    int predictLabel = model->predict( testSample ); // predicts the label of a given test image

    string result_message = cv::format( "Predicted class = %d, / Actual class = %d.", predictLabel, testLabel );
    cout << result_message << endl;

    cv::Mat eigenvalues = model->getEigenValues();
    cv::Mat V = model->getEigenVectors(); // each eigen vector is a column vector
    cv::Mat mean = model->getMean();

    // Display or save mean
    if( argc == 2 ){
        cv::imshow( "mean", norm_0_255( mean.reshape( 1, images[0].rows ) ) ); // must be gray images
    }
    else
        cv::imwrite( cv::format( "%s/mean.png", output_folder.c_str() ), norm_0_255( mean.reshape( 1, images[0].rows ) ) );

    // Display or save the Eigenfaces
    for( int i = 0; i < std::min( 10, V.cols ); i++ ){
        string msg = cv::format( "Eigenvalue #%d = %.5f", i, eigenvalues.at<double>( i ) );
        cout << msg << endl;
        // get eigenvector #i
        cv::Mat ev = V.col( i ).clone();
        cv::Mat grayscale = norm_0_255( ev.reshape( 1, height ) ); // Reshape to original size & normalize to [0...255] for imshow.
        cv::Mat cgrayscale; // show the image & apply a Jet colormap for better sensing
        cv::applyColorMap( grayscale, cgrayscale, cv::COLORMAP_JET );

        // Display or save
        if( argc == 2 )
            cv::imshow( cv::format( "eigenface_%d", i ), cgrayscale );
        else
            cv::imwrite( cv::format( "%s/eigenface_%d.png", output_folder.c_str(), i ), norm_0_255( cgrayscale ) );
    }

    // Display or save the image reconstruction at some predifined steps
    for( int num_components = std::min( V.cols, 10 ); num_components < std::min( V.cols, 300 ); num_components += 15 ){
        // slice the eigenvectors from the model
        cv::Mat evs = cv::Mat( V, cv::Range::all(), cv::Range( 0, num_components ) ); // cv::Mat (const Mat &m, const Range &rowRange, const Range &colRange )
        cv::Mat projection = cv::LDA::subspaceProject( evs, mean, images[0].reshape( 1, 1 ) );
        cv::Mat reconstruction = cv::LDA::subspaceReconstruct( evs, mean, projection );

        // Normalize the result
        reconstruction = norm_0_255( reconstruction.reshape( 1, images[0].rows ) );

        // Display or save
        if( argc == 2 )
            cv::imshow( cv::format( "eigenface_reconstruction_%d", num_components ), reconstruction );
        else
            cv::imwrite( cv::format( "%s/eigenface_reconstruction_%d.png", output_folder.c_str(), num_components ), reconstruction );        
    }

    // Display if we are not writing to an output folder
    if( argc == 2 ){
        cv::waitKey( 0 );
    }
    return 0;
}