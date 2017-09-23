#ifndef FACE_ELBP_HPP
#define FACE_ELBP_HPP

#include "FACE/common_includes.hpp"

namespace FACE{

// elbp: extended local binary patterns
cv::Mat elbp( cv::InputArray src, int radius, int neighgors );

cv::Mat histc( cv::InputArray _src, int minVal, int maxVal, bool normed );

cv::Mat spatial_histogram( cv::InputArray _src, int numPatterns, int grid_x, int grid_y, bool /*normed*/ );

}; /* namespace FACE */

#endif