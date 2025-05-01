#include <opencv2/opencv.hpp>
#include <string>


cv::Mat applySmoothing(const cv::Mat& input, const std::string& method, int param1, int param2) { //param1 is window size, param2 is extra parameter like sigma
	int radius = param1 / 2; 
	cv::Mat padded;
	cv::copyMakeBorder(input, padded, radius, radius, radius, radius, cv::BORDER_REFLECT_101); //edge mirror padding

	cv::Mat output = input.clone(); // same size as original image

	// iterate each pixel
	for (int y = 0; y < input.rows; ++y) {
		for (int x = 0; x < input.cols; ++x) {
			std::vector<uchar> nb_r, nb_g, nb_b; // neighbor pixel values
			double sum_r = 0, sum_g = 0, sum_b = 0;
			double weight_sum = 0;

			// sliding window to extract neighbor from padded
			for (int dy = -radius; dy <= radius; ++dy) {
				for (int dx = -radius; dx <= radius; ++dx) {
					int py = y + dy + radius;
					int px = x + dx + radius;
					cv::Vec3b pixel = padded.at<cv::Vec3b>(py, px);

                    if (method == "mean") {
                        sum_b += pixel[0]; sum_g += pixel[1]; sum_r += pixel[2];
                    }
                    else if (method == "gaussian") {
                        double dist = dx * dx + dy * dy;
                        double sigma = param2 > 0 ? param2 : 1.0;
                        double weight = std::exp(-dist / (2 * sigma * sigma));
                        sum_b += weight * pixel[0];
                        sum_g += weight * pixel[1];
                        sum_r += weight * pixel[2];
                        weight_sum += weight;
                    }
                    else if (method == "median") {
                        nb_b.push_back(pixel[0]);
                        nb_g.push_back(pixel[1]);
                        nb_r.push_back(pixel[2]);
                    }
                    else if (method == "bilateral") {
                        cv::Vec3b center = padded.at<cv::Vec3b>(y + radius, x + radius);
                        double color_diff = std::pow(pixel[0] - center[0], 2) +
                            std::pow(pixel[1] - center[1], 2) +
                            std::pow(pixel[2] - center[2], 2);
                        double spatial = dx * dx + dy * dy;
                        double sigmaColor = param2 > 0 ? param2 : 25.0;
                        double sigmaSpace = param2 > 0 ? param2 : 25.0;
                        double w = std::exp(-color_diff / (2 * sigmaColor * sigmaColor)) *
                            std::exp(-spatial / (2 * sigmaSpace * sigmaSpace));
                        sum_b += w * pixel[0];
                        sum_g += w * pixel[1];
                        sum_r += w * pixel[2];
                        weight_sum += w;
                    }
                }
            }

            if (method == "mean") {
                int area = param1 * param1;
                output.at<cv::Vec3b>(y, x) = cv::Vec3b(sum_b / area, sum_g / area, sum_r / area);
            }
            else if (method == "gaussian" || method == "bilateral") {
                output.at<cv::Vec3b>(y, x) = cv::Vec3b(sum_b / weight_sum, sum_g / weight_sum, sum_r / weight_sum);
            }
            else if (method == "median") {
                std::sort(nb_b.begin(), nb_b.end());
                std::sort(nb_g.begin(), nb_g.end());
                std::sort(nb_r.begin(), nb_r.end());
                int mid = nb_b.size() / 2;
                output.at<cv::Vec3b>(y, x) = cv::Vec3b(nb_b[mid], nb_g[mid], nb_r[mid]);
            }
        }
    }

    return output;
}
