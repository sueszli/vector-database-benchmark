/**
 * @file edge.cpp
 * @brief Simple test for OpenCV
 * @author Denis Deryugin <deryugin.denis@gmail.com>
 * @version
 * @date 03.06.2019
 */

#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <stdio.h>
#include <time.h>

#include <drivers/video/fb.h>

using namespace cv;
using namespace std;

static void help(void) {
	printf("\nThis sample demonstrates Canny edge detection\n"
			"Call:\n"
			"	./edge [image [threshold]] \n\n");
}

static const char* keys = {
	"{help h||}"
	"{@image        | \"fruits.png\" | source image   }"
	"{@repeat |1     | number}"
};

static void imdrawfb(Mat& img) {
	struct fb_info *fbi;
	int w, h;

	fbi = fb_lookup(0);
	if (!fbi) {
		printf("fb0 not found\n");
		return;
	}

	printf("Framebuffer: %dx%d %dbpp\n", fbi->var.xres, fbi->var.yres, fbi->var.bits_per_pixel);

	h = min((int) fbi->var.yres, img.rows);
	w = min((int) (fbi->var.bits_per_pixel * fbi->var.xres) / 8, 3 * img.cols);

	for (int y = 0; y < h; y++) {
		const uchar *row = &img.at<uchar>(y, 0);
		for (int x = 0; x < w; x += 3) {
			unsigned rgb888	=
				0xFF000000 |
				unsigned(row[x]) |
				(unsigned(row[x + 1]) << 8) |
				(unsigned(row[x + 2]) << 16);

			((uint32_t *) fbi->screen_base)[fbi->var.xres * y + x / 3] = rgb888;
		}
	}
}

int main(int argc, const char** argv) {
	struct timeval tv_start, tv_end, tv_res;
	int edgeThresh = 2;
	Mat image, gray, edge, cedge;

	CommandLineParser parser(argc, argv, keys);
	if (parser.has("help"))  {
		help();
		return 0;
	}

	edgeThresh = parser.get<int>(1);
	string filename = parser.get<String>("@image");
	image = imread(filename, 1);
	if(image.empty()) {
		printf("Cannot read image file: %s\n", filename.c_str());
		help();
		return -1;
	}

	cedge.create(image.size(), image.type());
	cvtColor(image, gray, COLOR_BGR2GRAY);

	gettimeofday(&tv_start, NULL);
	{
		blur(gray, edge, Size(3,3));
		Canny(edge, edge, edgeThresh, edgeThresh*3, 3);
		cedge = Scalar::all(0);
	}
	gettimeofday(&tv_end, NULL);
	timersub(&tv_end, &tv_start, &tv_res);

	image.copyTo(cedge, edge);

	printf("Image: %dx%d; Threshold=%d\n", cedge.cols, cedge.rows, edgeThresh);
	printf("Detection time: %d s %d ms\n", (int) tv_res.tv_sec, (int) tv_res.tv_usec / 1000);

	imdrawfb(cedge);

	return 0;
}
