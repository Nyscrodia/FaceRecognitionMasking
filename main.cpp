#include<iostream>
#include<string>
//dlib 라이브러리를 사용하기 위함
#include<dlib/opencv.h>
#include<dlib/image_processing.h>
#include<dlib/image_processing/frontal_face_detector.h>
//opencv라이브러리를 사용하기 위함
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
using namespace dlib;

void imgover(Mat& src, const Mat& img, const Point& pt, full_object_detection landmark) {

	int sx = std::max(pt.x, 0);
	int sy = std::max(pt.y, 0);
	int ex = std::min(pt.x + img.cols, src.cols);
	int ey = std::max(pt.y + img.rows, int(landmark.part(8).x() - 50));

	for (int y = sy; y < ey; y++) {
		int y2 = y - pt.y;

		Vec3b* psrc = src.ptr<Vec3b>(y);
		const Vec4b* povr = img.ptr<Vec4b>(y2);

		for (int x = sx; x < ex; x++) {
			int x2 = x - pt.x;

			float alp = (float)povr[x2][3] / 255.f;

			if (alp > 0.f) {
				psrc[x][0] = saturate_cast<uchar>(psrc[x][0] * (1.f - alp) + povr[x2][0] * alp);
				psrc[x][1] = saturate_cast<uchar>(psrc[x][1] * (1.f - alp) + povr[x2][1] * alp);
				psrc[x][2] = saturate_cast<uchar>(psrc[x][2] * (1.f - alp) + povr[x2][2] * alp);
			}
		}
	}
}

int main() {
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor landmarkDetector;
	deserialize("shape_predictor_68_face_landmarks.dat") >> landmarkDetector;

	VideoCapture cap(0);
	Mat ROI, mask = imread("glasses.png", IMREAD_UNCHANGED);

	if (!cap.isOpened() || mask.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	GaussianBlur(mask, mask, Size(), 1);

	while (1) {
		Mat src;
		cap >> src;

		cv_image<bgr_pixel> dlib_src(src);
		std::vector<dlib::rectangle> faceRects = detector(dlib_src);
		int iFaceCount = faceRects.size();

		//draw
		for (int i = 0; i < iFaceCount; i++) {
			full_object_detection faceLandmark = landmarkDetector(dlib_src, faceRects[i]);
			float scale = float((faceLandmark.part(16).x()) - (faceLandmark.part(0).x())) / mask.cols;

			Mat img;
			resize(mask, img, Size(), scale, scale);
			Point pt(faceLandmark.part(0).x(), faceLandmark.part(0).y() - 30);
			imgover(src, img, pt, faceLandmark);
		}

		imshow("cap img", src);
		if (waitKey(10) == 27) break;
	}

	return 0;
}