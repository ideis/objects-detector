#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


    using namespace cv;
    using namespace dnn;

    std::vector<std::string> classes;

    void postprocess(Mat& frame, const Mat& out, Net& net);

    void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

    void callback(int pos, void* userdata);
    float confThreshold = 0.2;

    int main(int argc, const char** argv)
    {

       // Open file with classes names
        std::string file = "/home/ideis/Code/C++/ObjectsDetector/models/ssd_mobilenet/classes.txt";
        std::ifstream ifs(file.c_str());
        if (!ifs.is_open())
            CV_Error(Error::StsError, "File " + file + " not found");
        std::string line;
        while (std::getline(ifs, line))
        {
            classes.push_back(line);
        }

        // Load a model
        String model = "/home/ideis/Code/C++/ObjectsDetector/models/ssd_mobilenet/weights.pb";
        String config = "/home/ideis/Code/C++/ObjectsDetector/models/ssd_mobilenet/graph.pbtxt";
        Net net = readNetFromTensorflow(model, config);

        // Create a window
        static const std::string kWinName = "Object Detection";
        namedWindow(kWinName, WINDOW_NORMAL);
        int initialConf = (int)(confThreshold * 100);
        createTrackbar("Confidence threshold, %", kWinName, &initialConf, 99, callback);

        // Open a camera stream
        VideoCapture cap;
        cap.open(0);

        // Process frames
        Mat frame, blob, img;
        while (waitKey(1) < 0)
        {
            cap >> frame;
            if (frame.empty())
            {
                waitKey();
                break;
            }
            resize(frame, img, Size(300,300));
            // Create a 4D blob from a frame
            blob = blobFromImage(frame, 1.0/127.5f, Size(300, 300), Scalar(127.5, 127.5, 127.5), true, false);

            // Run a model
            net.setInput(blob);
            Mat out = net.forward();

            postprocess(frame, out, net);

            imshow(kWinName, frame);
        }
        return 0;
    }

    void postprocess(Mat& frame, const Mat& out, Net& net)
    {
        static std::vector<int> outLayers = net.getUnconnectedOutLayers();
        static std::string outLayerType = net.getLayer(outLayers[0])->type;

        float* data = (float*)out.data;
        if (net.getLayer(0)->outputNameToIndex("im_info") != -1)
        {

            for (size_t i = 0; i < out.total(); i += 7)
            {
                float confidence = data[i + 2];
                if (confidence > confThreshold)
                {
                    int left = (int)data[i + 3];
                    int top = (int)data[i + 4];
                    int right = (int)data[i + 5];
                    int bottom = (int)data[i + 6];
                    int classId = (int)(data[i]);
                    drawPred(classId, confidence, left, top, right, bottom, frame);
                }
            }
        }
        else if (outLayerType == "DetectionOutput")
        {

            for (size_t i = 0; i < out.total(); i += 7)
            {
                float confidence = data[i + 2];
                if (confidence > confThreshold)
                {
                    int left = (int)(data[i + 3] * frame.cols);
                    int top = (int)(data[i + 4] * frame.rows);
                    int right = (int)(data[i + 5] * frame.cols);
                    int bottom = (int)(data[i + 6] * frame.rows);
                    int classId = (int)(data[i + 1]) - 1;
                    drawPred(classId, confidence, left, top, right, bottom, frame);
                }
            }
        }
        else if (outLayerType == "Region")
        {

            for (int i = 0; i < out.rows; ++i, data += out.cols)
            {
                Mat confidences = out.row(i).colRange(5, out.cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(confidences, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int classId = classIdPoint.x;
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;
                    drawPred(classId, (float)confidence, left, top, left + width, top + height, frame);
                }
            }
        }
        else
            CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);
    }

    void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame) {
        rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));

        std::string label = format("%.2f", conf);
        if (!classes.empty() && classId < (int)classes.size())
        {
            label = classes[classId] + ": " + label;

            int baseLine;
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            top = max(top, labelSize.height);
            rectangle(frame, Point(left, top - labelSize.height),
                      Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
            putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
        }
    }

    void callback(int pos, void*)
    {
        confThreshold = pos * 0.01f;
    }