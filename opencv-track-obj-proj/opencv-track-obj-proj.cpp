#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace dnn;
using namespace std;

const float confThreshold = 0.5;
const float nmsThreshold = 0.4;
const int inpWidth = 416;
const int inpHeight = 416;

Net net;
vector<string> classes;
deque<Point> trajectory;
const size_t maxTrail = 50; // Длина линии движения

Rect selectedBox;
string selectedLabel;
bool objectSelected = false;

void loadClasses(const string& path) {
    ifstream ifs(path);
    string line;
    while (getline(ifs, line)) classes.push_back(line);
}

vector<string> getOutputsNames(const Net& net) {
    static vector<string> names;
    if (names.empty()) {
        vector<int> outLayers = net.getUnconnectedOutLayers();
        vector<string> layersNames = net.getLayerNames();
        for (size_t i = 0; i < outLayers.size(); ++i)
            names.push_back(layersNames[outLayers[i] - 1]);
    }
    return names;
}

void postprocess(Mat& frame, const vector<Mat>& outs, vector<Rect>& boxes, vector<int>& classIds, vector<float>& confidences) {
    for (size_t i = 0; i < outs.size(); ++i) {
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold) {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
}

void mouseCallback(int event, int x, int y, int, void* userdata) {
    if (event != EVENT_LBUTTONDOWN || objectSelected) return;

    vector<Rect>* boxes = static_cast<vector<Rect>*>(userdata);
    for (size_t i = 0; i < boxes->size(); ++i) {
        if ((*boxes)[i].contains(Point(x, y))) {
            selectedBox = (*boxes)[i];
            objectSelected = true;
            break;
        }
    }
}

int main() {
    string cfg = "yolov4-tiny.cfg";
    string weights = "yolov4-tiny.weights";
    string namesFile = "coco.names";
    string videoPath = "video2.mp4";

    loadClasses(namesFile);
    net = readNetFromDarknet(cfg, weights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "Video error\n";
        return -1;
    }

    namedWindow("Object Tracking");
    setMouseCallback("Object Tracking", mouseCallback);

    Mat frame;
    while (cap.read(frame)) {
        Mat blob;
        blobFromImage(frame, blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(), true, false);
        net.setInput(blob);
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));

        vector<Rect> boxes;
        vector<int> classIds;
        vector<float> confidences;
        postprocess(frame, outs, boxes, classIds, confidences);

        vector<int> indices;
        NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

        if (!objectSelected) {
            for (int i : indices) {
                rectangle(frame, boxes[i], Scalar(255, 0, 0), 2);
                string label = format("%.2f", confidences[i]);
                if (!classes.empty()) {
                    label = classes[classIds[i]] + ": " + label;
                }
                putText(frame, label, boxes[i].tl(), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
            }
            // Ждём выбора объекта
            setMouseCallback("Object Tracking", mouseCallback, &boxes);
            
        }

        else {
            // Отслеживаем ранее выбранный объект
            bool matched = false;
            for (int i : indices) {
                Rect current = boxes[i];
                if ((current & selectedBox).area() > 0.5 * selectedBox.area()) {
                    selectedBox = current;
                    selectedLabel = classes[classIds[i]] + ": " + to_string((int)(confidences[i] * 100)) + "%";
                    matched = true;
                    break;
                }
            }

            // Сохраняем текущую точку центра
            Point center(selectedBox.x + selectedBox.width / 2, selectedBox.y + selectedBox.height / 2);
            trajectory.push_back(center);
            if (trajectory.size() > maxTrail)
                trajectory.pop_front();

            // Рисуем линию движения
            for (size_t i = 1; i < trajectory.size(); ++i) {
                line(frame, trajectory[i - 1], trajectory[i], Scalar(0, 255, 255), 2);
            }


            if (!matched) {
                selectedLabel = "Object lost...";
            }

            rectangle(frame, selectedBox, Scalar(0, 255, 0), 2);
            putText(frame, selectedLabel, selectedBox.tl(), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        }

        imshow("Object Tracking", frame);
        if (waitKey(10) == 27) break;
    }

    return 0;
}
