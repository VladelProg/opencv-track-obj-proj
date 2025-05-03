// OpenCV Object Tracking with YOLOv4-tiny and KCF Tracker
//#define USE_IMAGE_SEQUENCE // обработка  
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <direct.h>
#include <omp.h>
#include <immintrin.h> 

#define CONF_THRESHOLD 0.5
#define NMS_THRESHOLD 0.4
#define FRAME_WIDTH 320
#define FRAME_HEIGHT 320
#define MAX_TRAIL 10

using namespace cv;
using namespace dnn;
using namespace std;

#ifdef USE_IMAGE_SEQUENCE
const string inputFolder = "images/";
const int numFrames = 40;
#else
const string videoPath = "video5.mp4";
#endif
TickMeter timer;
const string modelWeights = "yolov4-tiny.weights";
const string modelConfig = "yolov4-tiny.cfg";
const string classNamesFile = "coco.names";
const string outputFolder = "output_images/";

vector<string> classNames;
Net net;
bool initialized = false;
deque<Point> trajectory;
vector<Ptr<Tracker>> trackers;
vector<Rect> bboxes;
vector<string> labels;
vector<vector<Point>> paths;

bool objectSelected = false;
Rect selectedBox;
void loadClasses(const string& path) {
    ifstream ifs(path);
    string line;
    while (getline(ifs, line)) classNames.push_back(line);
}

vector<string> getOutputsNames(const Net& net) {
    static vector<string> names;
    if (names.empty()) {
        vector<int> outLayers = net.getUnconnectedOutLayers();
        vector<string> layersNames = net.getLayerNames();
        for (int i = 0; i < outLayers.size(); ++i)
            names.push_back(layersNames[outLayers[i] - 1]);
    }
    return names;
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom,
    Mat& frame, const vector<string>& classNames) {
    // Нарисовать прямоугольник
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 2);

    // Сформировать строку с меткой и вероятностью
    string label = format("%.2f", conf);
    if (!classNames.empty() && classId >= 0 && classId < classNames.size()) {
        label = classNames[classId] + ": " + label;
    }

    // Отрисовать фон и текст
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
        Point(left + labelSize.width, top + baseLine),
        Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(), 1);
}


void detectObjects(Mat& frame, vector<Rect>& boxes, vector<int>& classIds, vector<float>& confidences) {
    Mat blob;
    blobFromImage(frame, blob, 1 / 255.0, Size(416, 416), Scalar(), true, false);
    net.setInput(blob);

    vector<Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    float confThreshold = 0.5;
    float nmsThreshold = 0.4;

    //NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, outs);


    vector<int> tempClassIds;
    vector<float> tempConfidences;
    vector<Rect> tempBoxes;

    for (size_t i = 0; i < outs.size(); ++i) {
        float* data = (float*)outs[i].data;
        int rows = outs[i].rows;
        int cols = outs[i].cols;

#pragma omp parallel for
        for (int j = 0; j < rows; ++j) {
            Mat scores = outs[i].row(j).colRange(5, cols);
            Point classIdPoint;
            double confidence;
            std:: cout << "Прив";
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold) {
                int centerX = (int)(data[j * cols + 0] * frame.cols);
                int centerY = (int)(data[j * cols + 1] * frame.rows);
                int width = (int)(data[j * cols + 2] * frame.cols);
                int height = (int)(data[j * cols + 3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                std::cout << "Тут";

                Rect box(left, top, width, height);

#pragma omp critical
                {
                    tempBoxes.push_back(box);
                    tempClassIds.push_back(classIdPoint.x);
                    tempConfidences.push_back((float)confidence);
                }
            }
        }
    }

    vector<int> indices;
    dnn::NMSBoxes(tempBoxes, tempConfidences, confThreshold, nmsThreshold, indices);

    for (int idx : indices) {
        boxes.push_back(tempBoxes[idx]);
        classIds.push_back(tempClassIds[idx]);
        confidences.push_back(tempConfidences[idx]);
    }
}




int main() {
    setlocale(LC_ALL, "Russian");
    int frameCount = 0; // глобальная или внешняя переменная
    _mkdir(outputFolder.c_str());

    loadClasses(classNamesFile);
    net = readNetFromDarknet(modelConfig, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
    //net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE); // Intel OpenVINO
    cout << cv::getNumberOfCPUs();
    cv::setNumThreads(cv::getNumberOfCPUs());
    //std::cout << "Оптимизации OpenCV: " << cv::getBuildInformation() << std::endl;
    //cv::dnn::Net net = cv::dnn::readNetFromDarknet();

#ifdef USE_IMAGE_SEQUENCE
    for (int i = 0; i < numFrames; ++i) {
        char filename[100];
        sprintf_s(filename, sizeof(filename), "images/frame_%06d.bmp", i);
        Mat frame = imread(filename);
        if (frame.empty()) {
            cerr << "Ошибка загрузки " << filename << endl;
            continue;
        }

        // ====== НОВЫЙ КОД: HSV + GaussianBlur ======
        /*
        Mat hsvFrame, blurredFrame;
        cvtColor(frame, hsvFrame, COLOR_BGR2HSV); // Конвертация в HSV
        GaussianBlur(hsvFrame, blurredFrame, Size(3, 3), 0); // Маленькое размытие
        Mat processedFrame;
        cvtColor(blurredFrame, processedFrame, COLOR_HSV2BGR); // Вернуться в BGR для YOLO
        */
        timer.reset();
        timer.start();
#else
    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "Не удалось открыть видео." << endl;
        return -1;
    }
    Mat frame;
    int frameIdx = 0;
    while (cap.read(frame)) {
        Mat blurredFrame;
        GaussianBlur(frame, blurredFrame, Size(3, 3), 0);
        frame = blurredFrame;
        timer.reset();
        timer.start();
#endif

        // ====== ВАШ КОД ИНИЦИАЛИЗАЦИИ И ТРЕКИНГА ======
        if (!initialized) {
            vector<Rect> boxes;
            vector<int> classIds;
            vector<float> confidences;
            detectObjects(frame, boxes, classIds, confidences);

            for (size_t j = 0; j < boxes.size(); ++j) {
                rectangle(frame, boxes[j], Scalar(255, 0, 0), 2);
                string label = classNames[classIds[j]];
                putText(frame, label, boxes[j].tl(), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
            }

            cout << "Выберите объекты (Enter), затем пробел для запуска." << endl;
            while (true) {
                imshow("Выбор объектов", frame);
                int key = waitKey(0);
                if (key == 32) break;

                Rect2d roi = selectROI("Выбор объектов", frame, false);
                // 64 106
                // 63 131
                if (roi.width > 0 && roi.height > 0) {
                    string label = "Unknown";
                    for (size_t k = 0; k < boxes.size(); ++k) {
                        if ((Rect(roi) & boxes[k]).area() > 0) {
                            label = classNames[classIds[k]];
                            Rect box = boxes[k];
                            std::cout << "Объект: " << label << endl;
                            std::cout << "Координаты: x = " << box.x << ", y = " << box.y
                                << ", width = " << box.width << ", height = " << box.height << endl;
                            break;
                        }
                    }
                    std::cout << "Параметры: " << roi.height << " " << roi.width;
                    //Tracker
                    auto tracker = TrackerCSRT::create();
                    tracker->init(frame, roi);
                    trackers.push_back(tracker);
                    bboxes.push_back(roi);
                    
                    labels.push_back(label);
                    paths.push_back(vector<Point>());
                }
            }
            destroyWindow("Выбор объектов");
            initialized = true;
        }
        else {
            
            
            // Внутри вашего основного цикла обработки кадров:
            frameCount++;
                // Ваша логика обработки трекеров
                struct TrackerResult {
                    bool ok;
                    Rect2d box;
                    string label;
                    vector<Point> path;
                    bool lost;
                };

                vector<TrackerResult> results(trackers.size());

#pragma omp parallel for
                for (int t = 0; t < trackers.size(); ++t) {
                    Rect box = bboxes[t];
                    bool ok = trackers[t]->update(frame, box);
                    results[t].ok = ok;
                    results[t].box = box;
                    results[t].label = labels[t];
                    results[t].lost = !ok;

                    if (ok) {
                        Point center(box.x + box.width / 2, box.y + box.height / 2);
#pragma omp critical
                        {
                            paths[t].push_back(center);
                            if (paths[t].size() > 50)
                                paths[t].erase(paths[t].begin(), paths[t].begin() + (paths[t].size() - 50));
                            results[t].path = paths[t];
                        }
                    }
                }

                for (size_t t = 0; t < results.size(); ++t) {
                    const Rect2d& box = results[t].box;
                    Point labelPos(box.x, box.y - 10);

                    if (results[t].ok) {
                        rectangle(frame, box, Scalar(255, 0, 0), 2);
                        putText(frame, results[t].label, labelPos, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);

                        const vector<Point>& path = results[t].path;
                        for (size_t j = 1; j < path.size(); ++j) {
                            line(frame, path[j - 1], path[j], Scalar(0, 255, 0), 2);
                        }
                    }
                    else {
                        putText(frame, "LOST", labelPos, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
                    }

                    bboxes[t] = results[t].box;
                }
            
        }

        timer.stop();
#ifdef USE_IMAGE_SEQUENCE
        char outFilename[100];
        cout << "Время обработки кадра " << i << ": " << timer.getTimeMilli() << " мс" << endl;
        sprintf_s(outFilename, sizeof(outFilename), "output_images/output_%06d.bmp", i);
        imwrite(outFilename, frame);
#else
        cout << "Время обработки кадра " << frameIdx++ << ": " << timer.getTimeMilli() << " мс" << endl;
        imshow("Object Tracking", frame);
        int key = waitKey(30);
        if (key == 27) break;
        else if (key == 't' || key == 'T') {
            cout << "Переключение трекера. Выберите новые объекты." << endl;
            trackers.clear();
            bboxes.clear();
            labels.clear();
            paths.clear();

            vector<Rect> boxes;
            vector<int> classIds;
            vector<float> confidences;
            detectObjects(frame, boxes, classIds, confidences);

            for (size_t j = 0; j < boxes.size(); ++j) {
                rectangle(frame, boxes[j], Scalar(255, 0, 0), 2);
                string label = classNames[classIds[j]];
                putText(frame, label, boxes[j].tl(), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
            }

            while (true) {
                imshow("Выбор объектов", frame);
                int k = waitKey(0);
                if (k == 32) break;

                Rect2d roi = selectROI("Выбор объектов", frame, false);
                if (roi.width > 0 && roi.height > 0) {
                    string label = "Unknown";
                    for (size_t k = 0; k < boxes.size(); ++k) {
                        if ((Rect(roi) & boxes[k]).area() > 0) {
                            label = classNames[classIds[k]];
                            break;
                        }
                    }
                    auto tracker = TrackerCSRT::create();
                    tracker->init(frame, roi);
                    trackers.push_back(tracker);
                    bboxes.push_back(roi);
                    labels.push_back(label);
                    paths.push_back(vector<Point>());
                }
            }
            destroyWindow("Выбор объектов");
        }
#endif
    }

    cout << "Готово. Результаты сохранены." << endl;
    return 0;
}