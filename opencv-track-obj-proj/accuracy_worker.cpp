#ifndef ACCURACY_WORKER_HPP
#define ACCURACY_WORKER_HPP

#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

class AccuracyWorker {
public:
    AccuracyWorker();
    ~AccuracyWorker();

    void start();
    void stop();

    void submit(const cv::Mat& frame, const cv::Mat& templ, const cv::Rect& roi, float* outAccuracy);

private:
    void workerLoop();

    struct Task {
        cv::Mat frame;
        cv::Mat templ;
        cv::Rect roi;
        float* outAccuracy;
    };

    std::thread workerThread;
    std::queue<Task> tasks;
    std::mutex mutex;
    std::condition_variable cv;
    bool running = false;
};

#endif
