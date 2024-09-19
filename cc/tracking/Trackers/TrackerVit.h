#include "../Tracker.h"

#if CV_VERSION_GREATER_EQUAL(3, 2, 0)

#ifndef __FF_TRACKERVIT_H__
#define __FF_TRACKERVIT_H__

// Ensure that TrackerVit is only defined for OpenCV 4.7.0 or greater
#if CV_VERSION_GREATER_EQUAL(4, 9, 0)
class TrackerVit : public FF::ObjectWrapBase<TrackerVit>, public Nan::ObjectWrap {
public:
  cv::Ptr<cv::TrackerVit> tracker;

  static NAN_MODULE_INIT(Init);
  static NAN_METHOD(New);
  static NAN_METHOD(Init);
  static NAN_METHOD(Update);

  static Nan::Persistent<v8::FunctionTemplate> constructor;

  cv::Ptr<cv::Tracker> getTracker() {
    return tracker;
  }
};
#endif

#endif

#endif
