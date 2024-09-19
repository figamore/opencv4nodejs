#include "opencv_modules.h"

#ifdef HAVE_OPENCV_TRACKING

#include "TrackerVit.h"

// Ensure that this code is only compiled if OpenCV is 4.7.0 or greater
#if CV_VERSION_GREATER_EQUAL(4, 9, 0)

Nan::Persistent<v8::FunctionTemplate> TrackerVit::constructor;

NAN_METHOD(TrackerVit::Init) {
  FF::TryCatch tryCatch("TrackerVit::Init");
  cv::Mat image;
  cv::Rect2d boundingBox;

  // Check if the arguments are correctly passed
  if (Mat::Converter::arg(0, &image, info) || Rect::Converter::arg(1, &boundingBox, info)) {
    return tryCatch.reThrow();
  }

  try {
    TrackerVit::unwrapThis(info)->getTracker()->init(image, boundingBox);
    
    // If no error is thrown, return true
    info.GetReturnValue().Set(Nan::True());
  } catch (const std::exception& e) {
    return tryCatch.throwError(e.what());
  }
}


NAN_METHOD(TrackerVit::Update) {
  FF::TryCatch tryCatch("TrackerVit::Update");
  cv::Mat image;
  if (Mat::Converter::arg(0, &image, info)) {
    return tryCatch.reThrow();
  }

  cv::Rect rect;
  bool ret = false;

  try {
    ret = TrackerVit::unwrapThis(info)->getTracker()->update(image, rect);
  } catch (std::exception& e) {
    return tryCatch.throwError(e.what());
  }

  if (ret) {
    info.GetReturnValue().Set(Rect::Converter::wrap(rect));
  } else {
    info.GetReturnValue().Set(Nan::Null());
  }
}

NAN_MODULE_INIT(TrackerVit::Init) {
  v8::Local<v8::FunctionTemplate> ctor = Nan::New<v8::FunctionTemplate>(TrackerVit::New);
  v8::Local<v8::ObjectTemplate> instanceTemplate = ctor->InstanceTemplate();

  Nan::SetPrototypeMethod(ctor, "init", TrackerVit::Init);
  Nan::SetPrototypeMethod(ctor, "update", TrackerVit::Update);

  constructor.Reset(ctor);
  ctor->SetClassName(FF::newString("TrackerVit"));
  instanceTemplate->SetInternalFieldCount(1);

  Nan::Set(target, FF::newString("TrackerVit"), FF::getFunction(ctor));
};

NAN_METHOD(TrackerVit::New) {
  FF::TryCatch tryCatch("TrackerVit::New");
  FF_ASSERT_CONSTRUCT_CALL();

  // Default model paths
  std::string netModelPath = "vittracknet.onnx";

  // Check if the user passed model paths as arguments
  if (info.Length() > 0) {
    if (FF::StringConverter::arg(0, &netModelPath, info)) {
      return tryCatch.reThrow();
    }
  }


  // Initialize TrackerVit with provided or default models
  TrackerVit* self = new TrackerVit();

  // Create tracker with provided ONNX models
  cv::TrackerVit::Params params;
  params.net = netModelPath;

  // Create the tracker instance with these parameters
  self->tracker = cv::TrackerVit::create(params);

  self->Wrap(info.Holder());
  info.GetReturnValue().Set(info.Holder());
}

#endif

#endif
