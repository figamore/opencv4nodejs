#include "opencv_modules.h"

#ifdef HAVE_OPENCV_DNN

#include "macros.h"
#include "opencv2/core.hpp"

#include "Net.h"
#include "NetBindings.h"
#include "LayerConverter.h"


Nan::Persistent<v8::FunctionTemplate> Net::constructor;

NAN_MODULE_INIT(Net::Init) {
  v8::Local<v8::FunctionTemplate> ctor = Nan::New<v8::FunctionTemplate>(Net::New);
  Net::constructor.Reset(ctor);
  ctor->InstanceTemplate()->SetInternalFieldCount(1);
  ctor->SetClassName(Nan::New("Net").ToLocalChecked());

  // setInput(blob: Mat, name?: string, scalefactor?: number, mean?: number): void;
  // setInput(blob: Mat, inputName?: string): void;
  Nan::SetPrototypeMethod(ctor, "setInput", SetInput);
  Nan::SetPrototypeMethod(ctor, "setInputAsync", SetInputAsync);
  // forward(inputName?: string): Mat;
  Nan::SetPrototypeMethod(ctor, "forward", Forward);
  Nan::SetPrototypeMethod(ctor, "forwardAsync", ForwardAsync);
  // getLayerNames(): string[];
  
  Nan::SetPrototypeMethod(ctor, "getLayer", GetLayer);
  Nan::SetPrototypeMethod(ctor, "setLayerBlobs", SetLayerBlobs);
    Nan::SetPrototypeMethod(ctor, "getLayerId", GetLayerId);
  Nan::SetPrototypeMethod(ctor, "getLayerNames", GetLayerNames);
  Nan::SetPrototypeMethod(ctor, "getLayerNamesAsync", GetLayerNamesAsync);
  // getUnconnectedOutLayers(): number[];
  Nan::SetPrototypeMethod(ctor, "getUnconnectedOutLayers", GetUnconnectedOutLayers);
  Nan::SetPrototypeMethod(ctor, "getUnconnectedOutLayersAsync", GetUnconnectedOutLayersAsync);
  // dump(): string;
  Nan::SetPrototypeMethod(ctor, "dump", Dump);
  // setPreferableBackend(backendId: number): void;
  Nan::SetPrototypeMethod(ctor, "setPreferableBackend", SetPreferableBackend);
  // setPreferableTarget(targetId: number): void;
  Nan::SetPrototypeMethod(ctor, "setPreferableTarget", SetPreferableTarget);
  // getPerfProfile(): {	retval: number, timings: number[] };
  Nan::SetPrototypeMethod(ctor, "getPerfProfile", GetPerfProfile);

  Nan::Set(target, Nan::New("Net").ToLocalChecked(), FF::getFunction(ctor));
};

NAN_METHOD(Net::New) {
  FF::TryCatch tryCatch("Net::New");
  FF_ASSERT_CONSTRUCT_CALL();
  Net* self = new Net();
  self->Wrap(info.Holder());
  info.GetReturnValue().Set(info.Holder());
}

NAN_METHOD(Net::SetInput) {
  FF::executeSyncBinding(
      std::make_shared<NetBindings::SetInputWorker>(Net::unwrapSelf(info)),
      "Net::SetInput",
      info);
}

NAN_METHOD(Net::SetInputAsync) {
  FF::executeAsyncBinding(
      std::make_shared<NetBindings::SetInputWorker>(Net::unwrapSelf(info)),
      "Net::SetInputAsync",
      info);
}

NAN_METHOD(Net::Forward) {
  FF::executeSyncBinding(
      std::make_shared<NetBindings::ForwardWorker>(Net::unwrapSelf(info)),
      "Net::Forward",
      info);
}

NAN_METHOD(Net::ForwardAsync) {
  FF::executeAsyncBinding(
      std::make_shared<NetBindings::ForwardWorker>(Net::unwrapSelf(info)),
      "Net::ForwardAsync",
      info);
}

NAN_METHOD(Net::GetLayer) {
  FF::TryCatch tryCatch("Net::GetLayer");
  cv::dnn::Net self = Net::unwrapSelf(info);

  std::string layerName;
  int layerId = -1;

  if (FF::StringConverter::optArg(0, &layerName, info)) {
    return tryCatch.reThrow();
  }

  if (layerName.empty() && FF::IntConverter::optArg(0, &layerId, info)) {
    return tryCatch.reThrow();
  }

  cv::Ptr<cv::dnn::Layer> layer;
  if (!layerName.empty()) {
    layer = self.getLayer(self.getLayerId(layerName));
  } else {
    layer = self.getLayer(layerId);
  }

  info.GetReturnValue().Set(LayerConverter::wrap(layer));
}

NAN_METHOD(Net::SetLayerBlobs) {
  FF::TryCatch tryCatch("Net::SetLayerBlobs");

  int layerId;
  v8::Local<v8::Value> jsBlobs;

  if (FF::IntConverter::arg(0, &layerId, info) || info.Length() < 2 || !info[1]->IsArray()) {
    return tryCatch.throwError("Invalid arguments");
  }

  jsBlobs = info[1];

  FF::executeSyncBinding(
      std::make_shared<NetBindings::SetLayerBlobsWorker>(Net::unwrapSelf(info), layerId, jsBlobs),
      "Net::SetLayerBlobs",
      info);
}


NAN_METHOD(Net::GetLayerNames) {
  FF::executeSyncBinding(
      std::make_shared<NetBindings::GetLayerNamesWorker>(Net::unwrapSelf(info)),
      "Net::GetLayerNames",
      info);
}

NAN_METHOD(Net::GetLayerId) {
  FF::TryCatch tryCatch("Net::GetLayerId");
  
  cv::dnn::Net self = Net::unwrapSelf(info);
  std::string layerName;

  // Ensure we get the layer name from the arguments
  if (FF::StringConverter::arg(0, &layerName, info)) {
    return tryCatch.reThrow();
  }

  // Now instantiate the GetLayerIdWorker with both the Net and layerName
  FF::executeSyncBinding(
      std::make_shared<NetBindings::GetLayerIdWorker>(self, layerName),
      "Net::GetLayerId",
      info);
}


NAN_METHOD(Net::GetLayerNamesAsync) {
  FF::executeAsyncBinding(
      std::make_shared<NetBindings::GetLayerNamesWorker>(Net::unwrapSelf(info)),
      "Net::GetLayerNamesAsync",
      info);
}

NAN_METHOD(Net::GetUnconnectedOutLayers) {
  FF::executeSyncBinding(
      std::make_shared<NetBindings::GetUnconnectedOutLayersWorker>(Net::unwrapSelf(info)),
      "Net::GetUnconnectedOutLayers",
      info);
}

NAN_METHOD(Net::GetUnconnectedOutLayersAsync) {
  FF::executeAsyncBinding(
      std::make_shared<NetBindings::GetUnconnectedOutLayersWorker>(Net::unwrapSelf(info)),
      "Net::GetUnconnectedOutLayersAsync",
      info);
}

NAN_METHOD(Net::Dump) {
  FF::TryCatch tryCatch("Core::Dump");
  cv::dnn::Net self = Net::unwrapSelf(info);
  info.GetReturnValue().Set(FF::newString(self.dump()));
}

NAN_METHOD(Net::SetPreferableBackend) {
  FF::TryCatch tryCatch("Core::SetPreferableBackend");
  cv::dnn::Net self = Net::unwrapSelf(info);
  int backendId;
  if (FF::IntConverter::arg(0, &backendId, info)) {
    return tryCatch.reThrow();
  }
  self.setPreferableBackend(backendId);
}

NAN_METHOD(Net::SetPreferableTarget) {
  FF::TryCatch tryCatch("Core::SetPreferableTarget");
  cv::dnn::Net self = Net::unwrapSelf(info);
  int targetId;
  if (FF::IntConverter::arg(0, &targetId, info)) {
    return tryCatch.reThrow();
  }
  self.setPreferableTarget(targetId);
}

// ret {	retval: number, timings: number[] }

NAN_METHOD(Net::GetPerfProfile) {
  FF::TryCatch tryCatch("Core::GetPerfProfile");
  cv::dnn::Net self = Net::unwrapSelf(info);

  // int64 cv::dnn::Net::getPerfProfile	(	std::vector< double > & 	timings	)
  std::vector<double> layersTimes;
  int64 time = self.getPerfProfile(layersTimes);

  v8::Local<v8::Object> obj = Nan::New<v8::Object>();

  Nan::Set(obj, Nan::New("retval").ToLocalChecked(), FF::DoubleConverter::wrap(time));
  Nan::Set(obj, Nan::New("timings").ToLocalChecked(), FF::DoubleArrayConverter::wrap(layersTimes));

  info.GetReturnValue().Set(obj);
}

#endif
