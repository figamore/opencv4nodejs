#include "opencv2/dnn.hpp"
#include "Mat.h"

class LayerConverter {
public:
  static v8::Local<v8::Object> wrap(cv::Ptr<cv::dnn::Layer> layer) {
    v8::Local<v8::Object> obj = Nan::New<v8::Object>();
    
    // Add blobs field
    Nan::Set(obj, Nan::New("blobs").ToLocalChecked(), wrapBlobs(layer->blobs));

    return obj;
  }

  static std::vector<cv::Mat> unwrapBlobs(v8::Local<v8::Value> jsBlobs) {
    std::vector<cv::Mat> blobs;
    v8::Local<v8::Array> jsArray = v8::Local<v8::Array>::Cast(jsBlobs);
    for (uint32_t i = 0; i < jsArray->Length(); ++i) {
      blobs.push_back(Mat::Converter::unwrap(Nan::Get(jsArray, i).ToLocalChecked()));
    }
    return blobs;
  }

  static v8::Local<v8::Array> wrapBlobs(const std::vector<cv::Mat>& blobs) {
    v8::Local<v8::Array> jsArray = Nan::New<v8::Array>(blobs.size());
    for (size_t i = 0; i < blobs.size(); ++i) {
      Nan::Set(jsArray, i, Mat::Converter::wrap(blobs[i]));
    }
    return jsArray;
  }
};