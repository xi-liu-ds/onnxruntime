//
//  objc_wrapper.m
//  cpp_objcpp
//
//  Created by gwang on 1/5/21.
//

#import <Foundation/Foundation.h>

#import "objc_wrapper.h"
#include <string>

@interface ObjCTest : NSObject {
  NSString* _msg;
}

- (instancetype)initWithMsg:(const std::string&)msg;
- (void)print;
- (void)dealloc;

@end

class ObjCWrapperInternal {
 public:
  ObjCWrapperInternal(const std::string& msg);
  ~ObjCWrapperInternal() = default;

  void print();

 private:
  ObjCTest* obj_;
};

ObjCWrapperInternal::ObjCWrapperInternal(const std::string& msg) {
  obj_ = [[ObjCTest alloc] initWithMsg:msg];
}

void ObjCWrapperInternal::print() {
  [obj_ print];
}

ObjCWrapper::ObjCWrapper(const std::string& msg)
    : wrapper_(std::make_unique<ObjCWrapperInternal>(msg)) {
}

ObjCWrapper::~ObjCWrapper() = default;

void ObjCWrapper::print() {
  wrapper_->print();
}

@implementation ObjCTest

- (instancetype)initWithMsg:(const std::string&)msg {
  self = [super init];
  _msg = [NSString stringWithUTF8String:msg.c_str()];
  return self;
}

- (void)print {
  NSLog(@"msg %@, yeah, I am here", _msg);
}

- (void)dealloc {
  NSLog(@"dealloced this class!!");
}

@end
