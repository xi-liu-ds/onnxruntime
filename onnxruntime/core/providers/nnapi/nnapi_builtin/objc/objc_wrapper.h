//
//  objc_wrapper.h
//  cpp_objcpp
//
//  Created by gwang on 1/5/21.
//

#ifndef objc_wrapper_h
#define objc_wrapper_h

#include <memory>

class ObjCWrapperInternal;

class ObjCWrapper {
public:
    ObjCWrapper(const std::string& msg);
    ~ObjCWrapper();
    
    void print();
    
private:
    std::unique_ptr<ObjCWrapperInternal> wrapper_;
    
};

#endif /* objc_wrapper_h */
