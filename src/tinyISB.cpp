#include "tinyISB.h"

// Explicit instantiation and library entry points
namespace tinyisb {

const char* get_version() {
    return TINYISB_VERSION_STRING;
}

} // namespace tinyisb
