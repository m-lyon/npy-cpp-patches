#include "npy.hpp"


template<typename T>
class SimplePatcher{

    private:
        size_t start;
        char* buf;
        std::string filepath;
        std::ifstream stream;
        std::vector<size_t> data_shape;
        std::vector<T> patch;
        void first();
        void second();
        void third();
    public:
        std::vector<T> get_patch(std::string&);
};

template<typename T>
std::vector<T> SimplePatcher<T>::get_patch(std::string& fpath){
    filepath = fpath;
    first();
    second();
    third();

    return patch;
}

template<typename T>
void SimplePatcher<T>::first(){
    stream.open(filepath, std::ifstream::binary);
    std::string header_s = npy::read_header(stream);
    start = stream.tellg();
    npy::header_t header = npy::parse_header(header_s);
    data_shape = header.shape;
    std::reverse(data_shape.begin(), data_shape.end());
    patch.resize(8, 5);
    buf = reinterpret_cast<char *>(patch.data());
}

template<typename T>
void SimplePatcher<T>::second(){
    for (int i = 0; i < 4; i++){
        stream.read(buf, 16);
        buf += 16;
    }
}

template<typename T>
void SimplePatcher<T>::third(){
    stream.close();
}