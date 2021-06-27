#include "seq2seq.h"
#include <fstream>
#include <numeric>


const char* const DICT_PATH = "../models/dict/jieba.dict.utf8";
const char* const HMM_PATH = "../models/dict/hmm_model.utf8";
const char* const USER_DICT_PATH = "../models/dict/user.dict.utf8";
const char* const IDF_PATH = "../models/dict/idf.utf8";
const char* const STOP_WORD_PATH = "../models/dict/stop_words.utf8";

cppjieba::Jieba  jieba(DICT_PATH,
      HMM_PATH,
      USER_DICT_PATH,
      IDF_PATH,
      STOP_WORD_PATH);


void Sqeseq::setGpuIndex(int gpuIndex) {
#ifdef __VULKAN__
    if (gpuIndex >= 0) {
        net.opt.use_vulkan_compute = true;
        net.set_vulkan_device(gpuIndex);
        printf("CrnnNet try to use Gpu%d\n", gpuIndex);
    } else {
        net.opt.use_vulkan_compute = false;
        printf("CrnnNet use Cpu\n");
    }
#endif
}

Sqeseq::~Sqeseq() {
    encoder_net.clear();
    decoder_net.clear();
}

void Sqeseq::setNumThread(int numOfThread) {
    numThread = numOfThread;
}

bool Sqeseq::initModel() {

    encoder_net.load_param("../models/encoder.param");
    encoder_net.load_model("../models/encoder.bin");

    decoder_net.load_param("../models/decoder.param");
    decoder_net.load_model("../models/decoder.bin");

    std::ifstream src_vocab_in("../models/src_vocab.txt");
    std::string line;
    if (src_vocab_in) {
        int count = 0;
        while (getline(src_vocab_in, line)) {// line中不包括每行的换行符
            encoder_map[line] = count;
            count++;
        }
    }

    std::ifstream tgt_vocab_in("../models/tgt_vocab.txt");
    if (tgt_vocab_in) {
        int count = 0;
        while (getline(tgt_vocab_in, line)) {// line中不包括每行的换行符
            decoder_map[count] = line;
            count++;
        }
    }
    encoder_map_size = encoder_map.size();
    decoder_map_size = decoder_map.size();

    embeding_encode.create(encoder_hidden_size,encoder_map_size);
    embeding_decode.create(decoder_hidden_size,decoder_map_size);

    std::ifstream in_en("../models/encode_embedding_weight.bin",std::ios::in | std::ios::binary);
    in_en.read((char *) embeding_encode, sizeof(float) * encoder_hidden_size * encoder_map_size );

    std::ifstream in_de("../models/decode_embedding_weight.bin",std::ios::in | std::ios::binary);
    in_de.read((char *) embeding_decode, sizeof(float) * decoder_hidden_size * decoder_map_size );



    return true;
}

template<class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}

std::vector<std::string> Sqeseq::forward(std::string  &input_str) {
    std::vector<std::string> inputs;

    jieba.Cut(input_str, inputs, true);

    ncnn::Mat input,hidden;
    int input_len = inputs.size();


    input.create(encoder_hidden_size,input_len);
    hidden.create(decoder_hidden_size,1);

    for(int i=0;i<input_len;i++)
    {
        float *pt = input.row(i);
        const float *pt_em =  embeding_encode.row(encoder_map[inputs[i]]);
        memcpy(pt,pt_em,encoder_hidden_size*sizeof(float));

    }

    ncnn::Extractor ex = encoder_net.create_extractor();
    ex.input("input", input);
    ncnn::Mat  encode_outputs;
    ex.extract("out1", encode_outputs);
    float *ptr_hidden = hidden.row(0);
    const float *pf = encode_outputs.row(input_len - 1);
    const float *pr = encode_outputs.row(0);
    memcpy(ptr_hidden,pf,encoder_hidden_size*sizeof(float));
    memcpy(&ptr_hidden[encoder_hidden_size],&pr[encoder_hidden_size],encoder_hidden_size*sizeof(float));

    ncnn::Mat decode_input;
    decode_input.create(decoder_hidden_size,1);
    const float *pt_de =  embeding_decode.row(sos_index);
    memcpy(decode_input,pt_de,decoder_hidden_size*sizeof(float));

    std::vector<std::string> output;
    for(int i=0; i< max_len;i++)
    {
        ncnn::Extractor ex2 = decoder_net.create_extractor();
        ex2.input("inputs", decode_input);
        ex2.input("encoder_hidden", hidden);
        ex2.input("encoder_outputs", encode_outputs);
        ncnn::Mat  decode_output,decode_hidden;
        ex2.extract("out2", decode_hidden);

//        ex.extract("out3", step_attn);
        ex2.extract("out1", decode_output);

        const float *ptr =  decode_output.row(0);
//        printf("h:%d w:%d c:%d \n",decode_hidden.h,decode_hidden.w,decode_hidden.c);
        int max_index = 0;
        float max_value = -10.f;
        for (int j=0;j<decoder_map_size;j++)
        {
            if (ptr[j] > max_value)
            {
                max_index = j;
                max_value = ptr[j];
            }
        }
        const float *pt_de =  embeding_decode.row(max_index);
        memcpy(decode_input,pt_de,decoder_hidden_size*sizeof(float));
        memcpy(hidden,decode_hidden,decoder_hidden_size*sizeof(float));

//        printf("%s \n",decoder_map[max_index] .c_str());
        if (decoder_map[max_index] == "<eos>") break;
        if (decoder_map[max_index] == "<unk>") continue;
        output.push_back(decoder_map[max_index]);
    }
    return output;
}
