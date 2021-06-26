#include <iostream>
#include <chrono>
#include <string>
#include <sstream>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1
static const int INPUT_C = 3;
static const int INPUT_H = 160;
static const int INPUT_W = 160;
static const int OUTPUT_SIZE = 143;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;


ILayer* conv_bn_relu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int kernelsize, int stride, int padding, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{kernelsize, kernelsize}, weightMap[lname + ".0.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{stride, stride});
    conv1->setPaddingNd(DimsHW{padding, padding});
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    return relu1;
}

ILayer* dw_bn_relu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, 
                    int inch, int outch, int stride, int padding, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, inch, DimsHW{3, 3}, weightMap[lname + ".0.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{stride, stride});
    conv1->setPaddingNd(DimsHW{padding, padding});
    conv1->setNbGroups(inch);

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{1, 1}, weightMap[lname + ".3.weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{1, 1});

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".4", 1e-5);
    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    return relu2;
}

ILayer* reduce_mean(INetworkDefinition* network, ITensor& input) {
	Dims dims = input.getDimensions();
	IPoolingLayer* pool = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{ dims.d[1], dims.d[2] });
	return pool;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder,  DataType dt) {
    INetworkDefinition* network = builder->createNetwork();
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{INPUT_C, INPUT_H, INPUT_W });
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights("../slim_160_latest.wts");
    auto x = conv_bn_relu(network, weightMap, *data, 16, 3, 2, 1, "conv1");
    x = dw_bn_relu(network, weightMap, *x->getOutput(0), 16, 32, 1, 1, "conv2");
    x = dw_bn_relu(network, weightMap, *x->getOutput(0), 32, 32, 2, 1, "conv3");
    x = dw_bn_relu(network, weightMap, *x->getOutput(0), 32, 32, 1, 1, "conv4");
    x = dw_bn_relu(network, weightMap, *x->getOutput(0), 32, 64, 2, 1, "conv5");
    x = dw_bn_relu(network, weightMap, *x->getOutput(0), 64, 64, 1, 1, "conv6");
    x = dw_bn_relu(network, weightMap, *x->getOutput(0), 64, 64, 1, 1, "conv7");
    x = dw_bn_relu(network, weightMap, *x->getOutput(0), 64, 64, 1, 1, "conv8");
    auto output1 = x;

    x = dw_bn_relu(network, weightMap, *x->getOutput(0), 64, 128, 2, 1, "conv9");
    x = dw_bn_relu(network, weightMap, *x->getOutput(0), 128, 128, 1, 1, "conv10");
    x = dw_bn_relu(network, weightMap, *x->getOutput(0), 128, 128, 1, 1, "conv11");
    auto output2 = x;

    x = dw_bn_relu(network, weightMap, *x->getOutput(0), 128, 256, 2, 1, "conv12");
    x = dw_bn_relu(network, weightMap, *x->getOutput(0), 256, 256, 1, 1, "conv13");
    auto output3 = x;

    
    output1 = reduce_mean(network, *output1->getOutput(0));
    output2 = reduce_mean(network, *output2->getOutput(0));
    output3 = reduce_mean(network, *output3->getOutput(0));

    ITensor* inputTensors[] = {output1->getOutput(0), output2->getOutput(0), output3->getOutput(0)};
    IConcatenationLayer* multi_scale = network->addConcatenation(inputTensors, 3);
    assert(multi_scale);
#if 1
    /* just for debug */
    Dims dims1 = multi_scale->getOutput(0)->getDimensions();
    for (int i = 0; i < dims1.nbDims; i++)
    {
        std::cout << dims1.d[i] << "   ";
    }
    std::cout << std::endl;
#endif
    IConvolutionLayer* fc = network->addConvolutionNd(*multi_scale->getOutput(0), 143, DimsHW{ 1, 1 }, weightMap["fc.weight"], weightMap["fc.bias"]);
    fc->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*fc->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    if(builder->platformHasFastFp16()) {
        std::cout << "Platform supports fp16 mode and use it !!!" << std::endl;
        builder->setFp16Mode(true);
    } else {
        std::cout << "Platform doesn't support fp16 mode so you can't use it !!!" << std::endl;
    }
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

   return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float),
          cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float),
          cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{ nullptr };
    size_t size{ 0 };

    if (argc == 2 && std::string(argv[1]) == "-s")
    {
            IHostMemory* modelStream{ nullptr };
            APIToModel(BATCH_SIZE, &modelStream);
            assert(modelStream != nullptr);
            std::ofstream p("slim.engine", std::ios::binary);
            if (!p) {
                    std::cerr << "could not open plan output file" << std::endl;
                    return -1;
            }
            p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
            modelStream->destroy();
            return 0;
    }
    else if (argc == 2 && std::string(argv[1]) == "-d")
    {
            std::ifstream file("slim.engine", std::ios::binary);
            if (file.good()) {
                    file.seekg(0, file.end);
                    size = file.tellg();
                    file.seekg(0, file.beg);
                    trtModelStream = new char[size];
                    assert(trtModelStream);
                    file.read(trtModelStream, size);
                    file.close();
            }
    }
    else
    {
            std::cerr << "arguments not right!" << std::endl;
            std::cerr << "./slim -s  // serialize model to plan file" << std::endl;
            std::cerr << "./slim -d   // deserialize plan file and run inference" << std::endl;
            return -1;
    }

    /* prepare input data */
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Subtract mean from image
    static float data[3 * INPUT_H * INPUT_W];
    for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
        data[i] = 1.0;

    // Run inference
    static float prob[OUTPUT_SIZE];
    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, 1);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";
    for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    {
        std::cout << prob[i] << ", ";
        if (i % 10 == 0) std::cout << i / 10 << std::endl;
    }
    std::cout << std::endl;

    return 0;
    return 0;
}
