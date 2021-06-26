#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 160;
static const int INPUT_W = 160;
static const int OUTPUT_SIZE = 143;
static const int BATCH_SIZE = 1;
const char* INPUT_BLOB_NAME = "tower_0/images";
const char* OUTPUT_BLOB_NAME = "tower_0/SpatialSqueeze";

using namespace nvinfer1;

static Logger gLogger;

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

int paddingSize(int inSize, int kernelSize, int strideSize) {
	int mode = inSize % strideSize;
	if (mode == 0)
		return std::max(kernelSize - strideSize, 0); // max( kernelSize - strideSize , 0)
	return std::max(kernelSize - (inSize % strideSize), 0); // max(kernelSize - (inSize mod strideSize), 0)
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
	float *gamma = (float*)weightMap[lname + "BatchNorm/gamma"].values;
	float *beta = (float*)weightMap[lname + "BatchNorm/beta"].values;
	float *mean = (float*)weightMap[lname + "BatchNorm/moving_mean"].values;
	float *var = (float*)weightMap[lname + "BatchNorm/moving_variance"].values;
	int len = weightMap[lname + "BatchNorm/moving_variance"].count;
 
	float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		scval[i] = gamma[i] / sqrt(var[i] + eps);
	}
	Weights scale{ DataType::kFLOAT, scval, len };
 
	float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
	}
	Weights shift{ DataType::kFLOAT, shval, len };
 
	float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		pval[i] = 1.0;
	}
	Weights power{ DataType::kFLOAT, pval, len };
 
	weightMap[lname + ".scale"] = scale;
	weightMap[lname + ".shift"] = shift;
	weightMap[lname + ".power"] = power;
	IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
	assert(scale_1);
 
	return scale_1;
}

ILayer* convBnReLU(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, 
                    int nbOutputMaps, int kernelSize, int strideSize, std::string endName) 
{
    Dims d = input.getDimensions();
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	IConvolutionLayer* conv = network->addConvolutionNd(input, nbOutputMaps, DimsHW{ kernelSize , kernelSize }, weightMap[lname + endName], emptywts);
	assert(conv);
	conv->setStrideNd(DimsHW{ strideSize, strideSize });
	int padSize = paddingSize(d.d[1], kernelSize, strideSize);
	int postPadding = ceil(padSize / 2.0);
	int prePadding = padSize - postPadding;
	if (prePadding > 0)
	    conv->setPrePadding(DimsHW{ prePadding, prePadding });
	if (postPadding > 0)
		conv->setPostPadding(DimsHW{ postPadding, postPadding });				

    //std::cout << "unit_1/conv1x1_before/ " <<  lname <<std::endl;
    // if (lname == "ShuffleNetV2/Stage3/unit_1/conv1x1_before/")
    // {
    //     std::cout << "unit_1/conv1x1_before/" << std::endl;
    //     return conv;
    // }
	IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname, 1e-5);
    IActivationLayer* relu = network->addActivation(*bn->getOutput(0), ActivationType::kRELU);
	assert(relu);
	return relu;
}

ILayer* depthwiseConvolutionNd(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname,
                                int nbOutputMaps, int kernelSize, int strideSize, bool isPointwise) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    Dims d = input.getDimensions();
    int size = d.d[0];
    IConvolutionLayer* conv = network->addConvolutionNd(input, size, DimsHW{ kernelSize, kernelSize }, weightMap[lname + "depthwise_weights"], emptywts);
    conv->setStrideNd(DimsHW{ strideSize, strideSize });
    int padSize = paddingSize(d.d[1], kernelSize, strideSize);
    int postPadding = ceil(padSize / 2.0);
    int prePadding = padSize - postPadding;
    if (prePadding > 0)
        conv->setPrePadding(DimsHW{ prePadding, prePadding });
    if (postPadding > 0)
        conv->setPostPadding(DimsHW{ postPadding, postPadding });
    
    conv->setNbGroups(size);  // 每一个通道作卷积 

    if (isPointwise)
    {
        ILayer* layer2 = convBnReLU(network, weightMap, *conv->getOutput(0), lname, nbOutputMaps, 1, 1, "pointwise_weights");
        return layer2;
    }
    else
    {
        IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname, 1e-5);
        return bn;
    }
}

std::vector<ILayer*> concat_shuffle_split(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ILayer* layer1, ILayer* layer2) {
    // 合并分支
    std::vector<ILayer*> vec;
    // channel 作拼接
    ITensor* concatTensors[] = { layer1->getOutput(0),layer2->getOutput(0) };
    IConcatenationLayer* concatLayer = network->addConcatenation(concatTensors, 2);
    assert(concatLayer);

    IShuffleLayer *shuffleLayer = network->addShuffle(*concatLayer->getOutput(0));
    assert(shuffleLayer);
    // 数据格式 tf:HWC  trt:CHW
    shuffleLayer->setFirstTranspose(Permutation{ 1, 2, 0 }); // 116 * 20 * 20

    // tf.strack(xx,axis=3)  增加一个维度  
    Dims shuffleLayerDims = shuffleLayer->getOutput(0)->getDimensions();
    Dims shuffleLayerReshapeDims = Dims4(shuffleLayerDims.d[0], shuffleLayerDims.d[1], 2, shuffleLayerDims.d[2] / 2);
    shuffleLayer->setReshapeDimensions(shuffleLayerReshapeDims);// 20 * 20 * 2 *  58 
    shuffleLayer->setSecondTranspose(Permutation{ 0, 1 ,3 ,2 });// 20 * 20  * 58 * 2

    Dims shuffleLayerTransposeDims = shuffleLayer->getOutput(0)->getDimensions();
    IShuffleLayer *shuffleLayer2 = network->addShuffle(*shuffleLayer->getOutput(0));// 20 * 20  * 58 * 2
    shuffleLayer2->setReshapeDimensions(Dims3(shuffleLayerTransposeDims.d[0], shuffleLayerTransposeDims.d[1], shuffleLayerTransposeDims.d[2] * shuffleLayerTransposeDims.d[3]));// 20 * 20  * 116 

    shuffleLayer2->setSecondTranspose(Permutation{ 2, 0, 1 }); // 116 * 20 * 20

    // 按 channel 分隔成两个tensor
    Dims mergeSpliteDims = shuffleLayer2->getOutput(0)->getDimensions();
    ISliceLayer *mergeS1 = network->addSlice(*shuffleLayer2->getOutput(0), Dims3{ 0,0, 0 }, Dims3{ mergeSpliteDims.d[0] / 2  , mergeSpliteDims.d[1], mergeSpliteDims.d[2] }, Dims3{ 1,1, 1 });
    vec.push_back(mergeS1);

    ISliceLayer *mergeS2 = network->addSlice(*shuffleLayer2->getOutput(0), Dims3{ mergeSpliteDims.d[0] / 2, 0, 0 }, Dims3{ mergeSpliteDims.d[0] / 2 , mergeSpliteDims.d[1] , mergeSpliteDims.d[2] }, Dims3{ 1,1, 1 });
    vec.push_back(mergeS2);
    return vec;
}

std::vector<ILayer*> blob(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname,
             int nbOutputMaps, int num_units)
// ILayer* blob(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname,
//             int nbOutputMaps, int num_units)
{
    Dims d = input.getDimensions();
    int input_size = d.d[0];

    std::vector<ILayer*> res;
    std::cout << "input_size " << input_size << std::endl;
    ILayer* yy = convBnReLU(network, weightMap, input, lname + "unit_1/conv1x1_before/", input_size, 1, 1, "weights");
    ILayer* y = depthwiseConvolutionNd(network, weightMap, *yy->getOutput(0), lname + "unit_1/depthwise/", input_size, 3, 2, false);

    y = convBnReLU(network, weightMap, *y->getOutput(0), lname + "unit_1/conv1x1_after/", ceil(nbOutputMaps / 2.0), 1, 1, "weights");

    ILayer* x = depthwiseConvolutionNd(network, weightMap, input, lname + "unit_1/second_branch/depthwise/", input_size, 3, 2, false);

    x = convBnReLU(network, weightMap, *x->getOutput(0), lname + "unit_1/second_branch/conv1x1_after/", ceil(nbOutputMaps / 2.0), 1, 1, "weights");

    ILayer* u_x = x;
    std::vector<ILayer*> xy;
    for (int j = 2; j < num_units + 1; j++)
    {
        xy = concat_shuffle_split(network, weightMap, u_x, y);
        int out_put = xy[0]->getOutput(0)->getDimensions().d[0];
        u_x = convBnReLU(network, weightMap, *xy[0]->getOutput(0), lname + "unit_" + std::to_string(j) + "/conv1x1_before/", out_put, 1, 1, "weights");
        u_x = depthwiseConvolutionNd(network, weightMap, *u_x->getOutput(0), lname +"unit_" + std::to_string(j) + "/depthwise/", out_put, 3, 1, false);
        u_x = convBnReLU(network, weightMap, *u_x->getOutput(0), lname + "unit_" + std::to_string(j) + "/conv1x1_after/", out_put, 1, 1, "weights");
        y =  xy[1];
    }
    ITensor* inputTensors1[] = {u_x->getOutput(0), y->getOutput(0)};
    IConcatenationLayer* cat1 = network->addConcatenation(inputTensors1, 2);
    assert(cat1);
    res.push_back(cat1);
    res.push_back(yy);
    return res;
}


ILayer* reduce_mean(INetworkDefinition* network, ITensor& input) {
	Dims dims = input.getDimensions();
	IPoolingLayer* pool = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{ dims.d[1], dims.d[2] });
	return pool;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 3, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3,INPUT_H, INPUT_W});
    assert(data);

    //std::map<std::string, Weights> weightMap = loadWeights("../shufflenetv2.wts");
    std::map<std::string, Weights> weightMap = loadWeights("./test.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    ILayer* cbr1 = convBnReLU(network, weightMap, *data, "ShuffleNetV2/init_conv/", 24, 3, 2, "weights");
    assert(cbr1);

    ILayer* dwc1 = depthwiseConvolutionNd(network, weightMap, *cbr1->getOutput(0), "ShuffleNetV2/init_conv_2/", 32, 3, 2, true);
    assert(dwc1);

    std::vector<ILayer*> b1 = blob(network, weightMap, *dwc1->getOutput(0), "ShuffleNetV2/Stage2/", 116, 4);
    std::vector<ILayer*> b2 = blob(network, weightMap, *b1[0]->getOutput(0), "ShuffleNetV2/Stage3/", 116*2, 8);
    std::vector<ILayer*> b3 = blob(network, weightMap, *b2[0]->getOutput(0), "ShuffleNetV2/Stage4/", 116*2*2, 4);

    // ILayer* b1 = blob(network, weightMap, *dwc1->getOutput(0), "ShuffleNetV2/Stage2/", 116, 4);
    // ILayer* b2 = blob(network, weightMap, *b1->getOutput(0), "ShuffleNetV2/Stage3/", 116*2, 8);
    // ILayer* b3 = blob(network, weightMap, *b2->getOutput(0), "ShuffleNetV2/Stage4/", 116*2*2, 4);

    ILayer* s1 = reduce_mean(network, *b1[0]->getOutput(0));
    ILayer* s2 = reduce_mean(network, *b2[0]->getOutput(0));
    ILayer* s3 = reduce_mean(network, *b3[0]->getOutput(0));

    ITensor* inputTensors2[] = {s1->getOutput(0), s2->getOutput(0), s3->getOutput(0)};
    IConcatenationLayer* multi_scale = network->addConcatenation(inputTensors2, 3);
    assert(multi_scale);
#if 1
    std::cout  << "b2" << std::endl;
    Dims dims = b2[0]->getOutput(0)->getDimensions();
    std::cout << b2[0]->getOutput(0)->getName() << " dims: ";
    for (int i = 0; i < dims.nbDims; i++) {
        std::cout << dims.d[i] << ", ";
    }
    std::cout << std::endl;
#endif

    IConvolutionLayer* fc = network->addConvolutionNd(*multi_scale->getOutput(0), 143, DimsHW{ 1, 1 }, weightMap["logits/weights"], weightMap["logits/biases"]);
    fc->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*fc->getOutput(0));
    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }
    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
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
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./shufflenet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./shufflenet -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("shufflenet.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("shufflenet.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        return -1;
    }

    //static float data[3 * INPUT_H * INPUT_W];


    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];

    cv::Mat img = cv::imread("1.jpg");//BGR
    cv::Mat src = img.clone();
    cv::resize(img, img, cv::Size(160,160), 0, 0, cv::INTER_LINEAR);
    for (int b = 0; b < BATCH_SIZE; b++) {
        float *p_data = &data[b * 3 * INPUT_H * INPUT_W];
        for (int i = 0; i < INPUT_H * INPUT_W; i++) {
            p_data[i] = (img.at<cv::Vec3b>(i)[0] - 123.0) / 58.;
            p_data[i + INPUT_H * INPUT_W] = (img.at<cv::Vec3b>(i)[1] - 116.0) / 57.;
            p_data[i + 2 * INPUT_H * INPUT_W] = (img.at<cv::Vec3b>(i)[2] - 103.0) / 57.;
        }
    }

#if 0 //debug
    for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    {
        if (i > 160)
            return -1;
        std::cout << data[i] << std::endl;
    }
#endif

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Run inference
    static float prob[BATCH_SIZE*OUTPUT_SIZE];
    for (int i = 0; i < 10; i++) {
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, 1);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";
    for(int j = 0; j < OUTPUT_SIZE/2; j++)
    {
        float x = prob[2*j] * src.cols;
        float y = prob[2*j + 1] * src.rows;
        std::cout << x << " " << y << std::endl;
        cv::circle(src, cv::Point(x, y), 2, cv::Scalar(0,0,255), -1);
    }
    //cv::imshow("result", src);
    cv::imwrite("result.jpg", src);
    //cv::waitKey(0);
    std::cout << std::endl;
    return 0;
}
