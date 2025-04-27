# tensorRT inference  

tensorRT 推理onnx模型分为两部
- 解析onnx模型，生成engine数据
- 加载engine数据，构建context，进行推理

### 一 解析onnx模型生成engine数据

![](imgs/TensorRTParseOnnx.png "tensorRT 解析onnx") 

coda流程  
1. 创建构建工具  
``` c++  
NVlogger nvlogger;
auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(nvlogger));
auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1U<<static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));//默认配置
auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, nvlogger));
```  
2. 读取onnx数据
``` c++
std::string onnxPath="/path to onnx model/";
std::ifstream infile(onnxPath, std::ios::binary|std::ios::in);
infile.seekg(0,infile.end);
size_t dataLen=infile.tellg();
infile.seekg(0,infile.beg);
std::vector<char> onnxData(dataLen);
infile.read(onnxData.data(), onnxData.size());
infile.close();
```  
3. 解析onnx输出生成network  
``` c++
bool isParsed = parser->parse(onnxBuffer.data(), onnxBuffer.size()); //是否解析成功
int inputNum=network->getNbInputs();
const auto inputNode=network->getInput(i); //for i in range(0, inputNum)
const auto inputName=inputNode->getName(); //获取当前节点名
Dims inputDims=inputNode->getDimensions(); //获取当前节点形状，如果是动态shape，则对应维度为-1
// output同理
```  
4. 设置 IBuilderConfig  
``` c++
auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
config->setMaxWorkspaceSize(1 << 30); //设置中间工作数据显存1GB，engine和context不占用这部分显存，仅用来做中间计算数据缓存。
``` 
若是动态shape，则设置优化参数:  
``` c++
nvinfer1::IOptimizationProfile* defaultProfile = builder->createOptimizationProfile();
//for i in range(0, inputNum) 对应每个输入节点进行设置
defaultProfile->setDimensions(inputName_i, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(b0, c0, h0, w0));
defaultProfile->setDimensions(inputName_i, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(b1, c1, h1, w1));
defaultProfile->setDimensions(inputName_i, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(b2, c2, h2, w2));

config->addOptimizationProfile(defaultProfile);
```  
5. 生成engine数据,并保存
``` c++
nvinfer1::IHostMemory* hostSerialData = builder->buildSerializedNetwork(*network, *config);
auto serialData = std::vector<char>(static_cast<char*>(hostSerialData->data()), static_cast<char*>(hostSerialData->data()) + hostSerialData->size());

std::string enginePath="/path to engine";
std::ofstream outfile(enginePath, std::ios::out | std::ios::binary);
outfile.write(serialData.data(), serialData.size());
outfile.close();
```  
6. 生成推理实例
``` c++
auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(nvlogger));
auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(hostSerialData->data(), hostSerialData->size()));
auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
```  

### 二 推理

