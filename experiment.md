# 基于Intel的大模型加速技术实验

### 实验过程

启动一个PAI-DSW实例。

<img src="image\1.png" alt="54" style="zoom:30%;" />

启动实例成功后，查看Notebook，就可以看到一个JupyterLab界面。

<img src="image\2.png" alt="54" style="zoom:30%;" />

在这个界面，可以执行一些python代码。

<img src="image\3.png" alt="54" style="zoom:40%;" />

接下来，需要对大模型运行的环境进行配置。选择other-terminal，就可以打开服务器的终端，并查看服务器的一些配置信息。

<img src="image\4.png" alt="54" style="zoom:30%;" />

接着，使用如下命令对大模型环境进行配置。

```shell
cd /opt/conda/envs 
mkdir itrex
wget https://idz-ai.oss-cn-hangzhou.aliyuncs.com/LLM/itrex.tar.gz
tar -zxvf itrex.tar.gz -C itrex/ 
conda activate itrex 
python -m ipykernel install --name itrex 
```

配置好服务器环境后，启动一个基于itrex的Notebook，在Notebook就可以进行代码的编写和运行了。

<img src="image\5.png" alt="54" style="zoom:40%;" />

接下来，在Notebook将中文大模型和embedding模型下载下来，代码和结果如下。

```shell
! git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git 
! git clone https://www.modelscope.cn/AI-ModelScope/bge-base-zh-v1.5.git
```

<img src="image\6.png" alt="54" style="zoom:50%;" />

接着，新建一个名为sample.jsonl的文件，将数据集放进去，下面是一个例子。

```
{"content": "cnvrg.io 网站由 Yochay Ettun 和 Leah Forkosh Kolben 创建.", "link": 0} 
```

回到Notebook，执行下面的代码以构建一个聊天机器人chatbot。

```python
from intel_extension_for_transformers.neural_chat import PipelineConfig 
from intel_extension_for_transformers.neural_chat import build_chatbot 
from intel_extension_for_transformers.neural_chat import plugins 
from intel_extension_for_transformers.transformers import RtnConfig 
 
plugins.retrieval.enable = True 
plugins.retrieval.args['embedding_model'] = "./bge-base-zh-v1.5" 
plugins.retrieval.args["input_path"]="./sample.jsonl" 
config = PipelineConfig(model_name_or_path='./chatglm3-6b', plugins=plugins, optimization_config=RtnConfig(compute_dtype="int8", weight_dtype="int4_fullrange")) 
 
chatbot = build_chatbot(config) 
```

构建好chatbot后，向chatbot提问，就可以得到之前在数据集中填入的回答。

```python
plugins.retrieval.enable=True # enable retrieval 
response = chatbot.predict(query="cnvrg.io网站是由谁创建的？") 
print(response) 
```

<img src="image\7.png" alt="54" style="zoom:40%;" />

### 个人心得

相比于传统的服务器搭建大模型的方法，本次实验采用了Notebook方式进行，降低了学习成本和技术门槛。通过本次的实验也锻炼了自己部署和运行大模型的经验。