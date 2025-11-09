import jsonlines
import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class LangchainGPT:
    def __init__(self, model_name="gpt-3.5-turbo", keys_path=None):
        self.model_name = model_name
        self.keys = self._load_keys(keys_path) if keys_path else []
        self.current_key_index = 0
        
        # 设置初始API密钥
        if self.keys:
            os.environ["OPENAI_API_KEY"] = self.keys[self.current_key_index]
        
        #0.原代码
        # 创建模型和提示模板
        self.model = ChatOpenAI(model=self.model_name)# 4.ToT调整temperature temperature=0.7
        self.prompt = ChatPromptTemplate.from_messages([
            ("user", "{input}")
        ])
        '''
        #1.Zero-Shot Prompting
        # 增强的system prompt
        system_prompt = """你是一名资深临床药师，负责国家药师资格考试自动阅卷。请遵循：
                1. 严格依据《中国药典》(2020版)和《临床用药指南》
                2. 特别关注：
                   - CYP450酶系药物相互作用
                   - 肝肾功能不全剂量调整
                   - FDA妊娠分级D/X类药物
                3. 答案格式必须为纯选项字母"""


        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}")
        ])

        # 使用更低temperature提高确定性
        self.model = ChatOpenAI(
            model=self.model_name,
            temperature=0.2,  # 降低随机性
            max_tokens=50  # 限制输出长度
        )
        '''
        # 创建处理链
        self.chain = self.prompt | self.model | StrOutputParser()

    
    #4.ToT
    def _is_tot_prompt(self, message):
        """检测是否是ToT提示"""
        return "【思考步骤】" in message
    
    def _process_tot(self, message):
        """处理ToT提示的多阶段生成"""
        # 第一阶段：生成思考路径
        stage1_prompt = message + "\n请完成【思考步骤】中的角度分析部分"
        thoughts = self.chain.invoke({"input": stage1_prompt})
        
        # 第二阶段：评估和选择
        stage2_prompt = f"{message}\n当前分析结果:\n{thoughts}\n请继续完成评估和选择步骤"
        evaluation = self.chain.invoke({"input": stage2_prompt})
        
        # 第三阶段：最终答案
        stage3_prompt = f"{message}\n当前进展:\n{thoughts}\n{evaluation}\n请给出最终答案选项"
        final_answer = self.chain.invoke({"input": stage3_prompt})
        
        return f"思考过程:\n{thoughts}\n评估结果:\n{evaluation}\n最终答案:\n{final_answer}"
    

    
    def _load_keys(self, keys_path):
        """从文件加载API密钥"""
        keys = []
        with open(keys_path, 'r') as f:
            for line in f:
                key = line.strip()
                if key:
                    keys.append(key)
        return keys
    
    def _rotate_key(self):
        """轮换到下一个API密钥"""
        if not self.keys:
            return
        
        self.current_key_index = (self.current_key_index + 1) % len(self.keys)
        os.environ["OPENAI_API_KEY"] = self.keys[self.current_key_index]
        # 更新模型以使用新的API密钥
        self.model = ChatOpenAI(model=self.model_name)
        # 重新创建处理链
        self.chain = self.prompt | self.model | StrOutputParser()
    
    def __call__(self, message):
        '''
        """处理消息并返回响应"""
        if message is None or message == "":
            return "Your input is empty."
        
        max_attempts = min(len(self.keys), 5) if self.keys else 1
        attempts = 0
        
        while attempts < max_attempts:
            try:
                response = self.chain.invoke({"input": message})
                return response
            except Exception as e:
                print(f"Error with key {self.current_key_index}: {e}")
                attempts += 1
                if attempts < max_attempts:
                    self._rotate_key()
                else:
                    return f"Failed after {attempts} attempts. Last error: {e}"
        '''
        #4.ToT
        if not message:
            return "Empty input"
            
        max_attempts = min(len(self.keys), 5) if self.keys else 1
        
        for attempt in range(max_attempts):
            try:
                if self._is_tot_prompt(message):
                    return self._process_tot(message)
                return self.chain.invoke({"input": message})
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {str(e)}")
                if attempt < max_attempts - 1:
                    self._rotate_key()
        
        return f"Failed after {max_attempts} attempts"
        


def langchain_datagen(args):
    """使用LangChain处理数据生成"""
    # 初始化LangChain模型
    lgpt = LangchainGPT(model_name=args.model_name, keys_path=args.keys_path)
    
    def process_item(item):
        """处理单个数据项"""
        item["model_answer"] = lgpt(item["query"])
        return item
    
    output_path = args.output_path
    input_path = args.input_path
    
    # 收集已处理项目的ID
    processed_ids = set()
    if os.path.exists(output_path):
        with jsonlines.open(output_path, "r") as f:
            for item in f:
                processed_ids.add(item.get("id", None))
    
    # 收集未处理的项目
    items_to_process = []
    with jsonlines.open(input_path, "r") as reader:
        for item in reader:
            item_id = item.get("id", None)
            if item_id is not None and item_id in processed_ids:
                continue
            items_to_process.append(item)
    
    # 多线程并行处理
    with jsonlines.open(
        output_path, "a" if os.path.exists(output_path) else "w"
    ) as writer:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(process_item, item): item for item in items_to_process
            }
            
            # 使用tqdm显示进度
            for future in tqdm(
                futures, total=len(items_to_process), desc="处理项目中"
            ):
                try:
                    writer.write(future.result())
                except Exception as e:
                    print(
                        f"处理项目时出错: {futures[future]['query']}. 错误: {e}"
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用LangChain并发处理JSONL文件。")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o",
        help="要使用的OpenAI模型名称。",
    )
    parser.add_argument(
        "--keys_path",
        type=str,
        required=True,
        help="OpenAI API密钥文件路径。",
    )
    parser.add_argument(
        "--input_path", 
        type=str, 
        required=True, 
        help="输入JSONL文件的路径。"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True, 
        help="输出JSONL文件的路径。"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="并发处理的最大工作线程数。",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://api.yesapikey.com/v1",
        help="API基础URL。",
    )
    
    args = parser.parse_args()
    print(f"Using url: {args.base_url}")
    os.environ["OPENAI_BASE_URL"] = args.base_url
    langchain_datagen(args) 