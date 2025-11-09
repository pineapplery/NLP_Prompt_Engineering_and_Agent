import json
import argparse
"""
#0.原代码
prompt = '''
下面是一道{question_type}，请先详细分析问题，最后给出选项。
{question}
{option}
'''
def generate_query(data):
    chatgpt_query = prompt
    question = data['question']
    option = '\n'.join([k+'. '+v for k,v in data['option'].items() if v != ''])
    chatgpt_query = chatgpt_query.format_map({'question':question,'option':option,'question_type':data['question_type']})
    return chatgpt_query
"""

'''
#1.Zero-Shot Prompting
PROMPT_TEMPLATE = """【药师专业考试答题指令】
你正在参加2021年中国国家药师职业资格考试，请严格根据以下要求作答：

考试重点提醒：
- 特别注意药物相互作用和配伍禁忌
- 关注特殊人群（孕妇/儿童/老人）用药安全
- 剂量计算题需双重验证
- 优先选择最保守的治疗方案

题目类型：{question_type}
题目内容：{question}
选项：
{option}

作答要求：
1. 必须且只能输出选项字母（如"A"或"B,C"）
2. 禁止添加任何解释性文字
3. 不确定时选择最符合临床安全的答案

请直接输出答案："""

def generate_query(data):
    # 格式化选项
    option_text = '\n'.join([f"{k}. {v}" for k, v in data['option'].items() if v != ''])

    return PROMPT_TEMPLATE.format(
        question_type=data['question_type'],
        question=data['question'],
        option=option_text
    )
'''
'''
#2.Few-Shot Prompting
prompt = """
下面是一道{question_type}，请先详细分析问题，最后给出选项。
{question}
{option}
"""
# 在generate_query函数中添加few-shot示例
def generate_query(data):
    few_shot_examples = """
    【药师考试答题示例】
    示例1:
    问题：哪种药物与华法林合用会增加出血风险？
    选项：
    A) 阿司匹林
    B) 维生素K
    C) 头孢曲松
    D) 对乙酰氨基酚
    答案：A

    示例2:
    问题：孕妇禁用的抗生素是？
    选项：
    A) 青霉素
    B) 四环素
    C) 头孢菌素
    D) 阿奇霉素
    答案：B
    """

    prompt = f"""【药师专业考试规则】
{few_shot_examples}
请根据上述示例格式回答以下问题：

题目：{data['question']}
选项：
{format_options(data['option'])}
答案："""  # 注意这里故意不闭合，引导模型只输出字母

    return prompt

def format_options(options):
    return '\n'.join([f"{k}) {v}" for k,v in options.items() if v])
'''
'''
#3. Chain-of-Thought Prompting
prompt = """
下面是一道{question_type}，请先详细分析问题，最后给出选项。
{question}
{option}
"""
def generate_query(data):
    cot_examples = """
    【药师考试推理示例】
    示例1:
    问题：哪种药物与华法林合用会增加出血风险？
    选项：
    A) 阿司匹林
    B) 维生素K
    C) 头孢曲松
    D) 对乙酰氨基酚
    推理过程：
    1. 华法林是抗凝药，与抗血小板药合用会增效
    2. 阿司匹林是抗血小板药
    3. 维生素K是华法林拮抗剂
    4. 头孢曲松可能影响肠道菌群产生维生素K
    5. 对乙酰氨基酚不干扰凝血
    答案：A

    示例2:
    问题：孕妇禁用的抗生素是？
    选项：
    A) 青霉素
    B) 四环素
    C) 头孢菌素
    D) 阿奇霉素
    推理过程：
    1. 青霉素类孕期安全分级为B
    2. 四环素类会导致胎儿牙齿染色
    3. 头孢菌素类通常安全
    4. 阿奇霉素在必要时可谨慎使用
    答案：B
    """

    prompt = f"""【药师专业考试-推理题】
{cot_examples}
请按以下步骤作答：
1. 分析题目关键信息
2. 逐步排除错误选项
3. 给出最终答案

当前题目：{data['question']}
选项：
{format_options(data['option'])}

请开始推理："""  # 不闭合提示以引导模型逐步回答

    return prompt

def format_options(options):
    return '\n'.join([f"{k}) {v}" for k,v in options.items() if v])
'''

#4. Tree of Thoughts (ToT)
# 基础提示模板（原版）
basic_prompt = '''
下面是一道{question_type}，请先详细分析问题，最后给出选项。
{question}
{option}
'''
# ToT提示模板
tot_prompt_template = '''
你正在解决一道{question_type}题目，请按照以下结构化思考过程逐步分析：

【题目理解】
{question}
{option}

【思考步骤】
1. 问题分解：将复杂问题拆解为关键子问题
2. 多角度分析：从3个不同角度分析解题思路
   - 角度1：{{
   - 角度2：{{
   - 角度3：{{
3. 可能性评估：对每个角度的可行性评分（1-5分）
4. 最优路径选择：综合评估后选择最佳解法
5. 答案验证：检查答案是否满足所有题目条件

请按照这个框架思考，并在最后明确给出答案选项。
'''

def generate_query(data):
    # 自动根据问题类型决定是否使用ToT
    use_tot = data['question_type'] in ['综合分析选择题', '多项选择题']  # 对复杂题型自动启用
    
    question = data['question']
    option = '\n'.join([k+'. '+v for k,v in data['option'].items() if v != ''])
    
    if use_tot:
        # 构建ToT提示词时插入动态部分
        prompt = tot_prompt_template.format_map({
            'question_type': data['question_type'],
            'question': question,
            'option': option
        })
        
        # 添加具体角度提示（根据题型定制）
        if data['question_type'] == '综合分析选择题':
            prompt = prompt.replace('角度1：{', '角度1：临床相关性分析 {')
            prompt = prompt.replace('角度2：{', '角度2：病理机制分析 {')
            prompt = prompt.replace('角度3：{', '角度3：治疗方案比较 {')
        else:
            prompt = prompt.replace('角度1：{', '角度1：概念定义分析 {')
            prompt = prompt.replace('角度2：{', '角度2：排除法验证 {')
            prompt = prompt.replace('角度3：{', '角度3：反向推理 {')
    else:
        prompt = basic_prompt.format_map({
            'question': question,
            'option': option,
            'question_type': data['question_type']
        })
    
    return prompt


def Prepare_data(args):
    data = []
    # 读取上传的JSON文件
    with open(args.input_path, encoding='utf-8') as f:
        data = json.load(f)

    print(f"len:{len(data)}")
    # 根据要求转换
    jsonl_data = []

    for id, item in enumerate(data):
        jsonl_data.append(
            {
                "id":id,
                "query": generate_query(item),
                "model_answer": "",
                "question_type": item['question_type'],
                "groundtruth": item['answer']
            }
        )

    # 将转换后的数据保存为JSONL文件
    with open(args.output_path, "w", encoding="utf-8") as file:
        for entry in jsonl_data:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"Prepare finished, output to '{args.output_path}'")
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare data for OpenAIGPT generation")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output JSONL file.")
    args = parser.parse_args()
    Prepare_data(args)
