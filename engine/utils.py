import os
from PIL import Image
import openai
import numpy as np
import copy

from .step_interpreters import register_step_interpreters, parse_step


class Program:
    def __init__(self,prog_str,init_state=None):
        self.prog_str = prog_str
        self.state = init_state if init_state is not None else dict()  # 初始化程序状态字典
        self.instructions = self.prog_str.split('\n')


class ProgramInterpreter:
    def __init__(self,dataset='nlvr'):
        self.step_interpreters = register_step_interpreters(dataset)

    def execute_step(self,prog_step,inspect):
        step_name = parse_step(prog_step.prog_str,partial=True)['step_name']
        print(step_name)
        return self.step_interpreters[step_name].execute(prog_step,inspect)

    def execute(self,prog,init_state,inspect=False):
        if isinstance(prog,str):
            prog = Program(prog,init_state)
        else:
            assert(isinstance(prog,Program))

        prog_steps = [Program(instruction,init_state=prog.state) \
            for instruction in prog.instructions]

        html_str = '<hr>'
        for prog_step in prog_steps:
            if inspect:
                step_output, step_html = self.execute_step(prog_step,inspect)
                html_str += step_html + '<hr>'
            else:
                step_output = self.execute_step(prog_step,inspect)

        if inspect:
            return step_output, prog.state, html_str

        return step_output, prog.state


SYSTEM_MSG = (
    'Complete the visual program for the last question. '
    'Output only the program lines, one step per line, no explanation.'
)


class ProgramGenerator():
    def __init__(self,prompter,temperature=0.7,top_p=0.5,prob_agg='mean',model=None):
        """
        初始化程序生成器
        :param prompter: 接收 inputs 字典并返回完整 prompt 字符串的函数（通常由 partial 绑定好固定参数）
        :param temperature: 采样温度，控制生成多样性
        :param top_p: 核采样（nucleus sampling）阈值
        :param model: 模型名；默认读取环境变量 OPENAI_MODEL，未设置则用 gpt-3.5-turbo
        :param prob_agg: 概率聚合方式（Probability Aggregation）：
                         'mean' 表示对各 token 的 logprob 求均值后再取指数（即几何平均概率，消除程序长度影响）；
                         'sum' 表示对各 token 的 logprob 求和后再取指数（即整段序列的联合生成概率）
        """
        # 密钥与 base_url 分别读取 OPENAI_API_KEY / OPENAI_BASE_URL，可接任意 OpenAI 兼容服务
        self.client = openai.OpenAI()
        self.model = model or os.getenv("OPENAI_MODEL","gpt-3.5-turbo")
        self.prompter = prompter
        self.temperature = temperature
        self.top_p = top_p
        self.prob_agg = prob_agg

    def compute_prob(self,response):
        """根据模型返回的 token 对数概率（logprobs），计算整段生成程序的置信度/概率"""
        # 部分兼容服务不返回 logprobs
        if response.choices[0].logprobs is None:
            return None
        token_logprobs = [t.logprob for t in response.choices[0].logprobs.content]

        # 选择对数概率的聚合方式
        if self.prob_agg=='mean':
            agg_fn = np.mean  # 长度归一化的平均对数概率，对应几何平均数
        elif self.prob_agg=='sum':
            agg_fn = np.sum   # 对数概率求和，对应所有 token 概率连乘的联合概率
        else:
            raise NotImplementedError

        # 先聚合各 token 的 logprob，再通过 exp(x) 还原为 [0, 1] 区间的概率值
        return np.exp(agg_fn(token_logprobs))

    def generate(self,inputs):
        # 调用 chat 接口生成程序，logprobs=True 以返回每个生成 token 的对数概率
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role":"system","content":SYSTEM_MSG},
                {"role":"user","content":self.prompter(inputs)}],
            temperature=self.temperature,
            max_tokens=512,
            top_p=self.top_p,
            frequency_penalty=0,
            presence_penalty=0,
            n=1,
            logprobs=True
        )

        # 计算生成程序的置信度概率，并去掉模型可能输出的 markdown 代码围栏和首尾换行
        prob = self.compute_prob(response)
        text = response.choices[0].message.content or ''
        prog = '\n'.join(
            ln for ln in text.splitlines() if not ln.strip().startswith('```')).strip('\n')
        return prog, prob
    