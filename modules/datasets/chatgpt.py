from utils.utils import get_file_name
import openai

openai.api_base = 'https://api.openai-proxy.org/v1'
openai.api_key = 'sk-jrgIM6P2VqRkWIH1BcwfUCY1vulqfrsBaU0d7i7bFYFy502l'

class ChatGPT:
    def __init__(self, args):
        self.args = args
        if args.experiment == 1:
            self.model = "gpt-4"
            # self.model = "gpt-3.5-turbo"
        else:
            self.model = "gpt-4"
            # self.model = "gpt-3.5-turbo"

    def chat_once(self, user_input):
        # user_input = input("用户：")  # 获取用户输入
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": user_input},
            ],
            # 2024:1:6 谢若天师兄提示
            temperature=0.7,  # 调整随机性
            max_tokens=150,  # 输出的最大长度
            frequency_penalty=0.5,  # 减少重复
            presence_penalty=0.5  # 鼓励创新
        )
        model_reply = response['choices'][0]['message']['content']
        # print("助手：", model_reply)
        return model_reply

    def interaction_chat_once(self):
        user_input = input("用户：")  # 获取用户输入
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "user", "content": user_input},
            ]
        )
        model_reply = response['choices'][0]['message']['content']
        print("助手：", model_reply)

    def intreraction_chat_window(self):
        conversation_history = [
            {"role": "system", "content": ""},
        ]
        # 模拟交互，可以根据需要循环
        for _ in range(3):
            user_input = input("用户：")  # 获取用户输入
            user_message = {"role": "user", "content": user_input}  # 创建用户消息
            conversation_history.append(user_message)  # 将用户消息添加到对话历史

            # 向API发送请求，继续对话
            response = openai.ChatCompletion.create(model=self.model, messages=conversation_history)

            # 提取模型生成的回复
            model_reply = response['choices'][0]['message']['content']
            print("助手：" + model_reply)  # 打印模型回复
            assistant_message = {"role": "assistant", "content": model_reply}  # 创建助手消息
            conversation_history.append(assistant_message)  # 将助手消息添加到对话历史


class NAS_ChatGPT:
    def __init__(self, args):
        self.args = args
        self.gpt = ChatGPT(args)

    def get_device_more_info(self, input_device):
        prompt = self.contructing_cpu_promote(input_device)
        model_reply = self.gpt.chat_once(prompt)
        return model_reply

    def contructing_cpu_promote(self, input_device):
        """
            CPU
            CPU时钟频率::最大睿频频率::核心数::线程数::三级缓存::最大内存带宽
            CPU Clock Frequency, Maximum Turbo Boost Frequency, Number of Cores, Number of Threads, Level 3 Cache, Maximum Memory Bandwidth
            CPU_Clock_Frequency::Maximum_Turbo_Boost_Frequency::Number_of_Cores::Number_of_Threads::Level_3_Cache::Maximum_Memory_Bandwidth
        """
        pre_str = "You are now a search engines, and required to provide the inquired information of the given processer.\n"
        input_device = 'The processer is ' + input_device + '.\n'
        output_format = "The inquired information is : Maximum Turbo Boost Frequency, Number of Cores, Number of Threads, Level 3 Cache, Maximum Memory Bandwidth.\n \
                        And please output them in form of: Maximum_Turbo_Boost_Frequency::Number_of_Cores::Number_of_Threads::Level_3_Cache::Maximum_Memory_Bandwidth. \n  \
                        please output only the content in the form above, i.e., %.2lf GHz::%d cores::%d Threads::%.2lf MB::%.1lf GB/s\n, \
                        but no other thing else, no reasoning, no index.\n\n"
        prompt = pre_str + input_device + output_format
        # print(prompt)
        return prompt

    def contructing_gpu_promote(self, input_device):
        """
            GPU
            流处理器数量::核芯频率::显存::位宽
            Stream processor count, Core clock frequency, Video memory, Memory bus width
            Stream_processor_count::Core_clock_frequency::Video_memory::Memory_bus_width
        """
        pre_str = "You are now a search engines, and required to provide the inquired information of the given processer.\n"
        output_format = "The inquired information is : .\n \
                                And please output them in form of: Stream processor count, Core clock frequency, Video memory, Memory bus width\n \
                                Stream_processor_count::Core_clock_frequency::Video_memory::Memory_bus_width \
                                please output only the content in the form above,  \
                                i.e., Stream_processor_count::Core_clock_frequency::Video_memory::Memory_bus_width\n, \
                                but no other thing else, no reasoning, no index, only number.\n\n"
        prompt = pre_str + input_device + output_format
        return prompt

    def contructing_dsp_promote(self, input_device):
        """
            流处理器数量::核芯频率::显存::位宽
            Stream processor count, Core clock frequency, Video memory, Memory bus width
            Stream_processor_count::Core_clock_frequency::Video_memory::Memory_bus_width
        """
        pre_str = "You are now a search engines, and required to provide the inquired information of the given processer.\n"
        output_format = "The inquired information is : .\n \
                                And please output them in form of: Stream processor count, Core clock frequency, Video memory, Memory bus width\n \
                                Stream_processor_count::Core_clock_frequency::Video_memory::Memory_bus_width \
                                please output only the content in the form above,  \
                                i.e., Stream_processor_count::Core_clock_frequency::Video_memory::Memory_bus_width\n, \
                                but no other thing else, no reasoning, no index, only number.\n\n"
        prompt = pre_str + input_device + output_format
        return prompt



