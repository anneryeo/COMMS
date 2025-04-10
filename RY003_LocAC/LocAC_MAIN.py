''' 
pip install the following packages used via the python terminal:

pip install gradio
'''

import os #*
import gradio as gr #*
import openai #*
import torch #*
from transformers import GPT2LMHeadModel, GPT2Tokenizer #*
from openai import OpenAI
from openai.types import ChatModel #rs
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionAudio,
    ChatCompletionAudioParam,
    ChatCompletionChunk,
    ChatCompletionDeleted,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionModality,
    ChatCompletionRole,
    ChatCompletionStoreMessage,
    ChatCompletionSystemMessageParam,
    ChatCompletionTokenLogprob,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionReasoningEffort,
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
messages = []

def CustomChatGPT(user_input):
    messages.append({"role": "user", "content": user_input})

    input_text = " ".join([msg['content'] for msg in messages if msg['role'] == 'user'])
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    try:
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=100,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                early_stopping=True
            )

        ChatGPT_reply = tokenizer.decode(output[0], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": ChatGPT_reply})
        return ChatGPT_reply
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

demo = gr.Interface(
    fn=CustomChatGPT,
    inputs="text",
    outputs="text",
    title="Career Advisor",
    description="The Ikigai philosophy is rooted in understanding personal passions, skills, market needs, and financial viability, making it a robust framework for career guidance.",
    theme="default",
    css="body {background-color: #f0f0f0;} .output_text {background-color: #ffffff; border-radius: 5px; padding: 10px;}",
    allow_flagging="never",
    examples=[
        ["What should I do with my life?"],
        ["How can I combine my skills and passions into a career?"],
        ["What are some potential career paths based on my interests?"],
        ["How do I find a job that aligns with my values?"],
        ["What are the market demands for my skills?"],
    ]
)

demo.launch() #share=True later