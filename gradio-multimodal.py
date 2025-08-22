import datetime
import json
import os
import time
import gradio as gr
import torch
import logging
import threading
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor as QwenProcessor, TextIteratorStreamer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor as WhisperProcessor
from transformers.generation import GenerationConfig
import librosa
import numpy as np
from gtts import gTTS
import nltk
from PIL import Image
nltk.download('punkt', quiet=True)

print("Loading Qwen2.5-VL-7B-Instruct model and processor from local path...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "Qwen2.5-VL-7B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype="auto", device_map="auto"
)
processor = QwenProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"Qwen2.5-VL-7B-Instruct model loaded successfully on {DEVICE}")

LOGDIR = "logs"
if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

print("Loading Whisper model from local Hugging Face path...")
WHISPER_PATH = "whisper-medium"

whisper_processor = WhisperProcessor.from_pretrained(WHISPER_PATH)
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    WHISPER_PATH,
    device_map="auto",
    torch_dtype="auto"
)
whisper_model.config.forced_decoder_ids = None

def transcribe(audio_path):
    if not audio_path:
        return ''
    try:
        speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
        input_features = whisper_processor(
            speech_array, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(DEVICE)
        predicted_ids = whisper_model.generate(input_features)
        transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription[0]
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return "Error transcribing audio."

def text_to_speech(text, file_path, language):
    lang_code_map = {
        "English": "en",
        "Arabic": "ar",
        "Chinese": "zh-cn"
    }
    lang_code = lang_code_map.get(language, "en")
    try:
        audioobj = gTTS(text=text, lang=lang_code, slow=False)
        audioobj.save(file_path)
        return file_path
    except Exception as e:
        print(f"Error in TTS for language '{lang_code}': {e}")
        return None

logger = logging.getLogger("gradio_web_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

def process_vision_info(messages):
    image_inputs = []
    for message in messages:
        if message["role"] == "user":
            for item in message["content"]:
                if item["type"] == "image":
                    image_path = item["image"]
                    image = Image.open(image_path).convert("RGB")
                    image_inputs.append(image)
    return image_inputs, []

def convert_history_to_messages(history):
    messages = []
    for i in range(0, len(history), 2):
        user_turn = history[i]
        assistant_turn = history[i+1]
        content = []
        if user_turn.get("image_path"):
            content.append({"type": "image", "image": user_turn["image_path"]})
        content.append({"type": "text", "text": user_turn["content"]})
        messages.append({"role": "user", "content": content})
        if assistant_turn.get("content") is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_turn["content"]}]})
    return messages

def convert_history_to_gradio(history):
    gradio_chat = []
    for i in range(0, len(history), 2):
        user_turn = history[i]
        assistant_turn = history[i+1] if i + 1 < len(history) else {'content': None}
        user_content = user_turn.get('content', '')
        if user_turn.get('image_path'):
            image_path = user_turn['image_path']
            user_content = f"![User Upload](/gradio_api/file={image_path})\n\n{user_content}"
        assistant_content = assistant_turn.get('content', "") or ""
        gradio_chat.append([user_content, assistant_content])
    return gradio_chat

def add_text(history, text, image, request: gr.Request):
    if len(text) <= 0 and image is None:
        return history, convert_history_to_gradio(history), "", None, no_change_btn, no_change_btn
    image_path = None
    if image is not None:
        temp_dir = "/home/cdsw/temp_images"
        os.makedirs(temp_dir, exist_ok=True)
        rel_path = os.path.join(temp_dir, f"temp_{time.time()}.png")
        image_path = os.path.abspath(rel_path)
        image.save(image_path)
    history.append({'role': 'user', 'content': text, 'image_path': image_path})
    history.append({'role': 'assistant', 'content': None})
    chatbot_display = convert_history_to_gradio(history)
    return history, chatbot_display, "", None, disable_btn, disable_btn

def qwen_bot(history, temperature, top_p, max_new_tokens, language, request: gr.Request):
    messages = convert_history_to_messages(history)
    system_message = {
        "role": "system",
        "content": f"You are a helpful assistant. Please provide all your responses in {language}."
    }
    messages_with_system_prompt = [system_message] + messages
    text = processor.apply_chat_template(messages_with_system_prompt, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    processor_kwargs = {
        "text": [text],
        "images": image_inputs,
        "padding": True,
        "return_tensors": "pt"
    }
    if video_inputs:
        processor_kwargs["videos"] = video_inputs
    inputs = processor(**processor_kwargs).to(model.device)
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    history[-1]['content'] = ""
    for new_text in streamer:
        history[-1]['content'] += new_text
        chatbot_display = convert_history_to_gradio(history)
        yield history, chatbot_display, disable_btn, disable_btn
    thread.join()
    final_response = history[-1]['content']
    chatbot_display = convert_history_to_gradio(history)
    yield history, chatbot_display, enable_btn, enable_btn
    text_to_speech(final_response, "/home/cdsw/voicetmp.mp3", language)

def clear_history(request: gr.Request):
    return [], [], "", None, enable_btn, enable_btn

title_markdown = ("""
# Multimodal Chat 🤖
""")
block_css = "#buttons button { min-width: min(120px,100%); }"

def build_demo():
    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER, or upload an image.", container=False)

    with gr.Blocks(title="Qwen-VL", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State([])
        
        gr.Markdown(title_markdown)
        
        # **FIX:** Adjusted height to 20. Default alignment is left.
        gr.Image(value="/home/cdsw/clouderalogo.png", height=20, show_label=False, show_download_button=False, interactive=False, container=False)
        
        with gr.Row():
            with gr.Column(scale=3):
                imagebox = gr.Image(type="pil")
                language_selector = gr.Radio(
                    ["English", "Arabic", "Chinese"],
                    value="English",
                    label="Response Language 🌐"
                )
                audioinput = gr.Audio(sources=["microphone"], type="filepath", label="Voice Input 🎤")
                voice_btn = gr.Button("Transcribe 🎙️")
                with gr.Accordion("Parameters", open=False):
                    temperature = gr.Slider(minimum=0.0, maximum=1.5, value=0.7, step=0.1, interactive=True, label="Temperature")
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.8, step=0.1, interactive=True, label="Top P")
                    max_output_tokens = gr.Slider(minimum=256, maximum=2048, value=1024, step=64, interactive=True, label="Max output tokens")

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(elem_id="chatbot", label="Multimodal Chatbot", height=550, layout="panel")
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")
                with gr.Row(elem_id="buttons"):
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)

        btn_list = [regenerate_btn, clear_btn]
        voice_btn.click(fn=transcribe, inputs=audioinput, outputs=textbox)
        
        submit_inputs = [state, temperature, top_p, max_output_tokens, language_selector]
        
        textbox.submit(
            add_text, [state, textbox, imagebox], [state, chatbot, textbox, imagebox] + btn_list, queue=False
        ).then(
            qwen_bot, submit_inputs, [state, chatbot] + btn_list
        )
        submit_btn.click(
            add_text, [state, textbox, imagebox], [state, chatbot, textbox, imagebox] + btn_list, queue=False
        ).then(
            qwen_bot, submit_inputs, [state, chatbot] + btn_list
        )
        clear_btn.click(clear_history, None, [state, chatbot, textbox, imagebox] + btn_list, queue=False)
    return demo

if __name__ == "__main__":
    allowed_folder = "/home/cdsw/temp_images"
    os.makedirs(allowed_folder, exist_ok=True)
    logo_path = "/home/cdsw/stc.png"

    demo = build_demo()
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=8100,
        share=False,
        allowed_paths=[allowed_folder, logo_path]
    )
