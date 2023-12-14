# Get API Key from  https://makersuite.google.com/app/apikey

from PIL import Image, ImageEnhance, ImageOps
from typing import List, Tuple, Optional
import google.generativeai as genai
import gradio as gr
TITLE = """<h1 align="center">Gemini Pro and Pro Vision via API 🚀</h1>"""

def process_image(image: Image.Image) -> Image.Image:
    flipped_image = ImageOps.mirror(image)
    enhancer = ImageEnhance.Brightness(flipped_image)
    processed_image = enhancer.enhance(0.5)  
    return processed_image

def predict(
    google_key: str,
    text_prompt: str,
    # image_prompt: Optional[Image.Image],
    webcam_image: Optional[Image.Image],  # Added webcam image input
    temperature: float,
    chatbot: List[Tuple[str, str]]
) -> Tuple[str, List[Tuple[str, str]]]:
    if not google_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. "
            "Please follow the instructions in the README to set it up.")

    genai.configure(api_key=google_key)
    generation_config = genai.types.GenerationConfig(temperature=temperature)
    image_prompt = None
    if image_prompt is None and webcam_image is None:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(
            text_prompt,
            stream=True,
            generation_config=generation_config)
        response.resolve()
    else:
        model = genai.GenerativeModel('gemini-pro-vision')
        input_data = [text_prompt]
        if image_prompt:
            input_data.append(image_prompt)
        elif webcam_image and not image_prompt :
            input_data.append(process_image(webcam_image))
        response = model.generate_content(
            input_data,
            stream=True,
            generation_config=generation_config)
        response.resolve()

    chatbot.append((text_prompt, response.text))
    return "", chatbot


google_key_component = gr.Textbox(
    label="GOOGLE API KEY",
    value="",
    type="password",
    placeholder="...",
    info="You can get GOOGLE_API_KEY from https://makersuite.google.com/app/apikey ",
)

# image_prompt_component = gr.Image(type="pil", label="Image", scale=1)
# image_prompt_component = None
webcam_component = gr.Webcam(type="pil", label="Webcam", scale=1)  
chatbot_component = gr.Chatbot(label='Gemini', scale=2)
text_prompt_component = gr.Textbox(
    placeholder="Hi there!",
    label="Ask Gemini anything and press Enter"
)
run_button_component = gr.Button()
temperature_component = gr.Slider(
    minimum=0,
    maximum=1.0,
    value=0.5,
    step=0.05,
    label="Temperature",
    info="Controls the randomness of the output.")

inputs = [
    google_key_component,
    text_prompt_component,
    # image_prompt_component,
    webcam_component, 
    temperature_component,
    chatbot_component
]

with gr.Blocks() as demo:
    gr.HTML(TITLE)
    with gr.Column():
        google_key_component.render()
        with gr.Row():
            # image_prompt_component.render()
            webcam_component.render() 
            chatbot_component.render()
        text_prompt_component.render()
        run_button_component.render()
        with gr.Accordion("Parameters", open=False):
            temperature_component.render()

    run_button_component.click(
        fn=predict,
        inputs=inputs,
        outputs=[text_prompt_component, chatbot_component],
    )

    text_prompt_component.submit(
        fn=predict,
        inputs=inputs,
        outputs=[text_prompt_component, chatbot_component],
    )

demo.queue(max_size=99).launch(debug=True)
