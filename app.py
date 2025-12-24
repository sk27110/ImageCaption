import gradio as gr
import torch
from PIL import Image
import torchvision.transforms as T
import sys
import os
import torch

sys.path.append('/content/ImageCaption/src')

def get_models():
    from comet_ml import API
    from pathlib import Path
    from comet_ml import API
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    api = API(api_key="MbL2psOHT82Uc7ML5Cd7TSvmR")
    output_dir = Path("./from_experiment_assets")
    output_dir.mkdir(exist_ok=True)
    workspace = "-3767"
    project_name = "image_captioning"


    experiment_key = "1b034a86b05e4572abf6b0a68b95a324"
    exp = api.get_experiment(workspace, project_name, experiment_key)
    for asset in exp.get_asset_list():
        fname = asset["fileName"]
        if "best_model_latest" in fname:
            binary = exp.get_asset(asset["assetId"], return_type="binary")
            with open(output_dir / "clip_transformer_model.pth", "wb") as f:
                f.write(binary)

            print("✅ Скачан:", fname)

    experiment_key = "8c81ec6aa5fd437c9213a27a572b7fc3"
    exp = api.get_experiment(workspace, project_name, experiment_key)
    for asset in exp.get_asset_list():
        fname = asset["fileName"]
        if "best_model_latest" in fname:
            binary = exp.get_asset(asset["assetId"], return_type="binary")

            with open(output_dir / "resnet_lstm_model.pth", "wb") as f:
                f.write(binary)

            print("✅ Скачан:", fname)

    experiment_key = "8532d2ed1fe24e0fa88f14e3fe540a0c"
    exp = api.get_experiment(workspace, project_name, experiment_key)

    for asset in exp.get_asset_list():
        fname = asset["fileName"]
        if "best_model_latest" in fname:
            binary = exp.get_asset(asset["assetId"], return_type="binary")

            with open(output_dir / "resnet_transformer_model.pth", "wb") as f:
                f.write(binary)

            print("✅ Скачан:", fname)

    for asset in exp.get_asset_list():
      fname = asset["fileName"]
      if "tokenizer" in fname:
          binary = exp.get_asset(asset["assetId"], return_type="binary")

          with open(output_dir / fname, "wb") as f:
              f.write(binary)

          print("✅ Скачан:", fname)
    """
    Load the three trained models and tokenizer.
    Returns: list of models and tokenizer dictionaries
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_tokenizer = torch.load("/content/from_experiment_assets/tokenizer.pth",
                                      map_location=device)
    itos = checkpoint_tokenizer['itos']
    stoi = checkpoint_tokenizer['stoi']

    from clip_transformer_model import CLIPTransformerEncoderDecoder
    from resnet_transformer_model import ResNetTransformerEncoderDecoder
    from resnet_lstm_model import LSTMEncoderDecoder

    checkpoint = torch.load("/content/from_experiment_assets/resnet_transformer_model.pth",
                           map_location=device)
    resnet_transformer_model = ResNetTransformerEncoderDecoder(**checkpoint["model_args"]).to(device)
    resnet_transformer_model.load_state_dict(checkpoint["model_state_dict"])
    resnet_transformer_model.eval()

    checkpoint = torch.load("/content/from_experiment_assets/clip_transformer_model.pth",
                           map_location=device)
    clip_transformer_model = CLIPTransformerEncoderDecoder(**checkpoint["model_args"]).to(device)
    clip_transformer_model.load_state_dict(checkpoint["model_state_dict"])
    clip_transformer_model.eval()

    checkpoint = torch.load("/content/from_experiment_assets/resnet_lstm_model.pth",
                           map_location=device)
    resnet_lstm_model = LSTMEncoderDecoder(**checkpoint["model_args"]).to(device)
    resnet_lstm_model.load_state_dict(checkpoint["model_state_dict"])
    resnet_lstm_model.eval()

    return [resnet_lstm_model, resnet_transformer_model, clip_transformer_model, itos, stoi]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_lstm_model, resnet_transformer_model, clip_transformer_model, itos, stoi = get_models()

def generate_caption(image, model_name):
    """
    Generate caption for the input image using selected model.

    Args:
        image: PIL Image object
        model_name: string indicating which model to use

    Returns:
        Generated caption as string
    """
    if image is None:
        return "Please upload an image first!"

    try:
        if model_name == "CLIP Transformer":
            model = clip_transformer_model
        elif model_name == "LSTM":
            model = resnet_lstm_model
        elif model_name == "ResNet Transformer":
            model = resnet_transformer_model
        else:
            return "Please select a model!"

        img_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        img_tensor = img_transform(image).unsqueeze(0).to(device)

        gen_ids = model.generate_beam(
            img_tensor,
            max_len=25,
            beam_width=3,
            start_token=stoi["<START>"],
            end_token=stoi["<END>"]
        )[0]

        gen_tokens = [itos[int(idx)] for idx in gen_ids if int(idx) != stoi["<PAD>"]]
        clear_tokens = [token for token in gen_tokens if token not in {'<START>', '<END>'}]
        caption = " ".join(clear_tokens)

        return caption

    except Exception as e:
        return f"Error generating caption: {str(e)}"

def create_interface():

    model_choices = [
        "CLIP Transformer",  
        "LSTM",             
        "ResNet Transformer"
    ]

    with gr.Blocks(title="ImageCaption - Your Best img2text", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🖼️ ImageCaption")
        gr.Markdown("## Your best **img2text** tool for generating captions from images")

        with gr.Row():
            with gr.Column(scale=1):

                image_input = gr.Image(
                    type="pil",
                    label="Upload Image",
                    sources=["upload"],
                    image_mode="RGB"
                )

                model_selector = gr.Dropdown(
                    choices=model_choices,
                    label="Select Model",
                    value="CLIP Transformer",
                    info="Choose which model to use for caption generation"
                )

                generate_btn = gr.Button("✨ Etogo na dopsu", variant="primary")

            with gr.Column(scale=1):
                caption_output = gr.Textbox(
                    label="Generated Caption",
                    interactive=False,
                    show_copy_button=True
                )

                image_display = gr.Image(
                    label="Preview",
                    interactive=False,
                    height=300
                )

        image_input.change(
            fn=lambda img: img,
            inputs=image_input,
            outputs=image_display
        )

        generate_btn.click(
            fn=generate_caption,
            inputs=[image_input, model_selector],
            outputs=caption_output
        )

        # gr.Examples(
        #     examples=[
        #         ["/content/sample_image1.jpg", "CLIP Transformer"],
        #         ["/content/sample_image2.jpg", "LSTM"],
        #     ],
        #     inputs=[image_input, model_selector],
        #     outputs=[caption_output, image_display],
        #     fn=generate_caption,
        #     cache_examples=False,
        #     label="Try these examples"
        # )

        gr.Markdown("---")
        gr.Markdown(
            """
            <div style='text-align: center'>
            <p>Powered by PyTorch & Gradio | Made with ❤️ for image understanding</p>
            <p>Supports: JPG, PNG, BMP formats</p>
            </div>
            """
        )

    return demo

if __name__ == "__main__":
    demo = create_interface()

    demo.launch(
        debug=False,
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
