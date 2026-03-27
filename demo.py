# demo.py
"""
Gradio interactive demo for ABSA.
Run: python demo.py
"""

import gradio as gr
from cli import predict_single

ASPECT_CHOICES = [
    "battery", "screen", "sound", "performance",
    "price", "usability", "design", "connectivity",
    "build_quality", "overall",
]

MODEL_CHOICES = ["extended", "bert", "baseline"]

SENTIMENT_EMOJI = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral":  "Neutral",
}


def run_prediction(review: str, aspect: str, model: str) -> str:
    if not review.strip():
        return "Please enter a review."
    if not aspect:
        return "Please select an aspect."
    result = predict_single(review, aspect, model)
    return SENTIMENT_EMOJI.get(result, result)


with gr.Blocks(title="ABSA Demo — COMP6713") as demo:
    gr.Markdown("# Aspect-Based Sentiment Analysis\nEnter an electronics product review and select an aspect to analyse.")

    with gr.Row():
        review_input = gr.Textbox(
            label="Review",
            placeholder="e.g. The battery life is amazing but the screen looks dull.",
            lines=3,
        )

    with gr.Row():
        aspect_input = gr.Dropdown(choices=ASPECT_CHOICES, label="Aspect", value="battery")
        model_input  = gr.Radio(choices=MODEL_CHOICES, label="Model", value="extended")

    submit_btn = gr.Button("Predict Sentiment")
    output_box = gr.Textbox(label="Sentiment")

    submit_btn.click(
        fn=run_prediction,
        inputs=[review_input, aspect_input, model_input],
        outputs=output_box,
    )

    gr.Examples(
        examples=[
            ["The battery lasts forever, easily 3 days!", "battery", "extended"],
            ["Screen resolution is terrible, looks pixelated.", "screen", "extended"],
            ["The price is reasonable for what you get.", "price", "bert"],
            ["Connectivity drops constantly, very unreliable wifi.", "connectivity", "baseline"],
        ],
        inputs=[review_input, aspect_input, model_input],
    )

if __name__ == "__main__":
    demo.launch(share=False)
