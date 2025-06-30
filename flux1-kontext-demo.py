import marimo

__generated_with = "0.14.7"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
    # Flux.1 Kontext Demo

    This is a demo of the image-editing model [Flux.1 Kontext [dev]](https://bfl.ai/announcements/flux-1-kontext).

    Upload an image to edit and provide edit instructions. Generation takes about 1 minute.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # from huggingface_hub import login
    # login()
    return


@app.cell
def _(mo):
    import accelerate
    # from diffusers import FluxKontextPipeline
    from pipeline_flux_kontext import FluxKontextPipeline  # temp fix until diffusers package is updated
    from diffusers.utils import load_image
    # import protobuf
    import sentencepiece
    import torch
    import transformers

    with mo.status.spinner(subtitle="Loading model ...") as _spinner:
        pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU
        # pipe = pipe.to("cuda")
    return (pipe,)


@app.cell
def _(mo):
    file_upload = mo.ui.file(
        filetypes=[".png", ".jpg"],
        multiple=False,
        kind="area",
        label="Select or drop an image to edit (PNG, JPEG)",
    )

    file_upload
    return (file_upload,)


@app.cell
def _(file_upload):
    # Indicate whether file has been uploaded
    awaiting_file_upload = len(file_upload.value) == 0
    return (awaiting_file_upload,)


@app.cell
def _(awaiting_file_upload, file_upload, mo):
    from io import BytesIO
    from PIL import Image

    mo.stop(awaiting_file_upload)  # Wait for image

    with mo.status.spinner(subtitle="Uploading image ...") as _spinner:
        image = file_upload.value[0]
        input_image = Image.open(BytesIO(image.contents))

    input_image
    return (input_image,)


@app.cell
def _(awaiting_file_upload, mo):
    mo.stop(awaiting_file_upload)  # Wait for image

    text_area = mo.ui.text_area(
        label="Enter your edit prompt",
        value="Put everything on fire", 
        full_width=True,
    )

    form_prompt = mo.md(
        """
        {text_area} 
        """
    ).batch(text_area=text_area).form(
        submit_button_label="Generate",
        bordered=False,
        show_clear_button=False,
    )

    form_prompt
    return (form_prompt,)


@app.cell
def _(form_prompt, mo):
    mo.stop(form_prompt.value is None)  # Wait for prompt

    prompt = form_prompt.value["text_area"]

    prompt
    return (prompt,)


@app.cell
def _(input_image, mo, pipe, prompt):
    with mo.status.spinner(subtitle="Editing image (~1 minute) ...") as _spinner:
        output = pipe(
          image=input_image,
          prompt=prompt,
          guidance_scale=2.5
        ).images[0]

    output
    return


if __name__ == "__main__":
    app.run()
