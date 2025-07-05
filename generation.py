# This file can use after yo train the model.
model = DiffusionModel().to(Config.device)
model.load_state_dict(torch.load("saved_models/model_epoch_30.pth"))


custom_prompts = [
    "a dragon flying over mountains",
    "cyberpunk street at night",
    "an astronaut riding a horse"
]


generate_images(model, diffusion, dataset.tokenizer, dataset.text_model, custom_prompts)
