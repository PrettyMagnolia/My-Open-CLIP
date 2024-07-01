import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='/home/yifei/code/Open_CLIP/src/logs/2024_07_01-16_47_42-model_RN50-lr_0.001-b_128-j_8-p_amp/checkpoints/epoch_30.pt')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('RN50')

image = preprocess(Image.open("/data/csq/Caption_Datasets/Flickr_30k/flickr30k-images/1000092795.jpg")).unsqueeze(0)
text = tokenizer(["Two young , White males are outside near many bushes", "a dog", "a cat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]