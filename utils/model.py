#%%
from transformers import ViTModel, ViTConfig
from PIL import Image
import requests
import torch


class ViTImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super(ViTImageEncoder, self).__init__()
        self.model = model
        self.to_img_descriptor = torch.nn.Identity()
    def forward(self, images):
        last_heddent_state = self.model(pixel_values=images).last_hidden_state
        outputs = self.to_img_descriptor(last_heddent_state)[:, 0]
        return outputs
    
def get_ViTImageEncoder(pretrained=True):
    if pretrained:
        model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    else:
        model = ViTModel(ViTConfig())
    return ViTImageEncoder(model)

if __name__ == '__main__':
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    model = get_ViTImageEncoder(False)
    with torch.no_grad():
        image_descriptor = model(image)
        
# %%
