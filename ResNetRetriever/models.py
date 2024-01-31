import torch
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def load_pretrained_model():
#     model = models.resnet50(pretrained=True).to(device)
#     model = model.eval()
#     return model

# model = load_pretrained_model()

def load_pretrained_model():
    model = models.resnet50(pretrained=True)
    
    # Remove the last fully connected layer to get higher dimensional feature
    modules = list(model.children())[:-1]  # remove last FC layer
    model = torch.nn.Sequential(*modules).to(device)
    
    model = model.eval()
    return model

model = load_pretrained_model()
