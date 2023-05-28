import torch
from torchvision import models

class_names = ['玉米灰斑病', '玉米锈病', '玉米大斑病', '健康玉米']


def get_model():
    model_ft = models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, len(class_names))
    model_ft.load_state_dict(torch.load('D://Programme//PyProgramme//PycharmProgramme//corn_disease_recognition//best_model_weights.pth', map_location=torch.device('cpu')))
    return model_ft
