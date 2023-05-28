from django.shortcuts import render
from .forms import ImageUploadForm
import torch
from torchvision import transforms
from PIL import Image
from . import model_load


def index(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_upload = form.save()
            image_path = image_upload.image.path
            result = predict_image(image_path)
            return render(request, 'corn_recognition/result.html', {'result': result})
    else:
        form = ImageUploadForm()

    return render(request, 'corn_recognition/index.html', {'form': form})


def predict_image(image_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_load.get_model()
    model.to(device)

    img = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_t = transform(img)
    img_t = img_t.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(img_t)
        _, predicted = torch.max(output, 1)
        class_name = model_load.class_names[predicted.item()]

    return class_name
