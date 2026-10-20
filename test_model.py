
import torch
from model import CNNModel

device = torch.device("cpu")

model = CNNModel().to(device)

model.load_state_dict(
    torch.load(
        "tomato_disease_cnn.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Model loaded successfully!")
print("Device:", device)
print("Number of classes:", 10)

