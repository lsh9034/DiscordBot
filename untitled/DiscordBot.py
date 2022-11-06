import discord
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import torch
import torchvision.transforms as transforms
import json
import requests
import warnings
from efficientnet_pytorch import EfficientNet
from matplotlib import cm

warnings.filterwarnings('ignore')
print(torch.__version__)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')
import matplotlib.pyplot as plt

f = open('discord_bot_token.txt','r')
discord_token = f.read()

# discord Client class를 생성합니다.
intents = discord.Intents.all()
client = discord.Client(intents=intents)
model = EfficientNet.from_pretrained('efficientnet-b0')
def predict(img):

    # Preprocess image
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
    img = tfms(img).unsqueeze(0)
    print(img.shape)  # torch.Size([1, 3, 224, 224])

    # Load ImageNet class names
    labels_map = json.load(open('labels_map.txt'))
    labels_map = [labels_map[str(i)] for i in range(1000)]

    # Classify
    model.eval()
    with torch.no_grad():
        outputs = model(img)

    # Print predictions
    ans =''
    print('-----')
    for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
        prob = torch.softmax(outputs, dim=1)[0, idx].item()
        print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob * 100))
        ans = labels_map[idx]
        break
    return ans
# event decorator를 설정하고 on_ready function을 할당해줍니다.
@client.event
async def on_ready():  # on_ready event는 discord bot이 discord에 정상적으로 접속했을 때 실행됩니다.
    print('We have logged in as {}'.format(client))
    print('Bot name: {}'.format(client.user.name))  # 여기서 client.user는 discord bot을 의미합니다. (제가 아닙니다.)
    print('Bot ID: {}'.format(client.user.id))  # 여기서 client.user는 discord bot을 의미합니다. (제가 아닙니다.)

# event decorator를 설정하고 on_message function을 할당해줍니다.
@client.event
async def on_message(message):
    # message란 discord 채널에 올라오는 모든 message를 의미합니다.
    # 따라서 bot이 보낸 message도 포함이되죠.
    # 아래 조건은 message의 author가 bot(=clinet.user)이라면 그냥 return으로 무시하라는 뜻입니다.
    if message.author == client.user:
        return
    # message를 보낸 사람이 bot이 아니라면 message가 hello로 시작하는 경우 채널에 Hello!라는 글자를 보내라는 뜻입니다.
    elif message.content.startswith('hello'):
        await message.channel.send('Hello!')
    elif message.content == 'ImageNet':
        url = message.attachments[0].url
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        ans = predict(img)
        await message.channel.send(ans)
# 위에서 설정한 client class를 token으로 인증하여 실행합니다.
client.run(discord_token)
