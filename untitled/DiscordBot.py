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
from discord.ext import commands
from discord import app_commands


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
tree = app_commands.CommandTree(client)

model = EfficientNet.from_pretrained('efficientnet-b0')

bot = commands.Bot(command_prefix='!', intents = discord.Intents.all(), application_id = 1038780297883435119)

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
    await client.wait_until_ready()
    await tree.sync(guild=discord.Object(id=1038402429735149618))

@tree.command(name='hello', description="bot answer Hello + name", guild= discord.Object(id=1038402429735149618))
async def helloworld(interaction: discord.Interaction, name: str):
    await interaction.response.send_message(f"Hello {name}! It was made by Discord.py")

@tree.command(name='efficient-net', description="EfficientNet classify class of image", guild= discord.Object(id=1038402429735149618))
async def helloworld(interaction: discord.Interaction, image: discord.Attachment):
    url = image.url
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    ans = predict(img)
    await interaction.response.send_message(ans)


# 위에서 설정한 client class를 token으로 인증하여 실행합니다.
client.run(discord_token)
