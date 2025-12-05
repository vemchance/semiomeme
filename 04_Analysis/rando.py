import torch
ckpt = torch.load(r'X:\PhD\SemioMeme_Graph\data\corpus_data\text_model.pth', map_location='cpu')
print(ckpt['config'])

for key in ckpt['model_state_dict'].keys():
    print(key)

ckpt['config']['num_hidden_layers'] = 2
torch.save(ckpt, r'X:\PhD\SemioMeme_Graph\data\corpus_data\text_model.pth')