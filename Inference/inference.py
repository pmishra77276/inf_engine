import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from torch import nn
import time

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
# def get_layers(model,drop=2):
#     device_memory=(torch.cuda.get_device_properties('cuda').total_memory)/(1024**3)
#     l=model.model.layers[0]
#     num_params = sum(p.numel() for p in l.parameters())
#     param_size_bytes = num_params * 4
#     param_size_GB = param_size_bytes / (1024 ** 3)
#     return int(device_memory/param_size_GB)-drop

# class Module(nn.Module):
#     def __init__(self, layers, i, j):
#         super().__init__()
#         self.model = nn.ModuleList(layers[i:j])

#     def forward(self, x, cos, sin,attention_mask):
#         for layer in self.model:
#             x = layer(x, position_embeddings=(cos, sin))[0]
#         return x
# # layers=model.model.layers

# def compute(model, x, layers_per_chunk, cos, sin,k,max_new_tokens,attention_mask):
#     total_layers = len(model.model.layers)
#     if layers_per_chunk >= total_layers:
#         i=0
#         l=total_layers
#         m=Module(model.model.layers,i,i+l).to('cuda')
#         x=m(x.to('cuda'),cos.to('cuda'),sin.to('cuda'),attention_mask.to('cuda'))
#         if(k==max_new_tokens-1):
#             m=m.to('cpu')
#             torch.cuda.empty_cache()
#         return x
#     i = 0
#     rem = total_layers
#     while rem > 0:
#         l = min(layers_per_chunk, rem)
#         m = Module(model.model.layers, i, i + l).to('cuda')
#         x = m(x.to('cuda'), cos.to('cuda'), sin.to('cuda'),attention_mask)
#         # if(k==max_new_tokens-1):
#         #     m=m.to('cpu')
#         #     torch.cuda.empty_cache()
#         m=m.to('cpu')
#         torch.cuda.empty_cache()
#         i += l
#         rem -= l

#     return x
# def model_inference(question,tokens=250):
#     prompt=tok(question,return_tensors='pt').to('cpu')
#     input_ids=prompt['input_ids'].to('cuda')
#     attention_mask=prompt['attention_mask']
#     del prompt
#     for i in range(tokens):
#         torch.cuda.empty_cache()
#         model.model.embed_tokens.to('cuda')
#         x = model.model.embed_tokens(input_ids)
#         # input_ids.to('cpu')  for better gpu memory 
#         # torch.cuda.empty_cache()
#         # model.model.embed_tokens.to('cpu')
#         # torch.cuda.empty_cache()
#         seq_len = x.shape[1]
#         batch_size = x.shape[0]
#         position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand(batch_size, -1)
#         rotary_emb = model.model.rotary_emb
#         cos, sin = rotary_emb(x=x,position_ids=position_ids)
#         cnt=0
#         with torch.no_grad():
#             l1=get_layers(model)
#             x=compute(model,x,1,cos,sin,i,tokens,attention_mask)
#             model.model.norm.to('cuda')
#             x=model.model.norm(x)
#             # model.model.norm.to('cpu')
#             # torch.cuda.empty_cache()
#             model.lm_head.to('cuda')
#             x=model.lm_head(x)
#             # model.lm_head.to('cpu')
#             # torch.cuda.empty_cache()
        
#         x1=x[:,-1,:]
#         x1=torch.argmax(x1,dim=-1,keepdim=True)
#         print(tok.decode(x1[0]),end="")
#         input_ids=torch.cat([input_ids,x1],dim=-1) 
def get_layers(model,drop=2):
    device_memory=(torch.cuda.get_device_properties('cuda').total_memory)/(1024**3)
    l=model.model.layers[0]
    num_params = sum(p.numel() for p in l.parameters())
    param_size_bytes = num_params * 2
    param_size_GB = param_size_bytes / (1024 ** 3)
    return (int(device_memory/param_size_GB)-drop)//2

class Module(nn.Module):
    def __init__(self, layers, i, j):
        super().__init__()
        self.model = nn.ModuleList(layers[i:j])
        # print(self.model.device)
       
    @torch.compile
    def forward(self, x, cos, sin,attention_mask):
        for layer in self.model:
            # print(layer.device)
            x = layer(x, position_embeddings=(cos, sin))[0]
        return x



def compute(model_instance, x, layers_per_chunk, cos, sin, k, max_new_tokens, attention_mask):
    total_layers = len(model_instance.model.layers)
    stream_compute = torch.cuda.Stream()
    stream_transfer = torch.cuda.Stream()
    stream_transfer2 = torch.cuda.Stream()
    stream_offload = torch.cuda.Stream()
    
    if layers_per_chunk >= total_layers:
        i = 0
        l = total_layers
        m = Module(model_instance.model.layers, i, i + l).to('cuda',non_blocking=True)
        x = m(x.to('cuda'), cos.to('cuda'), sin.to('cuda'), attention_mask.to('cuda'))
        if k == max_new_tokens - 1:
            m = m.to('cpu',non_blocking=True)
            torch.cuda.empty_cache()
        return x

    i = 0
    rem = total_layers
    m_current = None
    m_next = None
    m_next2 = None
    m_previous = None

    # Load first two chunks
    if rem > 0:
        l_first = min(layers_per_chunk, rem)
        with torch.cuda.stream(stream_transfer):
            m_next = Module(model_instance.model.layers, i, i + l_first).to('cuda', non_blocking=True)
        i += l_first
        rem -= l_first
        
        if rem > 0:
            l_second = min(layers_per_chunk, rem)
            with torch.cuda.stream(stream_transfer2):
                m_next2 = Module(model_instance.model.layers, i, i + l_second).to('cuda', non_blocking=True)
            i += l_second
            rem -= l_second

    while True:
        # Wait for transfer to complete before starting computation
        stream_compute.wait_stream(stream_transfer)
        
        # Move pointers
        m_previous = m_current
        m_current = m_next
        m_next = m_next2
        m_next2 = None

        # Compute on current chunk
        with torch.cuda.stream(stream_compute):
            with torch.no_grad():
                x = m_current(x.to('cuda', non_blocking=True),
                              cos.to('cuda', non_blocking=True),
                              sin.to('cuda', non_blocking=True),
                              attention_mask.to('cuda', non_blocking=True))

        # Offload previous chunk to CPU (if exists)
        if m_previous is not None:
            stream_offload.wait_stream(stream_compute)  # Wait for computation to finish before offloading
            with torch.cuda.stream(stream_offload):
                m_previous = m_previous.to('cpu', non_blocking=True)
                del m_previous
            torch.cuda.empty_cache()

        # If this is the last chunk, break
        if rem == 0:
            stream_compute.synchronize()
            # Offload the final chunk
            with torch.cuda.stream(stream_offload):
                m_current = m_current.to('cpu', non_blocking=True)
            stream_offload.synchronize()
            torch.cuda.empty_cache()
            break

        # Load next chunk (2 steps ahead)
        if rem > 0:
            time.sleep(0.01) 
            l_next2 = min(layers_per_chunk, rem)
            with torch.cuda.stream(stream_transfer2):
                m_next2 = Module(model_instance.model.layers, i, i + l_next2).to('cuda', non_blocking=True)
            time.sleep(0.01)
            i += l_next2
            rem -= l_next2

    return x

def model_inference(question,tokens=250):
    prompt=tok(question,return_tensors='pt').to('cpu',non_blocking=True)
    input_ids=prompt['input_ids'].to('cuda',non_blocking=True)
    attention_mask=prompt['attention_mask']
    model.model.embed_tokens.to('cuda',non_blocking=True)
    model.model.norm.to('cuda',non_blocking=True)
    model.lm_head.to('cuda',non_blocking=True)
    x = model.model.embed_tokens(input_ids)
    rotary_emb = model.model.rotary_emb
   
    del prompt
    for k in range(tokens):
        torch.cuda.empty_cache()
        # model.model.embed_tokens.to('cuda',non_blocking=True)
        # if i!=0:
        x = model.model.embed_tokens(input_ids)
        seq_len = x.shape[1]
        batch_size = x.shape[0]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand(batch_size, -1)
       
        cos, sin = rotary_emb(x=x,position_ids=position_ids)
        # input_ids.to('cpu')
        # torch.cuda.empty_cache()
        # model.model.embed_tokens.to('cpu')
        # torch.cuda.empty_cache()
       
        cnt=0
        with torch.no_grad():
            l1=get_layers(model,7)
            x=compute(model,x,3,cos,sin,k,tokens,attention_mask)
            # model.model.norm.to('cuda',non_blocking=True)
            x=model.model.norm(x)
            # model.model.norm.to('cpu',non_blocking=True)
            torch.cuda.empty_cache()
            # model.lm_head.to('cuda',non_blocking=True)
            x=model.lm_head(x)
            # model.lm_head.to('cpu',non_blocking=True)
            torch.cuda.empty_cache()
        
        x1=x[:,-1,:]
        x1=torch.argmax(x1,dim=-1,keepdim=True)
        print(tok.decode(x1[0]),end="")
        input_ids=torch.cat([input_ids,x1],dim=-1)
        # x1=x.argmax(-1)
        # print(tok.decode(x1[0]),end="\n")
        # question=question+" "+tok.decode(x1[0])
    input_ids.to('cpu')
    model.model.norm.to('cpu',non_blocking=True)
    torch.cuda.empty_cache()
    model.lm_head.to('cpu',non_blocking=True)
    torch.cuda.empty_cache()
tok=AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model=AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B",torch_dtype=torch.bfloat16,use_cache=True)
layers=model.model.layers
# question="Hi"
while True:
    question="Hi"
    start=time.time()
    model_inference(question,10)
    print(time.time()-start)



